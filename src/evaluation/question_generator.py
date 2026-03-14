from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from evaluation.qa_logic import nodes_by_type
from graph.graph_utils import date_distance_days, trend_label


CATEGORY_TARGETS = {
    "temporal_proximity": 100,
    "causal_chain": 100,
    "longitudinal_trend": 100,
    "absence_detection": 100,
    "multi_hop_reasoning": 100,
}


def _dedupe_node_ids(node_ids: list[str]) -> list[str]:
    seen = set()
    output = []
    for node_id in node_ids:
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        output.append(node_id)
    return output


def _support_bundle(graph: nx.DiGraph, *node_ids: str) -> list[str]:
    bundle = []
    for node_id in node_ids:
        if not node_id or node_id not in graph:
            continue
        bundle.append(node_id)
        encounter_id = graph.nodes[node_id].get("encounter_id", "")
        if encounter_id:
            enc_node_id = f"enc:{graph.graph['patient_id']}:{encounter_id}"
            if enc_node_id in graph:
                bundle.append(enc_node_id)
    return _dedupe_node_ids(bundle)


def _encounters(graph: nx.DiGraph) -> list[tuple[str, dict]]:
    return nodes_by_type(graph, "Encounter")


def _lab_series(graph: nx.DiGraph) -> dict[str, list[str]]:
    series: dict[str, list[str]] = {}
    for node_id, data in nodes_by_type(graph, "Lab_Result"):
        series.setdefault(data["test_name"], []).append(node_id)
    return {name: ids for name, ids in series.items() if len(ids) >= 2}


def _first_outgoing(graph: nx.DiGraph, node_id: str, edge_type: str) -> list[str]:
    return [
        neighbor
        for _, neighbor, edge_data in graph.out_edges(node_id, data=True)
        if edge_data.get("edge_type") == edge_type
    ]


def temporal_candidates(graphs: dict[int, nx.DiGraph]) -> list[dict]:
    candidates = []
    for patient_id, graph in graphs.items():
        for proc_id, proc in nodes_by_type(graph, "Procedure"):
            anchor_id = f"enc:{patient_id}:{proc['encounter_id']}"
            if anchor_id not in graph:
                continue
            anchor = graph.nodes[anchor_id]
            actual_days = date_distance_days(anchor["date"], proc["date"]) or 0
            for window in sorted({max(actual_days, 0), max(actual_days - 1, 0)}):
                answer = "Yes" if actual_days <= window else "No"
                supporting_node_ids = _support_bundle(graph, anchor_id, proc_id)
                candidates.append(
                    {
                        "category": "temporal_proximity",
                        "patient_id": patient_id,
                        "text": (
                            f"Did patient {patient_id} receive {proc['description']} within {window} days "
                            f"of the encounter on {anchor['date']}?"
                        ),
                        "ground_truth": answer,
                        "supporting_node_ids": supporting_node_ids,
                        "metadata": {
                            "anchor_node_id": anchor_id,
                            "procedure_node_id": proc_id,
                            "procedure_description": proc["description"],
                            "window_days": window,
                            "anchor_date": anchor["date"],
                            "search_end_date": proc["date"],
                        },
                    }
                )
    return candidates


def causal_candidates(graphs: dict[int, nx.DiGraph]) -> list[dict]:
    candidates = []
    for patient_id, graph in graphs.items():
        for img_id, img in nodes_by_type(graph, "Imaging_Report"):
            followups = _first_outgoing(graph, img_id, "RESULTED_IN")
            descriptions = [
                graph.nodes[node_id].get("description", graph.nodes[node_id].get("summary", ""))
                for node_id in followups
            ]
            answer = "; ".join(descriptions) if descriptions else "No additional workup was documented."
            supporting_node_ids = _support_bundle(graph, img_id, *(followups[:1]))
            candidates.append(
                {
                    "category": "causal_chain",
                    "patient_id": patient_id,
                    "text": (
                        f"What diagnostic workup followed the abnormal {img['modality']} of the {img['body_part']} "
                        f"on {img['date']} for patient {patient_id}?"
                    ),
                    "ground_truth": answer,
                    "supporting_node_ids": supporting_node_ids,
                    "metadata": {
                        "anchor_node_id": img_id,
                        "followup_node_ids": followups,
                    },
                }
            )
    return candidates


def trend_candidates(graphs: dict[int, nx.DiGraph]) -> list[dict]:
    candidates = []
    for patient_id, graph in graphs.items():
        for test_name, node_ids in _lab_series(graph).items():
            values = [graph.nodes[node_id]["value"] for node_id in node_ids]
            label = trend_label(values)
            start = graph.nodes[node_ids[0]]
            end = graph.nodes[node_ids[-1]]
            answer = (
                f"{test_name} {label} from {start['value']}{start['unit']} on {start['date']} "
                f"to {end['value']}{end['unit']} on {end['date']}."
            )
            candidates.append(
                {
                    "category": "longitudinal_trend",
                    "patient_id": patient_id,
                    "text": (
                        f"How did patient {patient_id}'s {test_name} change between {start['date']} "
                        f"and {end['date']}?"
                    ),
                    "ground_truth": answer,
                    "supporting_node_ids": _support_bundle(graph, node_ids[0], node_ids[-1]),
                    "metadata": {
                        "test_name": test_name,
                        "lab_node_ids": node_ids,
                        "start_date": start["date"],
                        "end_date": end["date"],
                    },
                }
            )
    candidates.sort(key=lambda item: (item["patient_id"], item["metadata"]["test_name"]))
    return candidates


def absence_candidates(graphs: dict[int, nx.DiGraph]) -> list[dict]:
    candidates = []
    for patient_id, graph in graphs.items():
        imaging = nodes_by_type(graph, "Imaging_Report")
        for index, (img_id, img) in enumerate(imaging):
            followups = []
            for later_id, later in imaging[index + 1 :]:
                delta = date_distance_days(img["date"], later["date"])
                if delta is not None and delta <= 90:
                    followups.append(later_id)
            answer = (
                f"Yes, follow-up imaging was documented on {graph.nodes[followups[0]]['date']}."
                if followups
                else "No follow-up imaging was documented in the requested window."
            )
            candidates.append(
                {
                    "category": "absence_detection",
                    "patient_id": patient_id,
                    "text": (
                        f"Was there any follow-up imaging within 90 days after the imaging study on {img['date']} "
                        f"for patient {patient_id}?"
                    ),
                    "ground_truth": answer,
                    "supporting_node_ids": _support_bundle(graph, img_id, *(followups[:1])),
                    "metadata": {
                        "anchor_node_id": img_id,
                        "followup_imaging_ids": followups,
                        "anchor_date": img["date"],
                        "search_end_date": "9999-12-31" if not followups else graph.nodes[followups[0]]["date"],
                    },
                }
            )
    return candidates


def multi_hop_candidates(graphs: dict[int, nx.DiGraph]) -> list[dict]:
    candidates = []
    for patient_id, graph in graphs.items():
        encounters = _encounters(graph)
        if len(encounters) < 2:
            continue
        for start_index in range(len(encounters) - 1):
            initial_id, initial = encounters[start_index]
            final_id, final = encounters[min(start_index + 1, len(encounters) - 1)]
            final_dx_ids = _first_outgoing(graph, final_id, "DIAGNOSED_WITH")
            if not final_dx_ids:
                continue
            final_note_ids = _first_outgoing(graph, final_id, "DOCUMENTED_IN")
            final_dx = graph.nodes[final_dx_ids[0]]["description"]
            candidates.append(
                {
                    "category": "multi_hop_reasoning",
                    "patient_id": patient_id,
                    "text": (
                        f"What was the final diagnosis reached after the problem trajectory that began in the "
                        f"encounter on {initial['date']} and continued to the later encounter on {final['date']} "
                        f"for patient {patient_id}?"
                    ),
                    "ground_truth": final_dx,
                    "supporting_node_ids": _support_bundle(graph, initial_id, final_id, final_dx_ids[0]),
                    "metadata": {
                        "initial_encounter_id": initial_id,
                        "final_encounter_id": final_id,
                        "final_diagnosis_node_id": final_dx_ids[0],
                        "final_note_id": final_note_ids[0] if final_note_ids else "",
                    },
                }
            )
    return candidates


def _expand_to_target(candidates: list[dict], target: int, category: str) -> list[dict]:
    if not candidates:
        return []
    expanded = []
    for index in range(target):
        source = candidates[index % len(candidates)].copy()
        source["id"] = f"{category}-{index:03d}"
        expanded.append(source)
    return expanded


def generate_question_set(graphs: dict[int, nx.DiGraph]) -> list[dict]:
    category_builders = {
        "temporal_proximity": temporal_candidates,
        "causal_chain": causal_candidates,
        "longitudinal_trend": trend_candidates,
        "absence_detection": absence_candidates,
        "multi_hop_reasoning": multi_hop_candidates,
    }
    questions = []
    for category, builder in category_builders.items():
        candidates = builder(graphs)
        questions.extend(_expand_to_target(candidates, CATEGORY_TARGETS[category], category))
    return questions


def save_questions(questions: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(questions, handle, indent=2)
