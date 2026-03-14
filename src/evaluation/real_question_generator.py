from __future__ import annotations

import networkx as nx

from evaluation.qa_logic import nodes_by_type
from graph.graph_utils import date_distance_days


def _dedupe_node_ids(node_ids: list[str]) -> list[str]:
    seen = set()
    output = []
    for node_id in node_ids:
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        output.append(node_id)
    return output


def _note_links(graph: nx.DiGraph, node_id: str) -> list[str]:
    linked = []
    for neighbor in graph.predecessors(node_id):
        if graph.nodes[neighbor].get("node_type") == "Clinical_Note":
            linked.append(neighbor)
    for neighbor in graph.successors(node_id):
        if graph.nodes[neighbor].get("node_type") == "Clinical_Note":
            linked.append(neighbor)
    return _dedupe_node_ids(linked)


def _encounter_link(graph: nx.DiGraph, node_id: str) -> str:
    encounter_id = graph.nodes[node_id].get("encounter_id", "")
    if encounter_id in graph:
        return str(encounter_id)
    return ""


def _support_bundle(graph: nx.DiGraph, *node_ids: str) -> list[str]:
    bundle = []
    for node_id in node_ids:
        if not node_id or node_id not in graph:
            continue
        bundle.append(node_id)
        encounter_id = _encounter_link(graph, node_id)
        if encounter_id:
            bundle.append(encounter_id)
        bundle.extend(_note_links(graph, node_id))
    return _dedupe_node_ids(bundle)


def _patient_labels(graph: nx.DiGraph, node_type: str) -> list[str]:
    labels = []
    for _, data in nodes_by_type(graph, node_type):
        value = data.get("description", "")
        if value and value not in labels:
            labels.append(value)
    return labels


def temporal_candidates(graphs: dict[int, nx.DiGraph]) -> list[dict]:
    questions = []
    for patient_id, graph in graphs.items():
        encounters = {node_id: data for node_id, data in nodes_by_type(graph, "Encounter")}
        for proc_id, proc in nodes_by_type(graph, "Procedure"):
            enc_id = proc.get("encounter_id")
            if enc_id not in encounters:
                continue
            anchor = encounters[enc_id]
            support = _support_bundle(graph, enc_id, proc_id)
            if len(support) < 2:
                continue
            delta = date_distance_days(anchor["date"], proc["date"]) or 0
            for window in sorted({max(delta, 0), max(delta - 1, 0), max(delta + 30, 30)}):
                answer = "YES" if delta <= window else "NO"
                questions.append(
                    {
                        "category": "temporal_proximity",
                        "patient_id": patient_id,
                        "text": (
                            f"Did patient {patient_id} have {proc['description']} within {window} days "
                            f"of the encounter on {anchor['date']}?"
                        ),
                        "ground_truth": answer,
                        "answer_options": ["YES", "NO"],
                        "supporting_node_ids": support,
                    }
                )
    return questions


def causal_candidates(graphs: dict[int, nx.DiGraph]) -> list[dict]:
    questions = []
    for patient_id, graph in graphs.items():
        treatment_options = _patient_labels(graph, "Procedure") + ["NONE"]
        for img_id, img in nodes_by_type(graph, "Imaging_Report"):
            followups = [
                neighbor
                for _, neighbor, edge in graph.out_edges(img_id, data=True)
                if edge.get("edge_type") == "RESULTED_IN" and graph.nodes[neighbor]["node_type"] == "Procedure"
            ]
            if not followups:
                continue
            answer = graph.nodes[followups[0]]["description"]
            support = _support_bundle(graph, img_id, followups[0])
            if len(support) < 3:
                continue
            questions.append(
                {
                    "category": "causal_chain",
                    "patient_id": patient_id,
                    "text": (
                        f"What treatment or procedure followed the {img['modality']} documented on {img['date']} "
                        f"for patient {patient_id}?"
                    ),
                    "ground_truth": answer,
                    "answer_options": treatment_options,
                    "supporting_node_ids": support,
                }
            )
    return questions


def trend_candidates(graphs: dict[int, nx.DiGraph]) -> list[dict]:
    questions = []
    for patient_id, graph in graphs.items():
        stage_nodes = [
            (node_id, data)
            for node_id, data in nodes_by_type(graph, "Diagnosis")
            if data["description"].startswith("STAGE ")
        ]
        if len(stage_nodes) < 2:
            continue
        first_stage_id, first_stage = stage_nodes[0]
        last_stage_id, last_stage = stage_nodes[-1]
        if first_stage["date"] == last_stage["date"] and first_stage["description"] == last_stage["description"]:
            continue
        answer = f"{first_stage['description']} -> {last_stage['description']}"
        support = _support_bundle(graph, first_stage_id, last_stage_id)
        if len(support) < 3:
            continue
        questions.append(
            {
                "category": "longitudinal_trend",
                "patient_id": patient_id,
                "text": f"What stage trajectory is documented from the earliest to the latest staging note for patient {patient_id}?",
                "ground_truth": answer,
                "answer_options": [answer],
                "supporting_node_ids": support,
            }
        )
    return questions


def absence_candidates(graphs: dict[int, nx.DiGraph]) -> list[dict]:
    questions = []
    for patient_id, graph in graphs.items():
        imaging = nodes_by_type(graph, "Imaging_Report")
        encounters = nodes_by_type(graph, "Encounter")
        for index, (img_id, img) in enumerate(imaging):
            support = []
            followup_found = False
            for later_id, later in imaging[index + 1 :]:
                delta = date_distance_days(img["date"], later["date"])
                if delta is not None and 0 < delta <= 180:
                    followup_found = True
                    support = _support_bundle(graph, img_id, later_id)
                    break
            if not followup_found:
                horizon_support = [img_id]
                for enc_id, encounter in encounters:
                    delta = date_distance_days(img["date"], encounter["date"])
                    if delta is not None and 0 < delta <= 180:
                        horizon_support.extend(_support_bundle(graph, enc_id))
                support = _dedupe_node_ids(horizon_support)
                if len(support) < 2:
                    continue
            questions.append(
                {
                    "category": "absence_detection",
                    "patient_id": patient_id,
                    "text": (
                        f"Was there any follow-up imaging within 180 days after the imaging study on {img['date']} "
                        f"for patient {patient_id}?"
                    ),
                    "ground_truth": "YES" if followup_found else "NO",
                    "answer_options": ["YES", "NO"],
                    "supporting_node_ids": support,
                }
            )
    return questions


def multi_hop_candidates(graphs: dict[int, nx.DiGraph]) -> list[dict]:
    questions = []
    for patient_id, graph in graphs.items():
        diagnoses = [
            (node_id, data)
            for node_id, data in nodes_by_type(graph, "Diagnosis")
            if "BREAST CANCER" in data["description"] or data["description"].startswith("STAGE ")
        ]
        procedures = nodes_by_type(graph, "Procedure")
        if not diagnoses or not procedures:
            continue
        first_dx_id, first_dx = diagnoses[0]
        later_procedures = [
            (node_id, data)
            for node_id, data in procedures
            if date_distance_days(first_dx["date"], data["date"]) is not None
            and date_distance_days(first_dx["date"], data["date"]) >= 0
        ]
        if not later_procedures:
            continue
        latest_proc_id, latest_proc = later_procedures[-1]
        options = _patient_labels(graph, "Procedure")
        support = _support_bundle(graph, first_dx_id, latest_proc_id)
        if len(support) < 3:
            continue
        questions.append(
            {
                "category": "multi_hop_reasoning",
                "patient_id": patient_id,
                "text": (
                    f"After the initial documented breast cancer assessment on {first_dx['date']}, "
                    f"what later therapy or procedure is documented most recently for patient {patient_id}?"
                ),
                "ground_truth": latest_proc["description"],
                "answer_options": options,
                "supporting_node_ids": support,
            }
        )
    return questions


def _expand(candidates: list[dict], target: int, category: str) -> list[dict]:
    if not candidates:
        return []
    expanded = []
    for index in range(target):
        item = dict(candidates[index % len(candidates)])
        item["id"] = f"{category}-{index:03d}"
        expanded.append(item)
    return expanded


def generate_question_set(graphs: dict[int, nx.DiGraph], per_category: int = 20) -> list[dict]:
    builders = {
        "temporal_proximity": temporal_candidates,
        "causal_chain": causal_candidates,
        "longitudinal_trend": trend_candidates,
        "absence_detection": absence_candidates,
        "multi_hop_reasoning": multi_hop_candidates,
    }
    questions = []
    for category, builder in builders.items():
        questions.extend(_expand(builder(graphs), per_category, category))
    return questions
