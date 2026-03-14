from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import networkx as nx

from graph.graph_utils import canonicalize_text, date_distance_days, trend_label


def nodes_by_type(graph: nx.DiGraph, node_type: str) -> list[tuple[str, dict]]:
    nodes = [(node_id, data) for node_id, data in graph.nodes(data=True) if data.get("node_type") == node_type]
    nodes.sort(key=lambda item: (item[1].get("date", ""), item[0]))
    return nodes


def node_text(graph: nx.DiGraph, node_id: str, include_full_text: bool = False) -> str:
    data = graph.nodes[node_id]
    fields = [data.get("node_type", ""), data.get("date", "")]
    for key in (
        "summary",
        "description",
        "diagnosis_text",
        "test_name",
        "value",
        "unit",
        "flag",
        "modality",
        "body_part",
        "note_type",
        "outcome_summary",
    ):
        value = data.get(key)
        if value not in (None, ""):
            fields.append(str(value))
    if include_full_text and data.get("full_text"):
        fields.append(str(data["full_text"]))
    return " ".join(fields)


def flatten_graph_record(graph: nx.DiGraph) -> str:
    pieces = []
    for node_id, data in sorted(graph.nodes(data=True), key=lambda item: (item[1].get("date", ""), item[0])):
        pieces.append(f"{node_id}: {node_text(graph, node_id, include_full_text=True)}")
    return "\n".join(pieces)


def _answer_temporal_proximity(question: dict, graph: nx.DiGraph, accessible: set[str] | None) -> tuple[str, list[str]]:
    meta = question["metadata"]
    required = [meta["anchor_node_id"], meta["procedure_node_id"]]
    if accessible is not None and not set(required).issubset(accessible):
        return "Insufficient evidence in retrieved context.", required
    anchor_date = graph.nodes[meta["anchor_node_id"]]["date"]
    proc_date = graph.nodes[meta["procedure_node_id"]]["date"]
    distance = date_distance_days(anchor_date, proc_date)
    answer = "Yes" if distance is not None and distance <= meta["window_days"] else "No"
    return answer, required


def _answer_causal_chain(question: dict, graph: nx.DiGraph, accessible: set[str] | None) -> tuple[str, list[str]]:
    meta = question["metadata"]
    required = [meta["anchor_node_id"], *meta["followup_node_ids"]]
    if accessible is not None and meta["anchor_node_id"] not in accessible:
        return "Insufficient evidence in retrieved context.", required
    descriptions = []
    for node_id in meta["followup_node_ids"]:
        if accessible is not None and node_id not in accessible:
            continue
        node = graph.nodes[node_id]
        descriptions.append(node.get("description", node.get("summary", "")))
    if not descriptions:
        return "No additional workup was documented.", required
    return "; ".join(descriptions), required


def _answer_longitudinal_trend(question: dict, graph: nx.DiGraph, accessible: set[str] | None) -> tuple[str, list[str]]:
    meta = question["metadata"]
    required = list(meta["lab_node_ids"])
    available_ids = required if accessible is None else [node_id for node_id in required if node_id in accessible]
    if len(available_ids) < 2:
        return "Insufficient evidence in retrieved context.", required
    values = [float(graph.nodes[node_id]["value"]) for node_id in available_ids]
    start_id = available_ids[0]
    end_id = available_ids[-1]
    label = trend_label(values)
    node_start = graph.nodes[start_id]
    node_end = graph.nodes[end_id]
    answer = (
        f"{meta['test_name']} {label} from {node_start['value']}{node_start['unit']} "
        f"on {node_start['date']} to {node_end['value']}{node_end['unit']} on {node_end['date']}."
    )
    return answer, required


def _answer_absence_detection(question: dict, graph: nx.DiGraph, accessible: set[str] | None) -> tuple[str, list[str]]:
    meta = question["metadata"]
    required = [meta["anchor_node_id"], *meta["followup_imaging_ids"]]
    if accessible is not None and meta["anchor_node_id"] not in accessible:
        return "Insufficient evidence in retrieved context.", required
    visible_followups = meta["followup_imaging_ids"]
    if accessible is not None:
        visible_followups = [node_id for node_id in visible_followups if node_id in accessible]
    if visible_followups:
        next_node = graph.nodes[visible_followups[0]]
        return f"Yes, follow-up imaging was documented on {next_node['date']}.", required
    return "No follow-up imaging was documented in the requested window.", required


def _answer_multi_hop(question: dict, graph: nx.DiGraph, accessible: set[str] | None) -> tuple[str, list[str]]:
    meta = question["metadata"]
    required = [meta["initial_encounter_id"], meta["final_encounter_id"], meta["final_diagnosis_node_id"]]
    if accessible is not None and not {meta["final_encounter_id"], meta["final_diagnosis_node_id"]}.issubset(accessible):
        return "Insufficient evidence in retrieved context.", required
    final_dx = graph.nodes[meta["final_diagnosis_node_id"]]["description"]
    return final_dx, required


ANSWERERS = {
    "temporal_proximity": _answer_temporal_proximity,
    "causal_chain": _answer_causal_chain,
    "longitudinal_trend": _answer_longitudinal_trend,
    "absence_detection": _answer_absence_detection,
    "multi_hop_reasoning": _answer_multi_hop,
}


def answer_question(question: dict, graph: nx.DiGraph, accessible_node_ids: Iterable[str] | None = None) -> dict:
    accessible = None if accessible_node_ids is None else set(accessible_node_ids)
    answer, required = ANSWERERS[question["category"]](question, graph, accessible)
    return {
        "answer": answer,
        "required_node_ids": required,
        "correct": normalize_answer(answer) == normalize_answer(question["ground_truth"]),
    }


def normalize_answer(text: str) -> str:
    return canonicalize_text(text)


def chunk_graph_documents(graph: nx.DiGraph) -> list[dict]:
    chunks = []
    for node_id, _ in graph.nodes(data=True):
        text = node_text(graph, node_id, include_full_text=True)
        chunks.append(
            {
                "chunk_id": f"chunk:{node_id}",
                "node_ids": [node_id],
                "text": text,
                "normalized_text": canonicalize_text(text),
            }
        )
    return chunks


def retrieval_scores(question_text: str, chunks: list[dict]) -> list[tuple[float, dict]]:
    query_terms = set(canonicalize_text(question_text).split())
    scored = []
    for chunk in chunks:
        chunk_terms = set(chunk["normalized_text"].split())
        overlap = len(query_terms & chunk_terms)
        scored.append((float(overlap), chunk))
    scored.sort(key=lambda item: (item[0], item[1]["chunk_id"]), reverse=True)
    return scored


def questions_by_category(questions: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for question in questions:
        grouped[question["category"]].append(question)
    return dict(grouped)
