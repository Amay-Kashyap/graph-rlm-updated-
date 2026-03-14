from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from graph.graph_utils import approx_token_count

try:
    from langchain_core.tools import tool
except Exception:  # pragma: no cover
    def tool(func):  # type: ignore[misc]
        func.is_tool = True
        return func


@dataclass
class ToolCallLog:
    tool_name: str
    tokens: int
    cumulative_tokens: int


@dataclass
class ToolTracker:
    call_logs: list[ToolCallLog] = field(default_factory=list)

    @property
    def cumulative_tokens(self) -> int:
        if not self.call_logs:
            return 0
        return self.call_logs[-1].cumulative_tokens

    def record(self, tool_name: str, payload: str) -> None:
        tokens = approx_token_count(payload)
        cumulative = self.cumulative_tokens + tokens
        self.call_logs.append(ToolCallLog(tool_name=tool_name, tokens=tokens, cumulative_tokens=cumulative))


@dataclass
class ToolSession:
    patient_id: int
    graph: nx.DiGraph
    tracker: ToolTracker = field(default_factory=ToolTracker)
    submitted_answer: dict[str, str] | None = None


ACTIVE_SESSION: ToolSession | None = None


def set_active_session(session: ToolSession) -> None:
    global ACTIVE_SESSION
    ACTIVE_SESSION = session


def clear_active_session() -> None:
    global ACTIVE_SESSION
    ACTIVE_SESSION = None


def _session() -> ToolSession:
    if ACTIVE_SESSION is None:
        raise RuntimeError("No active tool session.")
    return ACTIVE_SESSION


def _tracked(tool_name: str, content: str) -> str:
    session = _session()
    session.tracker.record(tool_name, content)
    return content


def _graph_nodes_for_patient(patient_id: str) -> list[tuple[str, dict[str, Any]]]:
    session = _session()
    if int(patient_id) != session.patient_id:
        raise ValueError(f"Active session patient is {session.patient_id}, not {patient_id}.")
    return list(session.graph.nodes(data=True))


@tool
def get_timeline(patient_id: str) -> str:
    encounters = [
        (node_id, data)
        for node_id, data in _graph_nodes_for_patient(patient_id)
        if data.get("node_type") == "Encounter"
    ]
    encounters.sort(key=lambda item: item[1].get("date", ""))
    lines = []
    for node_id, data in encounters:
        lines.append(f"{data['date']} | {node_id} | {data.get('type', '')} | {data.get('summary', '')}")
    return _tracked("get_timeline", "\n".join(lines))


@tool
def expand_neighbors(node_id: str) -> str:
    session = _session()
    graph = session.graph
    if node_id not in graph:
        return _tracked("expand_neighbors", f"Node {node_id} was not found.")
    lines = [f"Center node: {node_id}"]
    for neighbor in graph.successors(node_id):
        edge = graph.edges[node_id, neighbor]
        data = graph.nodes[neighbor]
        lines.append(
            f"OUT | {edge['edge_type']} | {neighbor} | {data.get('node_type')} | "
            f"{data.get('date')} | {data.get('summary', data.get('description', ''))}"
        )
    for neighbor in graph.predecessors(node_id):
        edge = graph.edges[neighbor, node_id]
        data = graph.nodes[neighbor]
        lines.append(
            f"IN | {edge['edge_type']} | {neighbor} | {data.get('node_type')} | "
            f"{data.get('date')} | {data.get('summary', data.get('description', ''))}"
        )
    return _tracked("expand_neighbors", "\n".join(lines))


@tool
def read_full_note(node_id: str) -> str:
    session = _session()
    if node_id not in session.graph:
        return _tracked("read_full_note", f"Node {node_id} was not found.")
    data = session.graph.nodes[node_id]
    if data.get("node_type") not in {"Clinical_Note", "Imaging_Report"}:
        return _tracked("read_full_note", f"Node {node_id} does not expose full text.")
    return _tracked("read_full_note", str(data.get("full_text", "")))


@tool
def get_node_details(node_id: str) -> str:
    session = _session()
    if node_id not in session.graph:
        return _tracked("get_node_details", f"Node {node_id} was not found.")
    details = []
    for key, value in sorted(session.graph.nodes[node_id].items()):
        if key == "full_text":
            continue
        details.append(f"{key}: {value}")
    return _tracked("get_node_details", "\n".join(details))


@tool
def search_by_type_and_date(patient_id: str, node_type: str, start_date: str, end_date: str) -> str:
    matched = []
    for node_id, data in _graph_nodes_for_patient(patient_id):
        if data.get("node_type") != node_type:
            continue
        date = str(data.get("date", ""))
        if start_date <= date <= end_date:
            summary = data.get("summary", data.get("description", ""))
            matched.append(f"{date} | {node_id} | {summary}")
    matched.sort()
    return _tracked("search_by_type_and_date", "\n".join(matched) if matched else "No matches.")


@tool
def submit_answer(answer: str, confidence: str, reasoning: str) -> str:
    session = _session()
    session.submitted_answer = {
        "answer": answer,
        "confidence": confidence,
        "reasoning": reasoning,
    }
    payload = f"ANSWER: {answer}\nCONFIDENCE: {confidence}\nREASONING: {reasoning}"
    return _tracked("submit_answer", payload)
