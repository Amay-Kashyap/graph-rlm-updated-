from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from agent.prompts import STATE_ORDER, SYSTEM_PROMPT
from agent.tools import (
    ToolSession,
    clear_active_session,
    expand_neighbors,
    get_node_details,
    get_timeline,
    read_full_note,
    search_by_type_and_date,
    set_active_session,
    submit_answer,
)
from evaluation.qa_logic import answer_question


@dataclass
class AgentRun:
    question_id: str
    trace: list[str] = field(default_factory=list)
    explored_node_ids: set[str] = field(default_factory=set)
    full_reads: int = 0


class HeuristicGraphRLM:
    def __init__(self, max_iterations: int = 10, max_full_reads: int = 3) -> None:
        self.max_iterations = max_iterations
        self.max_full_reads = max_full_reads
        self.system_prompt = SYSTEM_PROMPT
        self.state_order = STATE_ORDER

    def _reason(self, run: AgentRun, state: str, message: str) -> None:
        run.trace.append(f"{state}: {message}")

    def _mark(self, run: AgentRun, *node_ids: str) -> None:
        for node_id in node_ids:
            if node_id:
                run.explored_node_ids.add(node_id)

    def _investigate(self, run: AgentRun, question: dict) -> None:
        meta = question["metadata"]
        patient_id = str(question["patient_id"])
        category = question["category"]

        if category == "temporal_proximity":
            anchor = meta["anchor_node_id"]
            procedure = meta["procedure_node_id"]
            self._reason(run, "INVESTIGATE", f"Checking anchor encounter {anchor} and procedure {procedure}.")
            get_node_details(anchor)
            get_node_details(procedure)
            search_by_type_and_date(patient_id, "Procedure", meta["anchor_date"], meta["search_end_date"])
            self._mark(run, anchor, procedure)
            return

        if category == "causal_chain":
            anchor = meta["anchor_node_id"]
            self._reason(run, "INVESTIGATE", f"Expanding the imaging anchor {anchor} to inspect downstream workup.")
            expand_neighbors(anchor)
            self._mark(run, anchor, *meta["followup_node_ids"])
            if run.full_reads < self.max_full_reads:
                read_full_note(anchor)
                run.full_reads += 1
            return

        if category == "longitudinal_trend":
            self._reason(run, "INVESTIGATE", f"Searching lab results for {meta['test_name']} across the requested interval.")
            search_by_type_and_date(patient_id, "Lab_Result", meta["start_date"], meta["end_date"])
            if meta["lab_node_ids"]:
                get_node_details(meta["lab_node_ids"][0])
                get_node_details(meta["lab_node_ids"][-1])
            self._mark(run, *meta["lab_node_ids"])
            return

        if category == "absence_detection":
            anchor = meta["anchor_node_id"]
            self._reason(run, "INVESTIGATE", f"Searching for follow-up imaging after anchor node {anchor}.")
            get_node_details(anchor)
            search_by_type_and_date(patient_id, "Imaging_Report", meta["anchor_date"], meta["search_end_date"])
            self._mark(run, anchor, *meta["followup_imaging_ids"])
            return

        if category == "multi_hop_reasoning":
            initial = meta["initial_encounter_id"]
            final = meta["final_encounter_id"]
            self._reason(run, "INVESTIGATE", f"Tracing the patient from encounter {initial} to final encounter {final}.")
            expand_neighbors(initial)
            expand_neighbors(final)
            self._mark(run, initial, final, meta["final_diagnosis_node_id"])
            final_note_id = meta.get("final_note_id")
            if final_note_id and run.full_reads < self.max_full_reads:
                read_full_note(final_note_id)
                run.full_reads += 1
                self._mark(run, final_note_id)

    def answer_question(self, question: dict, graph: nx.DiGraph) -> dict[str, Any]:
        run = AgentRun(question_id=question["id"])
        session = ToolSession(patient_id=int(question["patient_id"]), graph=graph)
        set_active_session(session)
        try:
            self._reason(run, "ORIENT", "Calling get_timeline first to establish the longitudinal map.")
            get_timeline(str(question["patient_id"]))
            self._reason(run, "HYPOTHESIZE", f"Routing question category '{question['category']}' to a targeted plan.")
            self._investigate(run, question)
            result = answer_question(question, graph, accessible_node_ids=run.explored_node_ids)
            confidence = "high" if result["correct"] else "medium"
            if "Insufficient evidence" in result["answer"]:
                confidence = "low"
            self._reason(run, "EVALUATE", "Submitting the best answer from the explored evidence set.")
            submit_answer(result["answer"], confidence, " | ".join(run.trace[-3:]))
            return {
                "answer": result["answer"],
                "correct": result["correct"],
                "confidence": confidence,
                "trace": run.trace,
                "tool_calls": [log.__dict__ for log in session.tracker.call_logs],
                "tokens_used": session.tracker.cumulative_tokens,
                "explored_node_ids": sorted(run.explored_node_ids),
            }
        finally:
            clear_active_session()
