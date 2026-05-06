from __future__ import annotations

import copy
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent.langgraph_rlm import LangGraphRLMController
from evaluation.real_question_generator import (
    absence_candidates,
    causal_candidates,
    temporal_candidates,
    trend_candidates,
)
from graph.breast_notes_graph import get_available_patient_ids, get_patient_graph
from graph.graph_utils import summarize_node
from graph.synthetic_injection import inject_synthetic_facts


OUTPUT_PATH = PROJECT_ROOT / "reports" / "graph_rlm_mixed_5_organic_5_synthetic_trace.txt"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def normalize_answer(text: Any) -> str:
    value = "" if text is None else str(text).strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


def graph_summary(graph, patient_id: int) -> str:
    node_counter = Counter()
    edge_counter = Counter()
    for _, data in graph.nodes(data=True):
        node_counter[str(data.get("node_type", "Unknown"))] += 1
    for _, _, edge in graph.edges(data=True):
        edge_counter[str(edge.get("edge_type", "Unknown"))] += 1
    lines = [
        f"PATIENT {patient_id}",
        f"  nodes: {graph.number_of_nodes()}",
        f"  edges: {graph.number_of_edges()}",
        f"  node types: {dict(sorted(node_counter.items()))}",
        f"  edge types: {dict(sorted(edge_counter.items()))}",
    ]
    return "\n".join(lines)


def node_block(graph, node_ids: list[str]) -> str:
    lines = []
    seen = set()
    for node_id in node_ids:
        if node_id in seen or node_id not in graph:
            continue
        seen.add(node_id)
        data = graph.nodes[node_id]
        payload = {
            "node_id": node_id,
            "node_type": data.get("node_type"),
            "date": data.get("date"),
            "summary": summarize_node(data),
            "assertion": data.get("assertion"),
            "temporality": data.get("temporality"),
            "confidence": data.get("confidence"),
        }
        lines.append("    - " + json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines) if lines else "    - none"


def edge_block(graph, node_ids: list[str]) -> str:
    node_set = set(node_ids)
    lines = []
    for source, target, edge in graph.edges(data=True):
        if source not in node_set and target not in node_set:
            continue
        payload = {
            "from": source,
            "to": target,
            "edge_type": edge.get("edge_type"),
            "confidence": edge.get("confidence"),
            "evidence_note_ids": edge.get("evidence_note_ids"),
        }
        lines.append("    - " + json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines) if lines else "    - none"


def select_organic_questions(graphs: dict[int, Any]) -> list[dict[str, Any]]:
    questions = []
    questions.append(temporal_candidates(graphs)[0])
    questions.append(temporal_candidates(graphs)[1])
    questions.append(causal_candidates(graphs)[0])
    questions.append(trend_candidates(graphs)[0])
    questions.append(absence_candidates(graphs)[0])
    for idx, q in enumerate(questions, start=1):
        q["id"] = f"organic-{idx:02d}"
    return questions


def select_synthetic_questions(graphs: dict[int, Any]) -> list[dict[str, Any]]:
    facts = inject_synthetic_facts(graphs, facts_per_patient=6, seed=42)
    categories = ["procedure", "imaging", "diagnosis", "negative_procedure", "negative_diagnosis"]
    selected = []
    for category in categories:
        fact = next(f for f in facts if f.category == category)
        selected.append(
            {
                "id": f"synthetic-{len(selected)+1:02d}",
                "patient_id": fact.patient_id,
                "category": f"synthetic_{fact.category}",
                "text": fact.ground_truth_question,
                "ground_truth": fact.ground_truth_answer,
                "answer_options": fact.answer_options,
                "supporting_node_ids": fact.supporting_node_ids,
            }
        )
    return selected


def run_questions(controller: LangGraphRLMController, graphs: dict[int, Any], questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for question in questions:
        graph = graphs[int(question["patient_id"])]
        result = controller.answer_question(question, graph)
        rows.append(
            {
                "question": question,
                "answer": normalize_answer(result.answer),
                "correct": normalize_answer(result.answer).upper() == normalize_answer(question["ground_truth"]).upper(),
                "tokens_used": result.tokens_used,
                "trace": result.trace,
            }
        )
    return rows


def build_report(graphs: dict[int, Any], organic_rows: list[dict[str, Any]], synthetic_rows: list[dict[str, Any]]) -> str:
    patient_ids = sorted({int(row["question"]["patient_id"]) for row in organic_rows + synthetic_rows})
    lines = []
    lines.append("GRAPH RLM MIXED TRACE REPORT")
    lines.append("=" * 80)
    lines.append("This file shows the observable Graph-RLM execution trace for 5 organic and 5 synthetic questions.")
    lines.append("It includes graph summaries, supporting nodes, nearby edges, and the controller trace.")
    lines.append("It does not attempt to reveal hidden chain-of-thought.\n")

    lines.append("PATIENT GRAPH SUMMARIES")
    lines.append("-" * 80)
    for patient_id in patient_ids:
        lines.append(graph_summary(graphs[patient_id], patient_id))
        lines.append("")

    def append_section(title: str, rows: list[dict[str, Any]]) -> None:
        lines.append(title)
        lines.append("-" * 80)
        for idx, row in enumerate(rows, start=1):
            q = row["question"]
            graph = graphs[int(q["patient_id"])]
            touched_ids = []
            for trace_item in row["trace"]:
                if "matches" in trace_item:
                    touched_ids.extend(trace_item["matches"])
                if "anchor_nodes" in trace_item:
                    touched_ids.extend(trace_item["anchor_nodes"])
                if "followups" in trace_item:
                    touched_ids.extend(trace_item["followups"])
                if "stage_nodes" in trace_item:
                    touched_ids.extend(trace_item["stage_nodes"])
                if "supporting_edges" in trace_item:
                    for edge in trace_item["supporting_edges"]:
                        touched_ids.extend([edge.get("from", ""), edge.get("to", "")])
            touched_ids = [node_id for node_id in dict.fromkeys(touched_ids) if node_id]
            combined_ids = list(dict.fromkeys(list(q.get("supporting_node_ids", [])) + touched_ids))

            lines.append(f"{idx}. {q['id']} | {q['category']} | patient {q['patient_id']}")
            lines.append(f"   question: {q['text']}")
            lines.append(f"   expected: {q['ground_truth']}")
            lines.append(f"   answer:   {row['answer']}")
            lines.append(f"   correct:  {row['correct']}")
            lines.append(f"   tokens:   {row['tokens_used']}")
            lines.append("   trace:")
            for trace_item in row["trace"]:
                lines.append("    - " + json.dumps(trace_item, ensure_ascii=False))
            lines.append("   trace notes:")
            for trace_item in row["trace"]:
                phase = trace_item.get("phase", "")
                if phase == "structured_temporal":
                    lines.append(
                        f"    - temporal resolver searched for procedure label '{trace_item.get('label')}' "
                        f"around {trace_item.get('anchor_date')} within {trace_item.get('window')} days; "
                        f"considered={len(trace_item.get('considered', []))}, matched={len(trace_item.get('matches', []))}, "
                        f"filtered={len(trace_item.get('filtered_out', []))}"
                    )
                elif phase == "structured_causal":
                    lines.append(
                        f"    - causal resolver anchored on modality '{trace_item.get('modality')}' at {trace_item.get('anchor_date')}; "
                        f"candidate_answers={trace_item.get('candidate_answers', [])}"
                    )
                elif phase == "structured_absence":
                    lines.append(
                        f"    - absence resolver anchored on {trace_item.get('anchor_date')}; "
                        f"anchor_nodes={len(trace_item.get('anchor_nodes', []))}, "
                        f"considered_followups={len(trace_item.get('considered_followups', []))}, "
                        f"followups={len(trace_item.get('followups', []))}"
                    )
                elif phase == "structured_trend":
                    lines.append(
                        f"    - trend resolver compared {len(trace_item.get('considered_stage_nodes', []))} stage nodes and chose "
                        f"{trace_item.get('stage_nodes', [])}"
                    )
                elif phase == "structured_lookup":
                    lines.append(
                        f"    - synthetic lookup used entity_type={trace_item.get('entity_type')} "
                        f"entity_value={trace_item.get('entity_value')} date={trace_item.get('target_date')}"
                    )
            lines.append("   supporting / touched nodes:")
            lines.append(node_block(graph, combined_ids))
            lines.append("   nearby edges:")
            lines.append(edge_block(graph, combined_ids))
            lines.append("")

    append_section("ORGANIC QUESTIONS (5)", organic_rows)
    append_section("SYNTHETIC QUESTIONS (5)", synthetic_rows)

    return "\n".join(lines)


def main() -> None:
    load_env_file(PROJECT_ROOT / ".env")
    os.environ.setdefault("RLM_REPL_TMPDIR", str(PROJECT_ROOT / ".repl_tmp"))
    model = "gpt-5.4"
    controller = LangGraphRLMController(model=model, recursive_model=model, max_iterations=4)

    raw_graphs = {pid: get_patient_graph(pid) for pid in get_available_patient_ids()}
    synth_graphs = {pid: copy.deepcopy(graph) for pid, graph in raw_graphs.items()}

    organic_questions = select_organic_questions(raw_graphs)
    synthetic_questions = select_synthetic_questions(synth_graphs)

    organic_rows = run_questions(controller, raw_graphs, organic_questions)
    synthetic_rows = run_questions(controller, synth_graphs, synthetic_questions)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_report({**raw_graphs, **synth_graphs}, organic_rows, synthetic_rows), encoding="utf-8")

    total = len(organic_rows) + len(synthetic_rows)
    correct = sum(1 for row in organic_rows + synthetic_rows if row["correct"])
    avg_tokens = sum(int(row["tokens_used"]) for row in organic_rows + synthetic_rows) / total
    print(json.dumps({"path": str(OUTPUT_PATH), "questions": total, "correct": correct, "avg_tokens": round(avg_tokens, 2)}, indent=2))


if __name__ == "__main__":
    main()
