"""Evaluation runner using the LangGraph RLM architecture with synthetic
ground-truth injection.

This module:
1. Loads patient graphs from breast-notes CSVs.
2. Injects synthetic facts (procedures, imaging, diagnoses) into the graphs
   and note text so we have deterministic ground truth.
3. Generates two question pools:
   a) *Organic* questions from the existing ``real_question_generator``
      (temporal, causal, trend, absence, multi-hop).
   b) *Synthetic* questions derived from the injected facts.
4. Runs three systems against the combined question set:
   - Full-Context baseline
   - Naive RAG baseline
   - **LangGraph RLM** (new â€” replaces the old manual-loop controller)
5. Reports per-category and overall accuracy.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd
from openai import OpenAI

from agent.langgraph_rlm import LangGraphRLMController
from evaluation.real_question_generator import generate_question_set
from graph.breast_notes_graph import (
    PROJECT_ROOT,
    get_available_patient_ids,
    get_patient_graph,
    graph_stats,
)
from graph.graph_utils import canonicalize_text
from graph.synthetic_injection import SyntheticFact, inject_synthetic_facts

DEFAULT_CHAT_MODEL = "gpt-5.4"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


# â”€â”€ helpers (reused from openai_evaluator) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _cosine_similarity(left: list[float], right: list[float]) -> float:
    num = sum(a * b for a, b in zip(left, right))
    ln = math.sqrt(sum(a * a for a in left))
    rn = math.sqrt(sum(b * b for b in right))
    if ln == 0 or rn == 0:
        return 0.0
    return num / (ln * rn)


def _text_for_node(node_id: str, data: dict) -> str:
    fields = [
        node_id,
        data.get("node_type", ""),
        data.get("date", ""),
        data.get("summary", ""),
        data.get("description", ""),
        data.get("note_type", ""),
        data.get("modality", ""),
        data.get("body_part", ""),
        data.get("department", ""),
    ]
    if data.get("full_text"):
        fields.append(str(data["full_text"])[:1200])
    return " ".join(str(v) for v in fields if v)


def _record_summary(graph: nx.DiGraph) -> str:
    lines = []
    for nid, d in sorted(graph.nodes(data=True), key=lambda x: (x[1].get("date", ""), x[0])):
        payload = _text_for_node(nid, d)
        if d.get("node_type") == "Clinical_Note":
            payload = payload[:600]
        lines.append(payload)
    return "\n".join(lines)


def _normalize(text: str) -> str:
    return canonicalize_text(text).upper()


def _exact_match(predicted: str, ground_truth: str) -> bool:
    return _normalize(predicted) == _normalize(ground_truth)


# â”€â”€ answer prompt â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _answer_prompt(question: dict, context: str) -> tuple[str, str]:
    options = question.get("answer_options") or []
    opts_text = "\n".join(f"- {o}" for o in options) if options else "- Use the exact best answer from context."
    sys = (
        "You answer breast oncology record questions from provided EHR context. "
        "Return only the final answer. No explanation. "
        "If answer options are provided, output exactly one of them."
    )
    usr = (
        f"Question:\n{question['text']}\n\n"
        f"Allowed answer options:\n{opts_text}\n\n"
        f"Context:\n{context}\n"
    )
    return sys, usr


# â”€â”€ responder / embedder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class LLMResult:
    answer: str
    tokens_used: int


class _Responder:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client, self.model = client, model

    def complete(self, sys_prompt: str, usr_prompt: str, max_tokens: int = 80) -> LLMResult:
        for attempt in range(3):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0, max_completion_tokens=max_tokens,
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
                )
                return LLMResult(answer=(r.choices[0].message.content or "").strip(),
                                 tokens_used=int(getattr(r.usage, "total_tokens", 0) or 0))
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError("Unreachable")


class _EmbedIndex:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client, self.model = client, model
        self.index: dict[int, list[dict]] = {}

    def build(self, pid: int, graph: nx.DiGraph) -> None:
        if pid in self.index:
            return
        chunks = []
        for nid, d in graph.nodes(data=True):
            text = _text_for_node(nid, d)
            chunks.append({"chunk_id": nid, "node_id": nid, "text": text})
        embeddings = []
        for start in range(0, len(chunks), 64):
            batch = chunks[start:start + 64]
            r = self.client.embeddings.create(model=self.model, input=[c["text"] for c in batch])
            embeddings.extend(item.embedding for item in r.data)
        for c, v in zip(chunks, embeddings):
            c["embedding"] = v
        self.index[pid] = chunks

    def retrieve(self, pid: int, q_text: str, top_k: int = 6) -> tuple[list[dict], int]:
        chunks = self.index[pid]
        r = self.client.embeddings.create(model=self.model, input=q_text)
        qv = r.data[0].embedding
        qt = int(getattr(r.usage, "total_tokens", 0) or 0)
        scored = [(_cosine_similarity(qv, c["embedding"]), c) for c in chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]], qt


# â”€â”€ baselines â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class FullContextBaseline:
    def __init__(self, resp: _Responder) -> None:
        self.resp = resp

    def answer_question(self, q: dict, g: nx.DiGraph) -> dict:
        ctx = _record_summary(g)
        sp, up = _answer_prompt(q, ctx)
        r = self.resp.complete(sp, up)
        return {"answer": r.answer, "correct": _exact_match(r.answer, q["ground_truth"]),
                "tokens_used": r.tokens_used, "tool_calls": []}


class NaiveRAGBaseline:
    def __init__(self, resp: _Responder, idx: _EmbedIndex) -> None:
        self.resp, self.idx = resp, idx

    def answer_question(self, q: dict, g: nx.DiGraph) -> dict:
        pid = int(q["patient_id"])
        self.idx.build(pid, g)
        chunks, et = self.idx.retrieve(pid, q["text"], top_k=6)
        ctx = "\n\n".join(c["text"] for c in chunks)
        sp, up = _answer_prompt(q, ctx)
        r = self.resp.complete(sp, up)
        return {"answer": r.answer, "correct": _exact_match(r.answer, q["ground_truth"]),
                "tokens_used": r.tokens_used + et,
                "tool_calls": [{"tool_name": "embed_retrieve", "tokens": et}]}


class LangGraphRLMSystem:
    def __init__(self, model: str, recursive_model: str) -> None:
        self.ctrl = LangGraphRLMController(model=model, recursive_model=recursive_model, max_iterations=4)

    def answer_question(self, q: dict, g: nx.DiGraph) -> dict:
        r = self.ctrl.answer_question(q, g)
        return {"answer": r.answer, "correct": _exact_match(r.answer, q["ground_truth"]),
                "tokens_used": r.tokens_used, "tool_calls": r.trace}


# â”€â”€ synthetic question conversion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def synthetic_facts_to_questions(facts: list[SyntheticFact]) -> list[dict]:
    questions = []
    for i, f in enumerate(facts):
        questions.append({
            "id": f"synth-{i:03d}",
            "patient_id": f.patient_id,
            "category": f"synthetic_{f.category}",
            "text": f.ground_truth_question,
            "ground_truth": f.ground_truth_answer,
            "answer_options": f.answer_options,
            "supporting_node_ids": f.supporting_node_ids,
        })
    return questions


# â”€â”€ main evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_langgraph_evaluation(
    sample_size: int = 100,
    chat_model: str = DEFAULT_CHAT_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    synthetic_per_patient: int = 6,
    seed: int = 42,
    synthetic_only: bool = False,
    organic_only: bool = False,
    graph_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = None if graph_only else OpenAI()
    resp = None if graph_only else _Responder(client, chat_model)
    idx = None if graph_only else _EmbedIndex(client, embed_model)

    # 1. Load patient graphs (deep copy so injection is isolated)
    patient_ids = get_available_patient_ids()
    raw_graphs = {pid: get_patient_graph(pid) for pid in patient_ids}
    graphs = {pid: copy.deepcopy(g) for pid, g in raw_graphs.items()}

    # 2. Inject synthetic facts
    facts = inject_synthetic_facts(graphs, facts_per_patient=synthetic_per_patient, seed=seed)
    synth_questions = [] if organic_only else synthetic_facts_to_questions(facts)
    print(f"  Injected {len(facts)} synthetic facts across {len(graphs)} patients.", flush=True)

    # 3. Generate organic questions (on enriched graphs) â€” skip if synthetic_only
    if synthetic_only:
        organic_questions: list[dict] = []
        print("  Skipping organic questions (--synthetic-only).", flush=True)
    else:
        per_cat = max(1, sample_size // 5)
        organic_questions = generate_question_set(graphs, per_category=per_cat)
        print(f"  Generated {len(organic_questions)} organic questions.", flush=True)

    all_questions = organic_questions + synth_questions
    print(f"  Total questions: {len(all_questions)}", flush=True)

    # 4. Build systems
    if graph_only:
        systems = [("LangGraph-RLM", LangGraphRLMSystem(chat_model, chat_model))]
    else:
        systems = [
            ("Naive RAG", NaiveRAGBaseline(resp, idx)),
            ("LangGraph-RLM", LangGraphRLMSystem(chat_model, chat_model)),
            ("Full-Context", FullContextBaseline(resp)),
        ]

    # 5. Evaluate
    rows = []
    total = len(all_questions) * len(systems)
    done = 0
    for sys_name, system in systems:
        for q in all_questions:
            g = graphs[int(q["patient_id"])]
            try:
                result = system.answer_question(q, g)
            except Exception as exc:
                result = {"answer": f"ERROR: {exc}", "correct": False, "tokens_used": 0, "tool_calls": []}
            done += 1
            status = "CORRECT" if result["correct"] else "WRONG"
            print(f"  [{done}/{total}] {sys_name} | {q['category']} | {status}", flush=True)
            rows.append({
                "system": sys_name,
                "question_id": q["id"],
                "patient_id": q["patient_id"],
                "category": q["category"],
                "question": q["text"],
                "ground_truth": q["ground_truth"],
                "supporting_node_ids": json.dumps(q.get("supporting_node_ids", [])),
                "answer": result["answer"],
                "correct": int(bool(result["correct"])),
                "tokens_used": int(result["tokens_used"]),
                "tool_calls": len(result.get("tool_calls", [])),
            })

    # 6. Summarize
    detailed = pd.DataFrame(rows)
    summary_rows = []
    for sys_name, grp in detailed.groupby("system"):
        summary_rows.append({
            "system": sys_name, "category": "ALL",
            "questions": int(grp.shape[0]),
            "accuracy_pct": round(grp["correct"].mean() * 100, 2),
            "avg_tokens_per_query": round(grp["tokens_used"].mean(), 2),
        })
    for (sys_name, cat), grp in detailed.groupby(["system", "category"]):
        summary_rows.append({
            "system": sys_name, "category": cat,
            "questions": int(grp.shape[0]),
            "accuracy_pct": round(grp["correct"].mean() * 100, 2),
            "avg_tokens_per_query": round(grp["tokens_used"].mean(), 2),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["category", "system"]).reset_index(drop=True)

    # 7. Persist
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(results_dir / "langgraph_detailed_results.csv", index=False)
    summary.to_csv(results_dir / "langgraph_metrics.csv", index=False)
    detailed[detailed["correct"] == 0].to_csv(results_dir / "langgraph_failure_audit.csv", index=False)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with (processed_dir / "langgraph_questions.json").open("w", encoding="utf-8") as fh:
        json.dump(all_questions, fh, indent=2, default=str)
    with (processed_dir / "synthetic_facts.json").open("w", encoding="utf-8") as fh:
        json.dump([f.__dict__ for f in facts], fh, indent=2)

    return detailed, summary


# â”€â”€ CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph RLM evaluation with synthetic ground truth")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--chat-model", type=str, default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--synthetic-per-patient", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Only evaluate synthetic ground-truth questions (skip organic)")
    parser.add_argument("--organic-only", action="store_true",
                        help="Only evaluate organic questions (skip synthetic)")
    parser.add_argument("--graph-only", action="store_true",
                        help="Run only the LangGraph-RLM system and skip the baselines")
    args = parser.parse_args()

    detailed, summary = run_langgraph_evaluation(
        sample_size=args.sample_size,
        chat_model=args.chat_model,
        embed_model=args.embed_model,
        synthetic_per_patient=args.synthetic_per_patient,
        seed=args.seed,
        synthetic_only=args.synthetic_only,
        organic_only=args.organic_only,
        graph_only=args.graph_only,
    )

    print("\n" + "=" * 70)
    print("LANGGRAPH RLM EVALUATION - RESULTS")
    print("=" * 70)

    overall = summary[summary["category"] == "ALL"].copy()
    print("\n-- Overall --")
    print(overall.to_string(index=False))

    synth_cats = [c for c in summary["category"].unique() if c.startswith("synthetic_")]
    if synth_cats:
        synth_summary = summary[summary["category"].isin(synth_cats)]
        print("\n-- Synthetic Ground-Truth Categories --")
        print(synth_summary.to_string(index=False))

    organic_cats = [c for c in summary["category"].unique()
                    if c != "ALL" and not c.startswith("synthetic_")]
    if organic_cats:
        org_summary = summary[summary["category"].isin(organic_cats)]
        print("\n-- Organic Categories --")
        print(org_summary.to_string(index=False))

    print(f"\nResults saved to: {PROJECT_ROOT / 'results'}")


if __name__ == "__main__":
    main()
