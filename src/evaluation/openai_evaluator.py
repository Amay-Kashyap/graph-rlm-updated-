from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass

import networkx as nx
import pandas as pd
from openai import OpenAI

from agent.minimal_rlm_graph import MinimalRLMGraphController
from evaluation.real_question_generator import generate_question_set
from graph.breast_notes_graph import PROJECT_ROOT, get_available_patient_ids, get_patient_graph, graph_stats
from graph.graph_utils import canonicalize_text


DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _extract_dates(text: str) -> list[str]:
    return re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)


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
    return " ".join(str(value) for value in fields if value)


def _record_summary(graph: nx.DiGraph) -> str:
    lines = []
    for node_id, data in sorted(graph.nodes(data=True), key=lambda item: (item[1].get("date", ""), item[0])):
        payload = _text_for_node(node_id, data)
        if data.get("node_type") == "Clinical_Note":
            payload = payload[:600]
        lines.append(payload)
    return "\n".join(lines)


def _node_type_hint(question: str) -> list[str]:
    lower = question.lower()
    hints = []
    if any(token in lower for token in ("imaging", "scan", "ct", "mri", "mammogram", "pet")):
        hints.append("Imaging_Report")
    if any(token in lower for token in ("treatment", "therapy", "procedure", "mastectomy", "lumpectomy", "trastuzumab", "taxol", "carboplatin", "gemcitabine")):
        hints.append("Procedure")
    if any(token in lower for token in ("stage", "diagnosis", "cancer")):
        hints.append("Diagnosis")
    if not hints:
        hints.append("Clinical_Note")
    return hints


def _rank_nodes(graph: nx.DiGraph, question_text: str, limit: int = 5) -> list[str]:
    question_terms = set(canonicalize_text(question_text).split())
    dates = set(_extract_dates(question_text))
    ranked = []
    for node_id, data in graph.nodes(data=True):
        haystack = canonicalize_text(_text_for_node(node_id, data))
        node_terms = set(haystack.split())
        score = len(question_terms & node_terms)
        if data.get("date") in dates:
            score += 8
        if data.get("node_type") in _node_type_hint(question_text):
            score += 5
        ranked.append((score, node_id))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [node_id for score, node_id in ranked[:limit] if score > 0]


@dataclass
class LLMResult:
    answer: str
    tokens_used: int


class OpenAITextResponder:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 80) -> LLMResult:
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_completion_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                text = response.choices[0].message.content or ""
                tokens = int(getattr(response.usage, "total_tokens", 0) or 0)
                return LLMResult(answer=text.strip(), tokens_used=tokens)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError("Unreachable")


class OpenAIEmbeddingIndex:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model
        self.index: dict[int, list[dict]] = {}

    def _chunk_note(self, node_id: str, text: str) -> list[dict]:
        if not text:
            return []
        chunks = []
        start = 0
        chunk_size = 900
        overlap = 120
        part = 0
        while start < len(text):
            chunk = text[start : start + chunk_size]
            chunks.append({"chunk_id": f"{node_id}:{part}", "node_id": node_id, "text": chunk})
            if start + chunk_size >= len(text):
                break
            start += chunk_size - overlap
            part += 1
        return chunks

    def build_for_graph(self, patient_id: int, graph: nx.DiGraph) -> None:
        if patient_id in self.index:
            return
        chunks = []
        for node_id, data in graph.nodes(data=True):
            if data.get("node_type") == "Clinical_Note":
                chunks.extend(self._chunk_note(node_id, str(data.get("full_text", ""))))
            else:
                chunks.append({"chunk_id": node_id, "node_id": node_id, "text": _text_for_node(node_id, data)})
        embeddings = []
        for start in range(0, len(chunks), 64):
            batch = chunks[start : start + 64]
            response = self.client.embeddings.create(
                model=self.model,
                input=[item["text"] for item in batch],
            )
            embeddings.extend(item.embedding for item in response.data)
        for chunk, vector in zip(chunks, embeddings):
            chunk["embedding"] = vector
        self.index[patient_id] = chunks

    def retrieve(self, patient_id: int, question_text: str, top_k: int = 6) -> tuple[list[dict], int]:
        chunks = self.index[patient_id]
        response = self.client.embeddings.create(model=self.model, input=question_text)
        query_vector = response.data[0].embedding
        query_tokens = int(getattr(response.usage, "total_tokens", 0) or 0)
        scored = [(_cosine_similarity(query_vector, chunk["embedding"]), chunk) for chunk in chunks]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]], query_tokens


def _normalize_live_answer(answer: str) -> str:
    return canonicalize_text(answer).upper()


def _exact_match(predicted: str, ground_truth: str) -> bool:
    return _normalize_live_answer(predicted) == _normalize_live_answer(ground_truth)


def _answer_prompt(question: dict, context: str) -> tuple[str, str]:
    options = question.get("answer_options") or []
    options_text = "\n".join(f"- {option}" for option in options) if options else "- Use the exact best answer from context."
    system_prompt = (
        "You answer breast oncology record questions from provided EHR context. "
        "Return only the final answer. No explanation. "
        "If answer options are provided, output exactly one of them."
    )
    user_prompt = (
        f"Question:\n{question['text']}\n\n"
        f"Allowed answer options:\n{options_text}\n\n"
        f"Context:\n{context}\n"
    )
    return system_prompt, user_prompt


class LiveFullContextBaseline:
    def __init__(self, responder: OpenAITextResponder) -> None:
        self.responder = responder

    def answer_question(self, question: dict, graph: nx.DiGraph) -> dict:
        context = _record_summary(graph)
        system_prompt, user_prompt = _answer_prompt(question, context)
        result = self.responder.complete(system_prompt, user_prompt)
        return {
            "answer": result.answer,
            "correct": _exact_match(result.answer, question["ground_truth"]),
            "tokens_used": result.tokens_used,
            "tool_calls": [],
        }


class LiveNaiveRAGBaseline:
    def __init__(self, responder: OpenAITextResponder, embed_index: OpenAIEmbeddingIndex) -> None:
        self.responder = responder
        self.embed_index = embed_index

    def answer_question(self, question: dict, graph: nx.DiGraph) -> dict:
        patient_id = int(question["patient_id"])
        self.embed_index.build_for_graph(patient_id, graph)
        chunks, embed_tokens = self.embed_index.retrieve(patient_id, question["text"], top_k=6)
        context = "\n\n".join(chunk["text"] for chunk in chunks)
        system_prompt, user_prompt = _answer_prompt(question, context)
        result = self.responder.complete(system_prompt, user_prompt)
        return {
            "answer": result.answer,
            "correct": _exact_match(result.answer, question["ground_truth"]),
            "tokens_used": result.tokens_used + embed_tokens,
            "tool_calls": [{"tool_name": "embed_retrieve", "tokens": embed_tokens, "cumulative_tokens": embed_tokens}],
        }


class LiveGraphRLM:
    def __init__(self, model: str, recursive_model: str) -> None:
        self.controller = MinimalRLMGraphController(
            model=model,
            recursive_model=recursive_model,
            max_iterations=6,
        )

    def answer_question(self, question: dict, graph: nx.DiGraph) -> dict:
        result = self.controller.answer_question(question, graph)
        return {
            "answer": result.answer,
            "correct": _exact_match(result.answer, question["ground_truth"]),
            "tokens_used": result.tokens_used,
            "tool_calls": result.trace,
        }


def run_live_evaluation(
    sample_size: int = 100,
    chat_model: str = DEFAULT_CHAT_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = OpenAI()
    responder = OpenAITextResponder(client, chat_model)
    embed_index = OpenAIEmbeddingIndex(client, embed_model)

    patient_ids = get_available_patient_ids()
    graphs = {patient_id: get_patient_graph(patient_id) for patient_id in patient_ids}
    per_category = max(1, sample_size // 5)
    questions = generate_question_set(graphs, per_category=per_category)

    systems = [
        ("Naive RAG", LiveNaiveRAGBaseline(responder, embed_index)),
        ("Graph-RLM", LiveGraphRLM(chat_model, chat_model)),
        ("Full-Context", LiveFullContextBaseline(responder)),
    ]

    rows = []
    for system_name, system in systems:
        for question in questions:
            graph = graphs[int(question["patient_id"])]
            result = system.answer_question(question, graph)
            rows.append(
                {
                    "system": system_name,
                    "question_id": question["id"],
                    "patient_id": question["patient_id"],
                    "category": question["category"],
                    "question": question["text"],
                    "ground_truth": question["ground_truth"],
                    "supporting_node_ids": json.dumps(question.get("supporting_node_ids", [])),
                    "answer": result["answer"],
                    "correct": int(bool(result["correct"])),
                    "tokens_used": int(result["tokens_used"]),
                    "tool_calls": len(result.get("tool_calls", [])),
                }
            )

    detailed = pd.DataFrame(rows)
    summary_rows = []
    for system_name, group in detailed.groupby("system"):
        summary_rows.append(
            {
                "system": system_name,
                "category": "ALL",
                "questions": int(group.shape[0]),
                "accuracy_pct": round(group["correct"].mean() * 100.0, 2),
                "avg_tokens_per_query": round(group["tokens_used"].mean(), 2),
            }
        )
    for (system_name, category), group in detailed.groupby(["system", "category"]):
        summary_rows.append(
            {
                "system": system_name,
                "category": category,
                "questions": int(group.shape[0]),
                "accuracy_pct": round(group["correct"].mean() * 100.0, 2),
                "avg_tokens_per_query": round(group["tokens_used"].mean(), 2),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["category", "system"]).reset_index(drop=True)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(results_dir / "openai_detailed_results.csv", index=False)
    summary.to_csv(results_dir / "openai_metrics.csv", index=False)
    graph_stats().to_csv(results_dir / "breast_note_graph_stats.csv", index=False)
    detailed[detailed["correct"] == 0].to_csv(results_dir / "openai_failure_audit.csv", index=False)
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    with (processed_dir / "openai_questions.json").open("w", encoding="utf-8") as handle:
        json.dump(questions, handle, indent=2)
    return detailed, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--chat-model", type=str, default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    args = parser.parse_args()
    _, summary = run_live_evaluation(sample_size=args.sample_size, chat_model=args.chat_model, embed_model=args.embed_model)
    overall = summary[summary["category"] == "ALL"].copy()
    print("OpenAI evaluation complete.")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
