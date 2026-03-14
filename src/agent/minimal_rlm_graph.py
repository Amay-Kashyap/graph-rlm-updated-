from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
from openai import OpenAI

from graph.graph_utils import approx_token_count


VENDOR_RLM_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "rlm-minimal"
if str(VENDOR_RLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_RLM_ROOT))

from rlm.repl import REPLEnv  # type: ignore[import-not-found]
from rlm.utils.utils import (  # type: ignore[import-not-found]
    check_for_final_answer,
    find_code_blocks,
    process_code_execution,
)


ROOT_SYSTEM_PROMPT = """You are an oncology graph RLM controller working inside a Python REPL.

Your job is to answer one breast oncology EHR question using as little context as possible.

The REPL contains helper functions:
- timeline() -> compact encounter chronology
- search(term, node_type=None, limit=10) -> lexical search over indexed graph nodes and note snippets
- multi_search(terms, node_type=None, limit=10) -> ranked search using multiple terms
- node(node_id) -> metadata for a node
- neighbors(node_id) -> connected nodes with edge labels
- list_chunks(node_id, limit=10) -> note chunk metadata
- read_chunk(node_id, chunk_index) -> exact chunk text
- ask_chunk(node_id, chunk_index, prompt) -> recursive sub-LLM analysis on one chunk
- answer_options() -> allowed final answers, when provided

Rules:
- Start with timeline() or search()/multi_search() to orient.
- Prefer note summaries and chunk metadata before reading raw chunks.
- Use ask_chunk only for the most relevant chunks.
- Keep buffers in variables and return with FINAL_VAR(variable_name) when done.
- If answer options are provided, final answer must exactly match one option.
- Do not expose reasoning in the final answer.
"""


def _chunk_text(text: str, chunk_chars: int = 1400, overlap: int = 200) -> list[str]:
    text = text or ""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_chars]
        chunks.append(chunk)
        if start + chunk_chars >= len(text):
            break
        start += chunk_chars - overlap
    return chunks


def _node_brief(node_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "node_type": data.get("node_type", ""),
        "date": data.get("date", ""),
        "summary": data.get("summary", data.get("description", "")),
        "description": data.get("description", ""),
        "note_type": data.get("note_type", ""),
        "modality": data.get("modality", ""),
        "department": data.get("department", ""),
    }


def _build_rlm_context(graph: nx.DiGraph, answer_options: list[str] | None = None) -> dict[str, Any]:
    answer_options = answer_options or []
    nodes: dict[str, dict[str, Any]] = {}
    neighbors: dict[str, list[dict[str, Any]]] = {}
    timeline = []
    search_index = []
    note_chunks: dict[str, list[dict[str, Any]]] = {}
    note_chunk_meta: dict[str, list[dict[str, Any]]] = {}

    for node_id, data in sorted(graph.nodes(data=True), key=lambda item: (item[1].get("date", ""), item[0])):
        brief = _node_brief(node_id, data)
        nodes[node_id] = brief
        if data.get("node_type") == "Encounter":
            timeline.append(
                {
                    "node_id": node_id,
                    "date": data.get("date", ""),
                    "type": data.get("type", ""),
                    "summary": data.get("summary", ""),
                    "department": data.get("department", ""),
                }
            )
        search_text = " ".join(
            str(value)
            for value in [
                node_id,
                data.get("date", ""),
                data.get("summary", ""),
                data.get("description", ""),
                data.get("note_type", ""),
                data.get("modality", ""),
                data.get("department", ""),
            ]
            if value
        )
        search_index.append(
            {
                "node_id": node_id,
                "node_type": data.get("node_type", ""),
                "date": data.get("date", ""),
                "text": search_text,
            }
        )
        if data.get("node_type") == "Clinical_Note":
            chunks = _chunk_text(str(data.get("full_text", "")))
            note_chunks[node_id] = []
            note_chunk_meta[node_id] = []
            for idx, chunk in enumerate(chunks):
                chunk_summary = chunk[:220].replace("\n", " ")
                note_chunks[node_id].append({"chunk_index": idx, "text": chunk})
                note_chunk_meta[node_id].append(
                    {
                        "chunk_index": idx,
                        "date": data.get("date", ""),
                        "note_type": data.get("note_type", ""),
                        "summary": chunk_summary,
                    }
                )
                search_index.append(
                    {
                        "node_id": node_id,
                        "node_type": "Clinical_Note_Chunk",
                        "date": data.get("date", ""),
                        "text": f"{data.get('note_type', '')} {chunk_summary}",
                        "chunk_index": idx,
                    }
                )

    for node_id in graph.nodes:
        linked = []
        for _, target, edge in graph.out_edges(node_id, data=True):
            linked.append(
                {
                    "direction": "out",
                    "edge_type": edge.get("edge_type", ""),
                    "node": _node_brief(target, graph.nodes[target]),
                }
            )
        for source, _, edge in graph.in_edges(node_id, data=True):
            linked.append(
                {
                    "direction": "in",
                    "edge_type": edge.get("edge_type", ""),
                    "node": _node_brief(source, graph.nodes[source]),
                }
            )
        neighbors[node_id] = linked

    return {
        "timeline": timeline,
        "nodes": nodes,
        "neighbors": neighbors,
        "search_index": search_index,
        "note_chunks": note_chunks,
        "note_chunk_meta": note_chunk_meta,
        "answer_options": answer_options,
    }


HELPER_SETUP_CODE = r"""
def timeline():
    return context["timeline"]

def answer_options():
    return context.get("answer_options", [])

def node(node_id):
    return context["nodes"].get(node_id)

def neighbors(node_id):
    return context["neighbors"].get(node_id, [])

def search(term, node_type=None, limit=10):
    term = str(term).lower().strip()
    results = []
    for item in context["search_index"]:
        if node_type and item.get("node_type") != node_type:
            continue
        if term in item.get("text", "").lower():
            results.append(item)
    results = sorted(results, key=lambda item: (item.get("date", ""), item.get("node_id", "")))
    return results[:limit]

def multi_search(terms, node_type=None, limit=10):
    scores = []
    normalized = [str(term).lower().strip() for term in terms if str(term).strip()]
    for item in context["search_index"]:
        if node_type and item.get("node_type") != node_type:
            continue
        haystack = item.get("text", "").lower()
        score = sum(1 for term in normalized if term in haystack)
        if score > 0:
            scores.append((score, item))
    scores = sorted(scores, key=lambda pair: (pair[0], pair[1].get("date", ""), pair[1].get("node_id", "")), reverse=True)
    return [item for _, item in scores[:limit]]

def list_chunks(node_id, limit=10):
    return context["note_chunk_meta"].get(node_id, [])[:limit]

def read_chunk(node_id, chunk_index):
    chunks = context["note_chunks"].get(node_id, [])
    for item in chunks:
        if item["chunk_index"] == chunk_index:
            return item["text"]
    return ""

def ask_chunk(node_id, chunk_index, prompt):
    chunk_text = read_chunk(node_id, chunk_index)
    if not chunk_text:
        return "No chunk found."
    return llm_query(f"{prompt}\n\nChunk:\n{chunk_text}")
"""


@dataclass
class MinimalRLMResult:
    answer: str
    tokens_used: int
    trace: list[dict[str, Any]]


class _TokenAwareClient:
    def __init__(self, model: str) -> None:
        self.model = model
        self.client = OpenAI()
        self.total_tokens = 0

    def completion(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_completion_tokens=max_tokens,
            messages=messages,
        )
        self.total_tokens += int(getattr(response.usage, "total_tokens", 0) or 0)
        return response.choices[0].message.content or ""


class MinimalRLMGraphController:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        recursive_model: str = "gpt-4.1-mini",
        max_iterations: int = 6,
    ) -> None:
        self.model = model
        self.recursive_model = recursive_model
        self.max_iterations = max_iterations

    def answer_question(self, question: dict[str, Any], graph: nx.DiGraph) -> MinimalRLMResult:
        context = _build_rlm_context(graph, answer_options=question.get("answer_options", []))
        repl_env = REPLEnv(
            recursive_model=self.recursive_model,
            context_json=context,
            setup_code=HELPER_SETUP_CODE,
        )
        root_client = _TokenAwareClient(self.model)
        messages = [{"role": "system", "content": ROOT_SYSTEM_PROMPT}]
        query = question["text"]
        if question.get("answer_options"):
            query = query + "\nAllowed answers:\n" + "\n".join(f"- {item}" for item in question["answer_options"])

        trace: list[dict[str, Any]] = []

        for iteration in range(self.max_iterations):
            if iteration == 0:
                prompt = (
                    f"Question: {query}\n\n"
                    "Use the REPL immediately. Start by orienting with timeline() or search()/multi_search()."
                )
            else:
                prompt = (
                    f"Question: {query}\n\n"
                    "Continue using the REPL. Read the minimum number of chunks needed and then finalize."
                )
            response = root_client.completion(messages + [{"role": "user", "content": prompt}], max_tokens=700)
            trace.append({"iteration": iteration, "response": response})
            code_blocks = find_code_blocks(response)
            if code_blocks:
                messages = process_code_execution(
                    response=response,
                    messages=messages,
                    repl_env=repl_env,
                    repl_env_logger=_NullLogger(),
                    logger=_NullLogger(),
                )
            else:
                messages.append({"role": "assistant", "content": response})

            final_answer = check_for_final_answer(response, repl_env, _NullLogger())
            if final_answer:
                approx_repl_tokens = sum(
                    approx_token_count(message.get("content", "")) for message in messages
                )
                return MinimalRLMResult(
                    answer=final_answer,
                    tokens_used=root_client.total_tokens + approx_repl_tokens,
                    trace=trace,
                )

        final_prompt = (
            f"Question: {query}\n\n"
            "You have enough evidence. Provide only the final answer now using FINAL(...) or FINAL_VAR(...)."
        )
        response = root_client.completion(messages + [{"role": "user", "content": final_prompt}], max_tokens=200)
        trace.append({"iteration": self.max_iterations, "response": response})
        final_answer = check_for_final_answer(response, repl_env, _NullLogger())
        if not final_answer:
            final_answer = response.strip()
        approx_repl_tokens = sum(approx_token_count(message.get("content", "")) for message in messages)
        return MinimalRLMResult(
            answer=final_answer,
            tokens_used=root_client.total_tokens + approx_repl_tokens,
            trace=trace,
        )


class _NullLogger:
    def log_tool_execution(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def log_execution(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def display_last(self) -> None:
        return None
