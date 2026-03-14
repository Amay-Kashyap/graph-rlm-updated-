from __future__ import annotations

import networkx as nx

from evaluation.qa_logic import answer_question, chunk_graph_documents, retrieval_scores
from graph.graph_utils import approx_token_count


class NaiveRAGBaseline:
    name = "Naive RAG"

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def answer_question(self, question: dict, graph: nx.DiGraph) -> dict:
        chunks = chunk_graph_documents(graph)
        retrieved = [chunk for score, chunk in retrieval_scores(question["text"], chunks)[: self.top_k] if score > 0]
        accessible_node_ids = []
        for chunk in retrieved:
            accessible_node_ids.extend(chunk["node_ids"])

        result = answer_question(question, graph, accessible_node_ids=accessible_node_ids)
        if not retrieved:
            result["answer"] = "Insufficient evidence in retrieved context."
            result["correct"] = False
        tokens_used = approx_token_count(question["text"]) + sum(
            approx_token_count(chunk["text"]) for chunk in retrieved
        ) + approx_token_count(result["answer"])
        return {
            "answer": result["answer"],
            "correct": result["correct"],
            "tokens_used": tokens_used,
            "tool_calls": [{"tool_name": "retrieve_top_k", "tokens": tokens_used, "cumulative_tokens": tokens_used}],
            "retrieved_chunks": [chunk["chunk_id"] for chunk in retrieved],
        }
