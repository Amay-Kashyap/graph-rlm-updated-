from __future__ import annotations

import networkx as nx

from evaluation.qa_logic import answer_question, flatten_graph_record
from graph.graph_utils import approx_token_count


class FullContextBaseline:
    name = "Full-Context"

    def answer_question(self, question: dict, graph: nx.DiGraph) -> dict:
        record_text = flatten_graph_record(graph)
        result = answer_question(question, graph, accessible_node_ids=graph.nodes)
        answer_tokens = approx_token_count(result["answer"])
        question_tokens = approx_token_count(question["text"])
        return {
            "answer": result["answer"],
            "correct": result["correct"],
            "tokens_used": approx_token_count(record_text) + question_tokens + answer_tokens,
            "tool_calls": [],
        }
