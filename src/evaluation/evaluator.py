from __future__ import annotations

from pathlib import Path

import pandas as pd

from agent.state_machine import HeuristicGraphRLM
from baselines.full_context import FullContextBaseline
from baselines.naive_rag import NaiveRAGBaseline
from evaluation.plot_results import plot_pareto
from evaluation.question_generator import generate_question_set, save_questions
from graph.patient_graph import PROJECT_ROOT, get_available_patient_ids, get_patient_graph, graph_stats


PRICING = {
    "input_per_million": 2.50,
    "output_per_million": 10.00,
}


def estimate_cost(tokens_used: int) -> float:
    blended_rate = (PRICING["input_per_million"] + PRICING["output_per_million"]) / 2.0
    return tokens_used / 1_000_000.0 * blended_rate


def run_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_ids = get_available_patient_ids()
    graphs = {patient_id: get_patient_graph(patient_id) for patient_id in patient_ids}
    questions = generate_question_set(graphs)

    processed_questions = PROJECT_ROOT / "data" / "processed" / "questions.json"
    save_questions(questions, processed_questions)

    systems = [
        ("Naive RAG", NaiveRAGBaseline()),
        ("Graph-RLM", HeuristicGraphRLM()),
        ("Full-Context", FullContextBaseline()),
    ]

    rows = []
    for system_name, system in systems:
        for question in questions:
            graph = graphs[question["patient_id"]]
            result = system.answer_question(question, graph)
            rows.append(
                {
                    "system": system_name,
                    "question_id": question["id"],
                    "patient_id": question["patient_id"],
                    "category": question["category"],
                    "question": question["text"],
                    "ground_truth": question["ground_truth"],
                    "supporting_node_ids": str(question.get("supporting_node_ids", [])),
                    "answer": result["answer"],
                    "correct": int(bool(result["correct"])),
                    "tokens_used": int(result["tokens_used"]),
                    "cost_per_query": estimate_cost(int(result["tokens_used"])),
                    "tool_calls": len(result.get("tool_calls", [])),
                }
            )

    detailed = pd.DataFrame(rows)
    summary_rows = []
    for system_name, system_df in detailed.groupby("system"):
        accuracy = system_df["correct"].mean() * 100.0
        avg_tokens = system_df["tokens_used"].mean()
        avg_cost = system_df["cost_per_query"].mean()
        summary_rows.append(
            {
                "system": system_name,
                "category": "ALL",
                "questions": int(system_df.shape[0]),
                "accuracy_pct": round(accuracy, 2),
                "avg_tokens_per_query": round(avg_tokens, 2),
                "avg_cost_per_query": round(avg_cost, 6),
                "token_efficiency_ratio": round(accuracy / max(avg_tokens, 1.0), 6),
            }
        )
    for (system_name, category), system_df in detailed.groupby(["system", "category"]):
        accuracy = system_df["correct"].mean() * 100.0
        avg_tokens = system_df["tokens_used"].mean()
        avg_cost = system_df["cost_per_query"].mean()
        summary_rows.append(
            {
                "system": system_name,
                "category": category,
                "questions": int(system_df.shape[0]),
                "accuracy_pct": round(accuracy, 2),
                "avg_tokens_per_query": round(avg_tokens, 2),
                "avg_cost_per_query": round(avg_cost, 6),
                "token_efficiency_ratio": round(accuracy / max(avg_tokens, 1.0), 6),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["category", "system"]).reset_index(drop=True)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(results_dir / "detailed_results.csv", index=False)
    summary.to_csv(results_dir / "metrics.csv", index=False)
    plot_pareto(results_dir / "metrics.csv", results_dir / "pareto_curve.png")
    graph_stats().to_csv(results_dir / "graph_stats.csv", index=False)
    return detailed, summary


def main() -> None:
    _, summary = run_evaluation()
    overall = summary[summary["category"] == "ALL"].copy()
    print("Evaluation complete.")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
