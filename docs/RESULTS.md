# Results

## Scope

This repository keeps the offline evaluation artifacts and documents the historical live free-text configuration that produced a `52%` Graph-RLM accuracy result. The generated artifacts from the most recent live run were removed before packaging this repository.

## Retained offline artifacts

The following offline outputs remain in the repository:

- `results/metrics.csv`
- `results/detailed_results.csv`
- `results/graph_stats.csv`
- `results/pareto_curve.png`
- `data/processed/questions.json`

These files come from the offline MIMIC-III demo pipeline, not from the private breast oncology note files.

## Historical live configuration

The codebase retains the controller and evaluator configuration corresponding to the earlier live setup in which Graph-RLM reached `52%` accuracy on a `25`-question sample over the local breast oncology note cohort.

Documented historical summary:

- `Full-Context`: `76.0%` accuracy, `108,524.4` average tokens per query
- `Graph-RLM`: `52.0%` accuracy, `3,561.6` average tokens per query
- `Naive RAG`: `52.0%` accuracy, `819.0` average tokens per query

This result is documented here for reproducibility of the code state, but the underlying live artifacts are not included because they were derived from private note files.

## Historical category-level snapshot

The following category rows were recorded for that same historical live configuration:

- `Full-Context`, `absence_detection`, `5` questions, `100.0%` accuracy, `88,184.0` average tokens
- `Graph-RLM`, `absence_detection`, `5` questions, `0.0%` accuracy, `3,334.6` average tokens
- `Naive RAG`, `absence_detection`, `5` questions, `0.0%` accuracy, `1,621.8` average tokens
- `Full-Context`, `causal_chain`, `5` questions, `80.0%` accuracy, `88,233.8` average tokens
- `Graph-RLM`, `causal_chain`, `5` questions, `60.0%` accuracy, `3,937.4` average tokens
- `Naive RAG`, `causal_chain`, `5` questions, `40.0%` accuracy, `1,057.6` average tokens

Those values are preserved as documentation only.

## Why live artifacts were removed

The live evaluator reads local `patient_*.csv` note files. Even when the output files contain only model answers and metadata, they are downstream products of private clinical notes and should not be published casually.

For that reason, this repository excludes:

- `results/openai_metrics.csv`
- `results/openai_detailed_results.csv`
- `results/openai_failure_audit.csv`
- `results/breast_note_graph_stats.csv`
- `data/processed/openai_questions.json`

## Public-release note

This repository is packaged for public code release, not public data release. Anyone reproducing the live experiment will need their own local note files and an API key supplied through environment variables.
