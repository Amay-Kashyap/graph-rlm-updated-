# Graph-RLM

Graph-RLM is a research prototype for graph-guided recursive reasoning over longitudinal records. The clinical instantiation in this repository focuses on breast oncology timelines, but the architecture is intended for any domain where evidence is distributed across dated documents, typed events, and inter-event relations.

The repository contains:

- an offline pipeline built on the `mimic-iii-clinical-database-demo-1.4` sample data
- a free-text graph builder for external `patient_*.csv` note files
- heuristic Graph-RLM, LangGraph Graph-RLM, naive RAG, and full-context evaluation code
- optional OpenAI-backed live evaluation runners
- documentation for methodology, retained results, and question construction

The repository is packaged as code only. Patient note CSVs, local API keys, generated live reports, and private downstream live artifacts are not included.

## Repository Layout

- `src/graph/patient_graph.py`: graph builder for the MIMIC-III demo cohort
- `src/graph/breast_notes_graph.py`: graph builder for external free-text patient note CSVs
- `src/graph/synthetic_injection.py`: deterministic synthetic fact injection for controlled questions
- `src/agent/state_machine.py`: heuristic offline Graph-RLM controller
- `src/agent/minimal_rlm_graph.py`: minimal-RLM-backed live controller
- `src/agent/langgraph_rlm.py`: LangGraph state-machine Graph-RLM controller
- `src/agent/tools.py`: graph navigation utilities and token accounting helpers
- `src/evaluation/question_generator.py`: offline question generation
- `src/evaluation/real_question_generator.py`: free-text question generation with explicit evidence support sets
- `src/evaluation/langgraph_evaluator.py`: mixed organic/synthetic LangGraph evaluation runner
- `src/evaluation/openai_evaluator.py`: live OpenAI evaluation runner
- `scripts/mixed_trace_txt_report.py`: local trace report generator for mixed organic/synthetic questions
- `scripts/mixed_trace_tex_report.py`: local LaTeX trace report generator
- `vendor/rlm-minimal`: vendored minimal recursive language model scaffold
- `docs/METHODOLOGY.md`: methodology, ground-truth design, and key challenges
- `docs/QUESTION_PROTOCOL.md`: literature-backed 100-question curation protocol
- `docs/RESULTS.md`: retained offline results and the documented historical live configuration

## Data Assumptions

The offline pipeline auto-detects the raw MIMIC demo directory from:

1. `GRAPH_RLM_RAW_DATA_DIR`
2. `graph-rlm/data/raw/mimic-iii-clinical-database-demo-1.4`
3. `graph-rlm/../mimic-iii-clinical-database-demo-1.4`

External breast oncology notes are expected as `patient_*.csv` files outside the repository root. They are intentionally excluded from version control.

## Offline Run

```powershell
$env:PYTHONPATH = "src"
python -m evaluation.evaluator
```

Outputs:

- `results/metrics.csv`
- `results/detailed_results.csv`
- `results/graph_stats.csv`
- `results/pareto_curve.png`
- `data/processed/questions.json`

## LangGraph Graph-RLM Run

```powershell
$env:PYTHONPATH = "src"
$env:OPENAI_API_KEY = "<set-at-runtime>"
python -m evaluation.langgraph_evaluator --sample-size 30
```

The LangGraph runner can combine organic graph-backed questions with synthetic controlled questions. Generated live and trace artifacts are ignored by default because they may be downstream products of private local notes.

## Ground Truth Design

The free-text benchmark does not rely on loose answer heuristics alone. Each generated question stores:

- `ground_truth`: the expected answer string
- `supporting_node_ids`: graph nodes that justify the answer
- `supporting_edge_ids`, when available: graph edges that establish the reasoning path
- `required_path_pattern`, when available: the graph traversal pattern being tested

This makes the benchmark auditable. When a model answer is wrong, the evaluator can trace the failure to graph construction, retrieval/navigation, evidence reading, or answer synthesis.

See `docs/METHODOLOGY.md` for the full methodology and challenge analysis.
See `docs/QUESTION_PROTOCOL.md` for the proposed 100-question curation protocol.
See `docs/RESULTS.md` for retained offline results and the documented historical `52%` live Graph-RLM configuration.

## Repository Hygiene

- patient note CSVs are ignored
- `.env` and local API-key files are ignored
- generated live evaluation outputs are ignored
- generated reports are ignored
- Python cache and temporary REPL folders are ignored

This keeps the repository safe to publish as code without redistributing private source notes or derived live artifacts.
