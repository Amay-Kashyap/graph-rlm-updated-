# Graph-RLM

Graph-RLM is a local research prototype for graph-guided reasoning over longitudinal clinical records. The repository contains:

- an offline pipeline built on the `mimic-iii-clinical-database-demo-1.4` sample data
- a free-text graph builder for external `patient_*.csv` note files
- Graph-RLM, naive RAG, and full-context evaluation code
- an optional OpenAI-backed live evaluator

The repository is packaged as code only. Patient note CSVs are not included.

## Repository layout

- `src/graph/patient_graph.py`: graph builder for the MIMIC-III demo cohort
- `src/graph/breast_notes_graph.py`: graph builder for external free-text patient note CSVs
- `src/agent/state_machine.py`: heuristic offline Graph-RLM controller
- `src/agent/minimal_rlm_graph.py`: minimal-RLM-backed live controller
- `src/agent/tools.py`: graph navigation utilities and token accounting helpers
- `src/baselines/naive_rag.py`: lexical retrieval baseline
- `src/baselines/full_context.py`: full-record baseline
- `src/evaluation/question_generator.py`: offline question generation
- `src/evaluation/real_question_generator.py`: free-text question generation with explicit evidence support sets
- `src/evaluation/evaluator.py`: offline evaluation runner
- `src/evaluation/openai_evaluator.py`: live OpenAI evaluation runner
- `vendor/rlm-minimal`: vendored minimal recursive language model scaffold
- `docs/METHODOLOGY.md`: methodology, ground-truth design, and key challenges
- `docs/RESULTS.md`: retained offline results and the documented historical live configuration

## Data assumptions

The offline pipeline auto-detects the raw MIMIC demo directory from:

1. `GRAPH_RLM_RAW_DATA_DIR`
2. `graph-rlm/data/raw/mimic-iii-clinical-database-demo-1.4`
3. `graph-rlm/../mimic-iii-clinical-database-demo-1.4`

External breast oncology notes are expected as `patient_*.csv` files outside the repository root. They are intentionally excluded from version control.

## Offline run

```powershell
Set-Location "C:\Users\Amay Kashyap Deka\Downloads\Graph RLM project\graph-rlm"
$env:PYTHONPATH = "src"
python -m evaluation.evaluator
```

Outputs:

- `results/metrics.csv`
- `results/detailed_results.csv`
- `results/graph_stats.csv`
- `results/pareto_curve.png`
- `data/processed/questions.json`

## Live OpenAI run

```powershell
Set-Location "C:\Users\Amay Kashyap Deka\Downloads\Graph RLM project\graph-rlm"
$env:PYTHONPATH = "src"
$env:OPENAI_API_KEY = "<set-at-runtime>"
python -m evaluation.openai_evaluator --sample-size 25
```

The live run reads local `patient_*.csv` files if they are present. Generated live artifacts are not tracked by default.

## Ground truth design

The evaluation set for free-text notes does not rely on loose answer heuristics alone. Each generated question stores:

- `ground_truth`: the expected answer string
- `supporting_node_ids`: the exact graph nodes that justify that answer

This makes the benchmark auditable. When a model answer is wrong, the evaluator can trace the failure back to the specific note, encounter, imaging, diagnosis, or procedure nodes that established the label.

See [METHODOLOGY.md](/C:/Users/Amay%20Kashyap%20Deka/Downloads/Graph%20RLM%20project/graph-rlm/docs/METHODOLOGY.md) for the full methodology and challenge analysis.
See [RESULTS.md](/C:/Users/Amay%20Kashyap%20Deka/Downloads/Graph%20RLM%20project/graph-rlm/docs/RESULTS.md) for the retained offline results and the documented `52%` live Graph-RLM configuration.

## Repository hygiene

- patient note CSVs are ignored
- generated live evaluation outputs are ignored
- Python cache and local environment files are ignored

This keeps the repository safe to publish as code without redistributing the private source notes.
