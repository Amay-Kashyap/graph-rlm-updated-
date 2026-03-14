# Methodology

## Objective

This project evaluates whether a graph-guided recursive retrieval strategy can answer longitudinal clinical questions at materially lower token cost than full-context prompting, while preserving enough evidence fidelity to support audit and error analysis.

Two data settings are supported:

- an offline setting built from the `mimic-iii-clinical-database-demo-1.4` structured sample
- a free-text setting built from external `patient_*.csv` breast oncology notes stored outside the repository

## Data pipeline

### Offline cohort

The offline pipeline uses the local MIMIC-III demo release. Because the demo release lacks real free-text note coverage, the pipeline derives graph content from structured tables such as admissions, diagnoses, procedures, labs, and related encounter metadata.

### Free-text cohort

The free-text pipeline reads local breast oncology note CSVs. Each file is treated as one patient timeline. Notes are parsed into temporal encounter records and text-bearing documentation nodes. The repository does not include those CSVs.

## Graph construction

For each patient, the system builds a directed graph with temporal and semantic relations.

Core node types include:

- patient
- encounter
- note
- diagnosis
- procedure
- treatment
- imaging report
- laboratory observation

Core edge types include:

- chronological progression between encounters
- encounter-to-note documentation links
- diagnosis or procedure attribution to an encounter or note
- treatment occurrence over time
- imaging-to-follow-up relations when later evidence supports downstream action

The free-text graph builder applies domain-specific pattern extraction for breast oncology terms, therapy names, staging language, pathology concepts, and imaging references.

## Systems compared

Three answering systems are implemented:

- `Full-Context`: sends the full filtered patient record to the model
- `Naive RAG`: retrieves top lexical chunks from the source notes
- `Graph-RLM`: navigates the patient graph iteratively and reads only selected evidence

The live Graph-RLM controller uses a minimal recursive language model scaffold vendored under `vendor/rlm-minimal` and adapted to the graph tools in this project.

## Question generation

The benchmark covers several reasoning patterns:

- temporal proximity
- causal chain
- longitudinal trend
- absence detection
- multi-hop reasoning

For the free-text setting, questions are generated only when the graph contains enough structured support to justify a traceable answer.

## Ground-truth protocol

Ground truth is defined as evidence-linked supervision, not just an answer string.

Each generated question stores:

- `ground_truth`: the expected answer text
- `supporting_node_ids`: the exact node identifiers that support that answer

This design matters for four reasons.

### 1. Label validity

A question is kept only if the answer can be tied to specific graph evidence. For example, a causal-chain question is not accepted merely because a later treatment appears somewhere in the chart. It is accepted only when the graph contains an explicit or heuristically justified evidence path, such as:

- an imaging node
- a linked note describing the finding or implication
- a downstream procedure or treatment node that follows in time

Without that chain, the item is discarded rather than weakly labeled.

### 2. Auditability

When a model misses a question, the evaluator can inspect the exact supporting nodes and determine whether the failure came from:

- graph construction
- retrieval/navigation
- evidence reading
- answer synthesis

This is substantially stronger than benchmarks where the answer key exists without a traceable record of why that answer was correct.

### 3. Reduced heuristic noise

Clinical timelines contain repeated mentions, historical references, negations, and copied-forward text. A heuristic that says "first later procedure" or "latest stage mention" often produces ambiguous labels. By attaching supporting nodes, the benchmark avoids treating weak temporal coincidence as true supervision.

### 4. Failure analysis by evidence coverage

Because each item knows the evidence nodes that justify it, later evaluation can measure whether the retrieval system touched relevant evidence at all before answering. That makes it possible to separate retrieval failure from reasoning failure.

## Evaluation

Primary metrics are:

- exact-match or normalized answer accuracy
- average tokens per query

Secondary analysis includes:

- category-level accuracy
- tool-call counts
- qualitative error review against supporting evidence

The repository currently retains the code corresponding to the configuration that previously produced a `52%` Graph-RLM accuracy run in the live setup, but the generated artifacts from the most recent live run have been removed from versioned outputs.

## Key challenges

### Sparse and inconsistent clinical evidence

Clinical notes often mention plans, prior history, and follow-up actions in different sections and on different dates. A single clinically meaningful answer may require linking an imaging result, a later oncology note, and a treatment administration record.

### Demo-data limitations

The MIMIC-III demo release is useful for prototyping graph construction, but it lacks the breadth of real note text needed for note-heavy evaluation. That forced a split design: offline structured-data experiments and external free-text note experiments.

### Ambiguity in temporal reasoning

Questions such as "what followed" or "was there evidence of" are deceptively difficult. The correct answer depends on temporal scope, whether historical mentions should count, and whether a downstream event is actually attributable to the earlier finding.

### Absence detection

Absence questions are fragile because a missing statement is not the same as evidence of non-occurrence. The system must define the observation window clearly and avoid overclaiming from incomplete note coverage.

### Long-context cost

Full-context prompting is often accurate because it over-includes evidence, but it is token-expensive. Graph-RLM tries to reduce cost by selecting only relevant parts of the patient timeline, which introduces a risk of missing decisive evidence.

### Evaluation noise from copied-forward notes

Oncology records frequently reuse prior text. That can make a note look like new evidence when it is actually historical carryover. The graph and question generator must distinguish new events from repeated documentation.

## Reproducibility and privacy

- code is versioned locally in the repository
- patient note CSVs are excluded from version control
- generated live evaluation outputs are excluded by default
- API keys should be supplied only through environment variables

This keeps the repository publishable without redistributing the underlying private clinical notes.
