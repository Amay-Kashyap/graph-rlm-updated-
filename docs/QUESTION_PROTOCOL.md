# Graph-Essential Question Protocol

This document defines the 100-question protocol for evaluating Graph-RLM on longitudinal records. The protocol is intended to make the benchmark auditable, graph-grounded, and resistant to shortcut answering.

## Design Basis

The protocol follows recurring design patterns from established QA benchmarks:

- HotpotQA: require explicit supporting facts rather than answer-only labels.
- MuSiQue: build multi-hop questions compositionally from simpler evidence units.
- emrQA and related clinical QA work: ground clinical questions in structured annotations or logical forms rather than free-form intuition alone.
- PubMedQA and BioASQ: retain biomedical evidence snippets and expert-checkable answer rationales.
- HealthBench-style evaluation: prefer rubric-backed, reviewer-auditable labels over unconstrained model judgments.
- Temporal clinical NLP: treat event order, uncertainty, negation, and observation windows as first-class label constraints.

## Dataset Composition

The target benchmark contains 100 questions, stratified into five equal categories:

| Category | Count | Core Operation |
|---|---:|---|
| Temporal ordering | 20 | Identify whether an event occurred before, after, or within a date window of another event. |
| Event-to-follow-up attribution | 20 | Link an upstream event, such as an imaging finding, to a downstream action. |
| Longitudinal trend | 20 | Compare the earliest and latest reliable states of a variable over time. |
| Bounded absence and negation | 20 | Determine whether an event is absent inside a specified observation window. |
| Distractor-heavy multi-hop trajectory | 20 | Traverse two or more graph hops while ignoring near-miss distractors. |

## Required Fields Per Question

Each question record must include:

- `question_id`: stable identifier.
- `patient_id`: record identifier or anonymized instance identifier.
- `category`: one of the five categories above.
- `question_text`: natural-language question.
- `ground_truth`: expected answer.
- `answer_type`: boolean, categorical, date, span, or short phrase.
- `supporting_node_ids`: graph nodes that establish the answer.
- `supporting_edge_ids`: graph edges that establish the reasoning path.
- `required_path_pattern`: expected graph traversal pattern, such as `Imaging_Report -> RESULTED_IN -> Procedure`.
- `temporal_window`: relevant date interval, if applicable.
- `distractor_node_ids`: plausible but incorrect evidence nodes.
- `label_confidence`: high, medium, or exclude.
- `source_type`: organic, adversarial-organic, or synthetic-controlled.

## Inclusion Criteria

A question is included only when all of the following are true:

- The answer is supported by explicit graph-linked evidence.
- The supporting evidence includes at least one source document or note node.
- The answer does not depend only on loose lexical coincidence.
- The temporal window is explicit when the question involves absence, ordering, or follow-up.
- The required evidence path can be inspected from stored node and edge IDs.
- At least one plausible distractor exists for adversarial and multi-hop items.

## Exclusion Criteria

A candidate question is excluded when:

- The answer depends on copied-forward history that cannot be separated from a current event.
- The only evidence is planned, recommended, or negated rather than completed/current.
- Multiple equally plausible graph paths support different answers.
- The supporting nodes are missing source provenance.
- The label relies on clinical common sense without direct record evidence.

## Recommended 100-Question Mix

Use a mixed-source design:

- 60 organic graph-backed questions generated from real longitudinal records.
- 20 adversarial organic questions selected because they contain realistic distractors.
- 20 synthetic-controlled questions with planted facts and deterministic labels.

This mix prevents the benchmark from becoming either too synthetic or too noisy. Organic items test realistic record complexity. Adversarial items test whether the model can ignore tempting wrong evidence. Synthetic-controlled items provide clean calibration cases where the correct answer is known exactly.

## Label Validation

Each label should pass a two-step audit:

1. Graph audit: confirm that the supporting node and edge set is sufficient to answer the question without reading unrelated context.
2. Text audit: confirm that the source text behind those nodes actually states the event, date, assertion status, or absence condition used in the label.

If either audit fails, the item is excluded or rewritten.

## Rationale

The benchmark should test a computation, not a vibe. A good Graph-RLM question forces the system to traverse a small but precise evidence path through a directed event graph. This follows the supporting-fact discipline of multi-hop QA benchmarks, the compositional structure of MuSiQue-style question construction, and the provenance requirements common in biomedical QA. For the clinical use case, this is especially important because copied-forward notes, historical mentions, negation, planned care, and out-of-window events can all create plausible but wrong answers. The protocol therefore rewards systems that use graph structure, provenance, temporality, and uncertainty correctly instead of systems that merely produce fluent clinical guesses.
