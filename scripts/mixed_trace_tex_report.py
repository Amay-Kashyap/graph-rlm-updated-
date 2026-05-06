from __future__ import annotations

import copy
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent.langgraph_rlm import LangGraphRLMController
from graph.breast_notes_graph import get_available_patient_ids, get_patient_graph
from graph.graph_utils import summarize_node
from mixed_trace_txt_report import (
    load_env_file,
    normalize_answer,
    run_questions,
    select_organic_questions,
    select_synthetic_questions,
)


OUTPUT_PATH = PROJECT_ROOT / "reports" / "graph_rlm_mixed_5_organic_5_synthetic_trace.tex"


def latex_escape(text: Any) -> str:
    value = "" if text is None else str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def verbatim_safe(text: Any, limit: int = 10000) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\r\n", "\n")
    if len(value) > limit:
        value = value[:limit] + "\n...[clipped]..."
    return value


def graph_counts(graph) -> tuple[dict[str, int], dict[str, int]]:
    node_counter = Counter()
    edge_counter = Counter()
    for _, data in graph.nodes(data=True):
        node_counter[str(data.get("node_type", "Unknown"))] += 1
    for _, _, edge in graph.edges(data=True):
        edge_counter[str(edge.get("edge_type", "Unknown"))] += 1
    return dict(sorted(node_counter.items())), dict(sorted(edge_counter.items()))


def render_small_table(headers: list[str], rows: list[list[str]], colspec: str) -> str:
    pieces = [
        "\\begin{longtable}{" + colspec + "}",
        "\\toprule",
        " & ".join(latex_escape(h) for h in headers) + r" \\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        " & ".join(latex_escape(h) for h in headers) + r" \\",
        "\\midrule",
        "\\endhead",
    ]
    for row in rows:
        pieces.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")
    pieces.extend(["\\bottomrule", "\\end{longtable}"])
    return "\n".join(pieces)


def trace_summary(trace_item: dict[str, Any]) -> str:
    phase = trace_item.get("phase", "")
    if phase == "structured_temporal":
        return (
            f"The temporal resolver searched for the procedure label "
            f"'{trace_item.get('label')}' near {trace_item.get('anchor_date')} "
            f"within {trace_item.get('window')} days. It considered "
            f"{len(trace_item.get('considered', []))} candidates, rejected "
            f"{len(trace_item.get('filtered_out', []))}, and kept "
            f"{len(trace_item.get('matches', []))} match."
        )
    if phase == "structured_causal":
        return (
            f"The causal resolver anchored on modality '{trace_item.get('modality')}' "
            f"at {trace_item.get('anchor_date')}. It collected candidate downstream "
            f"answers {trace_item.get('candidate_answers', [])} and retained only "
            f"high-confidence RESULTED_IN edges."
        )
    if phase == "structured_absence":
        return (
            f"The absence resolver started from {trace_item.get('anchor_date')}, "
            f"found {len(trace_item.get('anchor_nodes', []))} anchor imaging node(s), "
            f"evaluated {len(trace_item.get('considered_followups', []))} later imaging candidates, "
            f"and accepted {len(trace_item.get('followups', []))} follow-up node(s)."
        )
    if phase == "structured_trend":
        return (
            f"The trend resolver compared {len(trace_item.get('considered_stage_nodes', []))} "
            f"reliable stage nodes and selected the earliest and latest stage mentions."
        )
    if phase == "structured_lookup":
        return (
            f"The synthetic resolver performed exact graph lookup over entity type "
            f"{trace_item.get('entity_type')} with value '{trace_item.get('entity_value')}' "
            f"on date {trace_item.get('target_date')}."
        )
    return f"Resolver phase: {phase}."


def collect_touched_ids(trace: list[dict[str, Any]]) -> list[str]:
    touched_ids: list[str] = []
    for trace_item in trace:
        for key in ("matches", "anchor_nodes", "followups", "stage_nodes"):
            for node_id in trace_item.get(key, []):
                if node_id and node_id not in touched_ids:
                    touched_ids.append(node_id)
        for key in ("considered", "filtered_out", "considered_images", "considered_stage_nodes", "considered_followups"):
            for payload in trace_item.get(key, []):
                node_id = payload.get("node_id")
                if node_id and node_id not in touched_ids:
                    touched_ids.append(node_id)
        for edge in trace_item.get("supporting_edges", []):
            for node_id in (edge.get("from"), edge.get("to")):
                if node_id and node_id not in touched_ids:
                    touched_ids.append(node_id)
        for edge in trace_item.get("rejected_edges", []):
            for node_id in (edge.get("from"), edge.get("to")):
                if node_id and node_id not in touched_ids:
                    touched_ids.append(node_id)
    return touched_ids


def node_rows(graph, node_ids: list[str]) -> list[list[str]]:
    rows = []
    seen = set()
    for node_id in node_ids:
        if node_id in seen or node_id not in graph:
            continue
        seen.add(node_id)
        data = graph.nodes[node_id]
        rows.append(
            [
                node_id,
                str(data.get("node_type", "")),
                str(data.get("date", "")),
                summarize_node(data),
                str(data.get("assertion", "")),
                str(data.get("temporality", "")),
                str(data.get("confidence", "")),
            ]
        )
    return rows


def edge_rows(graph, node_ids: list[str]) -> list[list[str]]:
    rows = []
    node_set = set(node_ids)
    for source, target, edge in graph.edges(data=True):
        if source not in node_set and target not in node_set:
            continue
        rows.append(
            [
                source,
                target,
                str(edge.get("edge_type", "")),
                str(edge.get("confidence", "")),
                json.dumps(edge.get("evidence_note_ids", []), ensure_ascii=False),
            ]
        )
    return rows


def patient_section(patient_id: int, graph) -> str:
    node_counts, edge_counts = graph_counts(graph)
    node_table = render_small_table(
        ["Node Type", "Count"],
        [[k, str(v)] for k, v in node_counts.items()],
        "p{0.45\\textwidth} p{0.2\\textwidth}",
    )
    edge_table = render_small_table(
        ["Edge Type", "Count"],
        [[k, str(v)] for k, v in edge_counts.items()],
        "p{0.45\\textwidth} p{0.2\\textwidth}",
    )
    return f"""
\\subsection{{Patient {patient_id}}}
\\textbf{{Graph size.}} {graph.number_of_nodes()} nodes and {graph.number_of_edges()} directed edges.

\\paragraph{{Node type distribution}}
{node_table}

\\paragraph{{Edge type distribution}}
{edge_table}
"""


def question_section(index: int, row: dict[str, Any], graph) -> str:
    q = row["question"]
    touched_ids = collect_touched_ids(row["trace"])
    combined_ids = list(dict.fromkeys(list(q.get("supporting_node_ids", [])) + touched_ids))
    nodes_table = render_small_table(
        ["Node ID", "Type", "Date", "Summary", "Assertion", "Temporality", "Confidence"],
        node_rows(graph, combined_ids) or [["-", "-", "-", "No nodes", "-", "-", "-"]],
        "p{0.15\\textwidth} p{0.12\\textwidth} p{0.08\\textwidth} p{0.30\\textwidth} p{0.09\\textwidth} p{0.1\\textwidth} p{0.08\\textwidth}",
    )
    edges_table = render_small_table(
        ["From", "To", "Edge", "Confidence", "Evidence Notes"],
        edge_rows(graph, combined_ids) or [["-", "-", "-", "-", "No edges"]],
        "p{0.16\\textwidth} p{0.16\\textwidth} p{0.14\\textwidth} p{0.08\\textwidth} p{0.24\\textwidth}",
    )
    trace_notes = "\n".join(f"\\item {latex_escape(trace_summary(item))}" for item in row["trace"])
    raw_trace = json.dumps(row["trace"], indent=2, ensure_ascii=False)
    return f"""
\\subsection{{Question {index}: {latex_escape(q['category'])}}}
\\begin{{tcolorbox}}[colback=gray!5,colframe=gray!50,title=Question Summary]
\\textbf{{Patient}}: {q['patient_id']}\\\\
\\textbf{{Question}}: {latex_escape(q['text'])}\\\\
\\textbf{{Expected Answer}}: {latex_escape(q['ground_truth'])}\\\\
\\textbf{{Graph-RLM Answer}}: {latex_escape(row['answer'])}\\\\
\\textbf{{Correct}}: {latex_escape(row['correct'])}\\\\
\\textbf{{Reported Tokens}}: {row['tokens_used']}
\\end{{tcolorbox}}

\\paragraph{{How the Graph-RLM resolved this question}}
\\begin{{itemize}}[leftmargin=1.5em]
{trace_notes}
\\end{{itemize}}

\\paragraph{{Why these details matter}}
This section is intended to make the Graph-RLM mechanics explicit. The supporting nodes show the evidence that the benchmark generator attached to the question. The touched nodes show what the controller or structured resolver actually inspected. The local edge neighborhood shows how graph structure encodes chronology, note-to-entity attachment, and follow-up relationships.

\\paragraph{{Supporting and touched nodes}}
{nodes_table}

\\paragraph{{Local edge neighborhood}}
{edges_table}

\\paragraph{{Raw structured trace}}
\\begin{{Verbatim}}[fontsize=\\small,breaklines=true,breakanywhere=true]
{verbatim_safe(raw_trace)}
\\end{{Verbatim}}
"""


def build_document(raw_graphs: dict[int, Any], synth_graphs: dict[int, Any], organic_rows: list[dict[str, Any]], synthetic_rows: list[dict[str, Any]]) -> str:
    all_rows = organic_rows + synthetic_rows
    correct = sum(1 for row in all_rows if row["correct"])
    avg_tokens = sum(int(row["tokens_used"]) for row in all_rows) / max(len(all_rows), 1)
    patient_ids = sorted({int(row["question"]["patient_id"]) for row in all_rows})
    overview_rows = [
        ["Organic sample", str(len(organic_rows)), str(sum(1 for row in organic_rows if row["correct"])), f"{sum(int(row['tokens_used']) for row in organic_rows)/max(len(organic_rows),1):.1f}"],
        ["Synthetic sample", str(len(synthetic_rows)), str(sum(1 for row in synthetic_rows if row["correct"])), f"{sum(int(row['tokens_used']) for row in synthetic_rows)/max(len(synthetic_rows),1):.1f}"],
        ["Combined", str(len(all_rows)), str(correct), f"{avg_tokens:.1f}"],
    ]
    overview_table = render_small_table(
        ["Slice", "Questions", "Correct", "Average Tokens"],
        overview_rows,
        "p{0.24\\textwidth} p{0.16\\textwidth} p{0.14\\textwidth} p{0.18\\textwidth}",
    )
    patient_sections = "\n".join(patient_section(pid, raw_graphs.get(pid, synth_graphs[pid])) for pid in patient_ids)
    organic_sections = "\n".join(
        question_section(index, row, raw_graphs[int(row["question"]["patient_id"])])
        for index, row in enumerate(organic_rows, start=1)
    )
    synthetic_sections = "\n".join(
        question_section(index, row, synth_graphs[int(row["question"]["patient_id"])])
        for index, row in enumerate(synthetic_rows, start=1)
    )
    return f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=0.9in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{array}}
\\usepackage{{tabularx}}
\\usepackage{{xcolor}}
\\usepackage{{hyperref}}
\\usepackage{{fancyvrb}}
\\usepackage{{enumitem}}
\\usepackage[most]{{tcolorbox}}
\\setlength{{\\parindent}}{{0pt}}
\\setlength{{\\parskip}}{{0.6em}}
\\definecolor{{AccentBlue}}{{HTML}}{{234B6B}}
\\definecolor{{AccentGray}}{{HTML}}{{F4F6F8}}
\\hypersetup{{colorlinks=true,linkcolor=AccentBlue,urlcolor=AccentBlue}}

\\begin{{document}}

\\begin{{center}}
{{\\LARGE \\textbf{{Graph-RLM Mixed Trace Walkthrough}}}}\\\\[0.5em]
{{\\large 5 Organic Questions + 5 Synthetic Questions}}\\\\[0.5em]
{{\\normalsize GPT-5.4 backed run with graph-native structured resolution}}
\\end{{center}}

\\begin{{tcolorbox}}[colback=AccentGray,colframe=AccentBlue,title=Purpose of this report]
This report is designed to make the Graph-RLM pipeline understandable with \\textbf{{crystal clarity}}.
It does three things:
\\begin{{enumerate}}[leftmargin=1.5em]
\\item explains how patient notes are converted into a directed clinical graph,
\\item shows how the Graph-RLM resolves questions over that graph,
\\item exposes the \\emph{{observable execution trace}} of the controller: candidates considered, filters applied, nodes touched, edges used, and final answers.
\\end{{enumerate}}
This report does \\textbf{{not}} claim to reveal hidden private chain-of-thought.
Instead, it shows the full decision trace that is available from the engineered Graph-RLM pipeline itself.
\\end{{tcolorbox}}

\\section*{{How to read this document}}
\\begin{{itemize}}[leftmargin=1.5em]
\\item \\textbf{{Nodes}} are the typed objects in the patient graph: encounters, clinical notes, diagnoses, imaging reports, and procedures.
\\item \\textbf{{Edges}} encode how those objects relate: chronology, note attachment, encounter membership, and follow-up links such as \\texttt{{RESULTED\\_IN}}.
\\item \\textbf{{Assertion / temporality / confidence}} are part of the cleaned extraction layer. These fields tell us whether a mention is current versus historical, asserted versus negated, and how strong the extraction or linkage is.
\\item \\textbf{{Trace}} is the Graph-RLM's observable reasoning path. For structured organic questions, it records what candidates were considered, what was filtered out, and what was finally accepted.
\\item \\textbf{{Supporting nodes}} come from the question generator. They represent the graph evidence attached to the question template itself.
\\end{{itemize}}

\\section*{{Graph-RLM Process in plain language}}
\\begin{{enumerate}}[leftmargin=1.5em]
\\item Parse raw clinical notes into typed entities with attributes such as temporality and confidence.
\\item Build a directed graph whose nodes represent encounters, notes, diagnoses, procedures, and imaging reports.
\\item Add edges for encounter membership, chronology, mention links, and only high-confidence follow-up relations.
\\item For each question, route to a structured resolver when possible:
  temporal questions use date-window matching,
  causal questions use high-confidence imaging-to-procedure edges,
  absence questions use bounded follow-up search,
  trend questions compare reliable stage nodes.
\\item Return the answer, along with an explicit trace of what the resolver considered and why it chose the final node or edge.
\\end{{enumerate}}

\\section*{{Run overview}}
Questions in this report cover patients: {", ".join(str(pid) for pid in patient_ids)}.
The combined run answered {correct} of {len(all_rows)} questions correctly with an average of {avg_tokens:.1f} reported tokens per question.

{overview_table}

\\section*{{Patient graph summaries}}
{patient_sections}

\\section*{{Organic question walkthroughs}}
Organic questions show how the cleaned graph and structured resolvers behave on naturally occurring evidence.

{organic_sections}

\\section*{{Synthetic question walkthroughs}}
Synthetic questions show how the same controller behaves on planted deterministic facts.

{synthetic_sections}

\\section*{{What this report makes clear}}
\\begin{{itemize}}[leftmargin=1.5em]
\\item The Graph-RLM is not reading the entire chart blindly. It operates over typed graph objects and confidence-scored edges.
\\item The cleaned extraction layer matters: historical or low-confidence mentions are explicitly filtered out.
\\item The follow-up logic matters: causal questions depend on the quality of the \\texttt{{RESULTED\\_IN}} edges and the evidence notes attached to them.
\\item The trace is interpretable because each step records candidates, filtering decisions, and the final graph objects that drove the answer.
\\end{{itemize}}

\\end{{document}}
"""


def main() -> None:
    load_env_file(PROJECT_ROOT / ".env")
    os.environ.setdefault("RLM_REPL_TMPDIR", str(PROJECT_ROOT / ".repl_tmp"))
    model = "gpt-5.4"
    controller = LangGraphRLMController(model=model, recursive_model=model, max_iterations=4)

    raw_graphs = {pid: get_patient_graph(pid) for pid in get_available_patient_ids()}
    synth_graphs = {pid: copy.deepcopy(graph) for pid, graph in raw_graphs.items()}

    organic_questions = select_organic_questions(raw_graphs)
    synthetic_questions = select_synthetic_questions(synth_graphs)

    organic_rows = run_questions(controller, raw_graphs, organic_questions)
    synthetic_rows = run_questions(controller, synth_graphs, synthetic_questions)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        build_document(raw_graphs, synth_graphs, organic_rows, synthetic_rows),
        encoding="utf-8",
    )
    total = len(organic_rows) + len(synthetic_rows)
    correct = sum(1 for row in organic_rows + synthetic_rows if row["correct"])
    avg_tokens = sum(int(row["tokens_used"]) for row in organic_rows + synthetic_rows) / max(total, 1)
    print(
        json.dumps(
            {"path": str(OUTPUT_PATH), "questions": total, "correct": correct, "avg_tokens": round(avg_tokens, 2)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
