"""Inject synthetic (random) ground-truth instances into patient graphs.

The idea: we fabricate known clinical facts â€” a new procedure, an imaging
study, a diagnosis, a lab result â€” embed them into the graph *and* into the
note text of an existing encounter, and record exactly what we planted so we
can later ask the model about these facts and deterministically verify the
answer.

Each planted fact produces a ``SyntheticFact`` record that downstream
question generators consume.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from graph.graph_utils import NODE_TYPES, iso_date, parse_timestamp

# ---------------------------------------------------------------------------
# Pools of synthetic clinical facts
# ---------------------------------------------------------------------------

SYNTHETIC_PROCEDURES = [
    "DOXORUBICIN INFUSION",
    "SENTINEL LYMPH NODE BIOPSY",
    "PORT-A-CATH PLACEMENT",
    "STEREOTACTIC BIOPSY",
    "AXILLARY DISSECTION",
    "TAMOXIFEN INITIATION",
    "LETROZOLE INITIATION",
    "PALBOCICLIB INITIATION",
    "PEMBROLIZUMAB INFUSION",
    "DOSE-DENSE AC-T CYCLE 1",
]

SYNTHETIC_IMAGING = [
    ("BONE SCAN", "whole body"),
    ("PET/CT", "chest/abdomen/pelvis"),
    ("BRAIN MRI", "brain"),
    ("MAMMOGRAM", "bilateral breast"),
    ("CHEST X-RAY", "chest"),
    ("LIVER ULTRASOUND", "abdomen"),
    ("CARDIAC ECHO", "heart"),
    ("CT CHEST", "chest"),
    ("MRI BREAST", "bilateral breast"),
    ("DEXA SCAN", "spine/hip"),
]

SYNTHETIC_DIAGNOSES = [
    "STAGE IIIA BREAST CANCER",
    "STAGE IB BREAST CANCER",
    "HER2-POSITIVE BREAST CANCER",
    "ER-POSITIVE/PR-POSITIVE BREAST CANCER",
    "BONE METASTASIS",
    "LIVER METASTASIS",
    "LYMPHEDEMA POST-SURGERY",
    "CHEMOTHERAPY-INDUCED NEUROPATHY",
    "RADIATION DERMATITIS",
    "NEUTROPENIC FEVER",
]

SYNTHETIC_SNIPPETS = [
    "Patient tolerated the {procedure} well with no immediate adverse reactions.",
    "Scheduled {procedure} was completed without complications. Patient resting comfortably.",
    "{procedure} performed per oncology protocol. Will reassess at next visit.",
    "Results of {imaging} on {date}: No evidence of progression. Stable findings.",
    "{imaging} performed on {date} shows new findings consistent with treatment response.",
    "Assessment: {diagnosis}. Plan: Continue current regimen and monitor.",
    "New diagnosis documented today: {diagnosis}. Multidisciplinary discussion planned.",
]


# ---------------------------------------------------------------------------
# Data class for planted facts
# ---------------------------------------------------------------------------

@dataclass
class SyntheticFact:
    """One planted fact with its deterministic ground truth."""

    fact_id: str
    patient_id: int
    category: str  # "procedure" | "imaging" | "diagnosis" | "temporal"
    encounter_id: str
    node_id: str
    date: str
    description: str
    ground_truth_question: str
    ground_truth_answer: str
    answer_options: list[str] = field(default_factory=list)
    supporting_node_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Injection logic
# ---------------------------------------------------------------------------

def _pick_encounter(graph: nx.DiGraph) -> tuple[str, dict[str, Any]] | None:
    """Return a random Encounter node that has at least one Clinical_Note."""
    encounters = [
        (nid, d)
        for nid, d in graph.nodes(data=True)
        if d.get("node_type") == "Encounter" and d.get("date")
    ]
    if not encounters:
        return None
    # prefer encounters that already have notes attached
    with_notes = []
    for enc_id, enc_data in encounters:
        for succ in graph.successors(enc_id):
            if graph.nodes[succ].get("node_type") == "Clinical_Note":
                with_notes.append((enc_id, enc_data))
                break
    pool = with_notes if with_notes else encounters
    return random.choice(pool)


def _find_note_for_encounter(graph: nx.DiGraph, enc_id: str) -> str | None:
    for succ in graph.successors(enc_id):
        if graph.nodes[succ].get("node_type") == "Clinical_Note":
            return succ
    return None


def _inject_text_into_note(graph: nx.DiGraph, note_id: str, snippet: str) -> None:
    """Append a synthetic snippet to an existing note's full_text."""
    data = graph.nodes[note_id]
    existing = str(data.get("full_text", ""))
    data["full_text"] = existing + "\n\n[SYNTHETIC] " + snippet
    # also update the summary to include a hint
    data["summary"] = str(data.get("summary", "")) + " " + snippet[:80]


def inject_synthetic_procedure(
    graph: nx.DiGraph, patient_id: int, rng: random.Random
) -> SyntheticFact | None:
    pick = _pick_encounter(graph)
    if not pick:
        return None
    enc_id, enc_data = pick
    enc_date = enc_data["date"]

    proc_desc = rng.choice(SYNTHETIC_PROCEDURES)
    uid = uuid.uuid4().hex[:8]
    proc_id = f"synth_proc:{patient_id}:{uid}"

    graph.add_node(
        proc_id,
        node_type=NODE_TYPES["procedure"],
        patient_id=patient_id,
        proc_id=proc_id,
        encounter_id=enc_id,
        date=enc_date,
        cpt_code="",
        description=proc_desc,
        outcome_summary="Synthetic ground-truth instance.",
        summary=f"Synthetic procedure: {proc_desc}",
        synthetic=True,
    )
    graph.add_edge(enc_id, proc_id, edge_type="PERFORMED_DURING")

    # inject text into note
    note_id = _find_note_for_encounter(graph, enc_id)
    snippet = rng.choice(SYNTHETIC_SNIPPETS).format(
        procedure=proc_desc, imaging="", diagnosis="", date=enc_date
    )
    if note_id:
        _inject_text_into_note(graph, note_id, snippet)
        graph.add_edge(note_id, proc_id, edge_type="PERFORMED_DURING")

    support = [enc_id, proc_id]
    if note_id:
        support.append(note_id)

    return SyntheticFact(
        fact_id=f"synth-proc-{uid}",
        patient_id=patient_id,
        category="procedure",
        encounter_id=enc_id,
        node_id=proc_id,
        date=enc_date,
        description=proc_desc,
        ground_truth_question=(
            f"Was {proc_desc} performed for patient {patient_id} "
            f"on or around {enc_date}?"
        ),
        ground_truth_answer="YES",
        answer_options=["YES", "NO"],
        supporting_node_ids=support,
    )


def inject_synthetic_imaging(
    graph: nx.DiGraph, patient_id: int, rng: random.Random
) -> SyntheticFact | None:
    pick = _pick_encounter(graph)
    if not pick:
        return None
    enc_id, enc_data = pick
    enc_date = enc_data["date"]

    modality, body_part = rng.choice(SYNTHETIC_IMAGING)
    uid = uuid.uuid4().hex[:8]
    img_id = f"synth_img:{patient_id}:{uid}"

    graph.add_node(
        img_id,
        node_type=NODE_TYPES["imaging"],
        patient_id=patient_id,
        report_id=img_id,
        encounter_id=enc_id,
        date=enc_date,
        modality=modality,
        body_part=body_part,
        summary=f"Synthetic imaging: {modality} of {body_part}",
        full_text=f"{modality} performed on {enc_date}. Findings are stable.",
        synthetic=True,
    )
    graph.add_edge(enc_id, img_id, edge_type="ORDERED_DURING")

    note_id = _find_note_for_encounter(graph, enc_id)
    snippet = rng.choice(SYNTHETIC_SNIPPETS).format(
        imaging=modality, date=enc_date, procedure="", diagnosis=""
    )
    if note_id:
        _inject_text_into_note(graph, note_id, snippet)
        graph.add_edge(note_id, img_id, edge_type="DOCUMENTED_IN")

    support = [enc_id, img_id]
    if note_id:
        support.append(note_id)

    return SyntheticFact(
        fact_id=f"synth-img-{uid}",
        patient_id=patient_id,
        category="imaging",
        encounter_id=enc_id,
        node_id=img_id,
        date=enc_date,
        description=modality,
        ground_truth_question=(
            f"Was a {modality} imaging study performed for patient {patient_id} "
            f"on or around {enc_date}?"
        ),
        ground_truth_answer="YES",
        answer_options=["YES", "NO"],
        supporting_node_ids=support,
    )


def inject_synthetic_diagnosis(
    graph: nx.DiGraph, patient_id: int, rng: random.Random
) -> SyntheticFact | None:
    pick = _pick_encounter(graph)
    if not pick:
        return None
    enc_id, enc_data = pick
    enc_date = enc_data["date"]

    diag_desc = rng.choice(SYNTHETIC_DIAGNOSES)
    uid = uuid.uuid4().hex[:8]
    dx_id = f"synth_dx:{patient_id}:{uid}"

    graph.add_node(
        dx_id,
        node_type=NODE_TYPES["diagnosis"],
        patient_id=patient_id,
        dx_id=dx_id,
        encounter_id=enc_id,
        date=enc_date,
        icd_code="",
        description=diag_desc,
        status="active",
        summary=f"Synthetic diagnosis: {diag_desc}",
        synthetic=True,
    )
    graph.add_edge(enc_id, dx_id, edge_type="DIAGNOSED_WITH")

    note_id = _find_note_for_encounter(graph, enc_id)
    snippet = rng.choice(SYNTHETIC_SNIPPETS).format(
        diagnosis=diag_desc, procedure="", imaging="", date=enc_date
    )
    if note_id:
        _inject_text_into_note(graph, note_id, snippet)
        graph.add_edge(note_id, dx_id, edge_type="DIAGNOSED_WITH")

    support = [enc_id, dx_id]
    if note_id:
        support.append(note_id)

    return SyntheticFact(
        fact_id=f"synth-dx-{uid}",
        patient_id=patient_id,
        category="diagnosis",
        encounter_id=enc_id,
        node_id=dx_id,
        date=enc_date,
        description=diag_desc,
        ground_truth_question=(
            f"Was {diag_desc} documented for patient {patient_id} "
            f"on or around {enc_date}?"
        ),
        ground_truth_answer="YES",
        answer_options=["YES", "NO"],
        supporting_node_ids=support,
    )


# ---------------------------------------------------------------------------
# Negative (absence) questions from synthetic facts
# ---------------------------------------------------------------------------

def _make_negative_fact(
    patient_id: int, category: str, rng: random.Random
) -> SyntheticFact:
    """Create a question about something that does NOT exist in the graph."""
    uid = uuid.uuid4().hex[:8]
    if category == "procedure":
        desc = rng.choice(SYNTHETIC_PROCEDURES)
        q = f"Was {desc} performed for patient {patient_id} on 1999-01-01?"
    elif category == "imaging":
        modality, _ = rng.choice(SYNTHETIC_IMAGING)
        desc = modality
        q = f"Was a {modality} imaging study performed for patient {patient_id} on 1999-01-01?"
    else:
        desc = rng.choice(SYNTHETIC_DIAGNOSES)
        q = f"Was {desc} documented for patient {patient_id} on 1999-01-01?"

    return SyntheticFact(
        fact_id=f"synth-neg-{uid}",
        patient_id=patient_id,
        category=f"negative_{category}",
        encounter_id="",
        node_id="",
        date="1999-01-01",
        description=desc,
        ground_truth_question=q,
        ground_truth_answer="NO",
        answer_options=["YES", "NO"],
        supporting_node_ids=[],
    )


# ---------------------------------------------------------------------------
# Public: inject a batch of synthetic facts into graphs
# ---------------------------------------------------------------------------

def inject_synthetic_facts(
    graphs: dict[int, nx.DiGraph],
    facts_per_patient: int = 6,
    seed: int = 42,
) -> list[SyntheticFact]:
    """Inject synthetic facts into every patient graph.

    For each patient we inject:
    - 1 synthetic procedure  (positive)
    - 1 synthetic imaging    (positive)
    - 1 synthetic diagnosis  (positive)
    - 1 negative procedure   (absence)
    - 1 negative imaging     (absence)
    - 1 negative diagnosis   (absence)

    Returns the full list of ``SyntheticFact`` objects for question generation.
    """
    rng = random.Random(seed)
    all_facts: list[SyntheticFact] = []

    injectors = [
        inject_synthetic_procedure,
        inject_synthetic_imaging,
        inject_synthetic_diagnosis,
    ]
    neg_categories = ["procedure", "imaging", "diagnosis"]

    for patient_id, graph in graphs.items():
        for injector in injectors:
            fact = injector(graph, patient_id, rng)
            if fact:
                all_facts.append(fact)

        for cat in neg_categories:
            all_facts.append(_make_negative_fact(patient_id, cat, rng))

    return all_facts
