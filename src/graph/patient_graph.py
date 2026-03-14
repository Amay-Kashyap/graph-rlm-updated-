from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from .graph_utils import NODE_TYPES, canonicalize_text, friendly_datetime, iso_date


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_NAME = "mimic-iii-clinical-database-demo-1.4"


def discover_raw_data_dir() -> Path:
    env_value = os.getenv("GRAPH_RLM_RAW_DATA_DIR", "")
    candidates = []
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(
        [
            PROJECT_ROOT / "data" / "raw" / DEFAULT_DATASET_NAME,
            PROJECT_ROOT.parent / DEFAULT_DATASET_NAME,
            PROJECT_ROOT.parent / "mimic-iii-clinical-database-demo-1.4",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the MIMIC-III demo dataset. "
        "Set GRAPH_RLM_RAW_DATA_DIR or place the dataset next to graph-rlm."
    )


@lru_cache(maxsize=1)
def load_source_tables() -> dict[str, pd.DataFrame]:
    raw_dir = discover_raw_data_dir()

    patients = pd.read_csv(
        raw_dir / "PATIENTS.csv",
        usecols=["subject_id", "gender", "dob", "dod", "expire_flag"],
        parse_dates=["dob", "dod"],
    )
    admissions = pd.read_csv(
        raw_dir / "ADMISSIONS.csv",
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "admission_type",
            "admission_location",
            "diagnosis",
            "hospital_expire_flag",
        ],
        parse_dates=["admittime", "dischtime"],
    )
    diagnoses = pd.read_csv(
        raw_dir / "DIAGNOSES_ICD.csv",
        usecols=["subject_id", "hadm_id", "seq_num", "icd9_code"],
    )
    diagnosis_dict = pd.read_csv(
        raw_dir / "D_ICD_DIAGNOSES.csv",
        usecols=["icd9_code", "short_title", "long_title"],
    )
    procedures = pd.read_csv(
        raw_dir / "PROCEDURES_ICD.csv",
        usecols=["subject_id", "hadm_id", "seq_num", "icd9_code"],
    )
    procedure_dict = pd.read_csv(
        raw_dir / "D_ICD_PROCEDURES.csv",
        usecols=["icd9_code", "short_title", "long_title"],
    )
    labs = pd.read_csv(
        raw_dir / "LABEVENTS.csv",
        usecols=[
            "row_id",
            "subject_id",
            "hadm_id",
            "itemid",
            "charttime",
            "valuenum",
            "valueuom",
            "flag",
        ],
        parse_dates=["charttime"],
        low_memory=False,
    )
    lab_items = pd.read_csv(
        raw_dir / "D_LABITEMS.csv",
        usecols=["itemid", "label", "fluid", "category"],
    )

    diagnoses = diagnoses.merge(diagnosis_dict, on="icd9_code", how="left")
    procedures = procedures.merge(procedure_dict, on="icd9_code", how="left")
    labs = labs.merge(lab_items, on="itemid", how="left")

    return {
        "raw_dir": raw_dir,
        "patients": patients,
        "admissions": admissions,
        "diagnoses": diagnoses,
        "procedures": procedures,
        "labs": labs,
    }


def select_patient_cohort(cohort_size: int = 5) -> list[int]:
    tables = load_source_tables()
    admissions = tables["admissions"]
    diagnoses = tables["diagnoses"]
    procedures = tables["procedures"]
    labs = tables["labs"]

    summary = admissions.groupby("subject_id")["hadm_id"].nunique().rename("encounters").to_frame()
    summary["diagnoses"] = diagnoses.groupby("subject_id").size()
    summary["procedures"] = procedures.groupby("subject_id").size()
    summary["labs"] = labs.groupby("subject_id").size()
    summary = summary.fillna(0).astype(int)
    summary["score"] = (
        summary["encounters"] * 100
        + summary["procedures"] * 5
        + summary["diagnoses"] * 2
        + summary["labs"]
    )
    eligible = summary[summary["encounters"] >= 2].sort_values(
        ["encounters", "score"], ascending=False
    )
    return eligible.head(cohort_size).index.astype(int).tolist()


def _dedupe_titles(values: list[str], limit: int = 4) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if pd.isna(value):
            continue
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
        if len(output) >= limit:
            break
    return output


def _select_labs_for_encounter(
    encounter: pd.Series,
    patient_labs: pd.DataFrame,
    max_labs: int = 6,
) -> pd.DataFrame:
    hadm_id = encounter["hadm_id"]
    admittime = encounter["admittime"]
    dischtime = encounter["dischtime"]
    stay_labs = patient_labs[patient_labs["hadm_id"] == hadm_id].copy()
    if stay_labs.empty:
        stay_labs = patient_labs[
            (patient_labs["charttime"] >= admittime)
            & (patient_labs["charttime"] <= dischtime)
        ].copy()
    if stay_labs.empty:
        return stay_labs

    stay_labs = stay_labs.dropna(subset=["label", "charttime", "valuenum"])
    if stay_labs.empty:
        return stay_labs

    abnormal = stay_labs[stay_labs["flag"].fillna("").str.lower() == "abnormal"]
    chosen = (
        abnormal.sort_values(["label", "charttime"])
        .groupby("label")
        .tail(1)
        .copy()
    )
    if chosen.shape[0] < max_labs:
        needed = max_labs - chosen.shape[0]
        filler = (
            stay_labs.sort_values(["label", "charttime"])
            .groupby("label")
            .tail(1)
            .copy()
        )
        filler = filler[~filler["label"].isin(chosen["label"])]
        filler = filler.head(needed)
        chosen = pd.concat([chosen, filler], ignore_index=True)

    chosen["abnormal_score"] = chosen["flag"].fillna("").str.lower().eq("abnormal").astype(int)
    chosen = chosen.sort_values(
        ["abnormal_score", "charttime", "label"], ascending=[False, False, True]
    ).head(max_labs)
    return chosen


def _infer_department(encounter: pd.Series) -> str:
    location = str(encounter.get("admission_location", "") or "")
    if "EMERGENCY" in location:
        return "Emergency Department"
    if "CLINIC" in location:
        return "Outpatient Clinic"
    if "TRANSFER" in location:
        return "Transfer Unit"
    return "General Medicine"


def _encounter_summary(encounter: pd.Series, dx_titles: list[str]) -> str:
    primary = str(encounter.get("diagnosis", "") or "").strip()
    extras = ", ".join(dx_titles[:2])
    if extras:
        return f"{primary.title()} with key findings of {extras.lower()}."
    return primary.title()


def _note_text(encounter: pd.Series, dx_titles: list[str], proc_titles: list[str], labs: pd.DataFrame) -> str:
    abnormal_labs = labs[labs["flag"].fillna("").str.lower() == "abnormal"]
    lab_line = "No representative labs were selected."
    if not abnormal_labs.empty:
        snippets = [
            f"{row.label} {row.valuenum:g}{row.valueuom or ''}"
            for row in abnormal_labs.head(4).itertuples(index=False)
        ]
        lab_line = "Notable abnormal labs: " + ", ".join(snippets) + "."
    proc_line = "No coded procedures were documented during this encounter."
    if proc_titles:
        proc_line = "Documented procedures: " + ", ".join(proc_titles[:4]) + "."
    dx_line = "Working diagnoses included " + ", ".join(dx_titles[:5]) + "."
    return (
        f"Encounter date: {friendly_datetime(encounter['admittime'])}. "
        f"Admission reason: {encounter['diagnosis']}. "
        f"{dx_line} {proc_line} {lab_line} "
        "This note was synthesized from structured demo data because free-text notes "
        "are not present in the MIMIC-III demo release."
    )


def _imaging_blueprints(encounter: pd.Series, dx_titles: list[str], proc_titles: list[str]) -> list[dict[str, str]]:
    joined = canonicalize_text(
        " ".join([str(encounter.get("diagnosis", ""))] + dx_titles + proc_titles)
    )
    mappings = [
        (
            ("pneumonia", "respirat", "vent", "trache", "pleural"),
            {
                "modality": "X-ray",
                "body_part": "chest",
                "summary": "Synthetic chest imaging showed pulmonary opacities and interval respiratory complications.",
                "full_text": "Findings: Portable chest radiograph demonstrates patchy bilateral air-space opacity and low-volume basilar atelectasis. Impression: Imaging supports pulmonary infection or aspiration-related change.",
            },
        ),
        (
            ("encephal", "facial numbness", "brain", "stroke", "seiz"),
            {
                "modality": "CT",
                "body_part": "head",
                "summary": "Synthetic head CT showed no large acute hemorrhage but documented neurologic workup.",
                "full_text": "Findings: Non-contrast CT head performed for neurologic symptoms with chronic changes and no large acute bleed. Impression: Imaging contributes to neurologic evaluation and follow-up planning.",
            },
        ),
        (
            ("hepatic", "ascites", "portal vein", "abd", "gastro"),
            {
                "modality": "Ultrasound",
                "body_part": "abdomen",
                "summary": "Synthetic abdominal ultrasound documented ascites and chronic hepatobiliary disease burden.",
                "full_text": "Findings: Abdominal ultrasound shows moderate ascites and heterogeneous hepatic echotexture. Impression: Findings are compatible with chronic liver disease and volume overload.",
            },
        ),
        (
            ("fracture", "humer", "bone"),
            {
                "modality": "X-ray",
                "body_part": "upper extremity",
                "summary": "Synthetic extremity radiograph documented osseous injury and alignment assessment.",
                "full_text": "Findings: Radiograph demonstrates fracture-related deformity with follow-up evaluation of alignment. Impression: Imaging supports orthopedic management.",
            },
        ),
        (
            ("cancer", "mal neo", "lymphoma", "tamponade", "esophageal"),
            {
                "modality": "CT",
                "body_part": "chest/abdomen",
                "summary": "Synthetic cross-sectional imaging documented oncologic or compressive thoracic pathology.",
                "full_text": "Findings: Contrast-enhanced CT demonstrates complex thoracic and upper abdominal pathology relevant to the coded malignancy or effusion. Impression: Imaging prompted additional diagnostic or procedural follow-up.",
            },
        ),
    ]

    results = []
    for keywords, payload in mappings:
        if any(keyword in joined for keyword in keywords):
            results.append(payload)
            break
    return results


def _procedure_date(encounter: pd.Series, seq_num: int) -> pd.Timestamp:
    admittime = encounter["admittime"]
    dischtime = encounter["dischtime"]
    stay_days = max(1, int((dischtime - admittime).days) + 1)
    offset = min(max(seq_num - 1, 0), stay_days - 1)
    return admittime + pd.Timedelta(days=offset)


@lru_cache(maxsize=16)
def get_patient_graph(patient_id: int) -> nx.DiGraph:
    tables = load_source_tables()
    admissions = tables["admissions"]
    diagnoses = tables["diagnoses"]
    procedures = tables["procedures"]
    labs = tables["labs"]
    patients = tables["patients"]

    patient_row = patients[patients["subject_id"] == patient_id].iloc[0]
    patient_adm = admissions[admissions["subject_id"] == patient_id].sort_values("admittime")
    patient_dx = diagnoses[diagnoses["subject_id"] == patient_id].copy()
    patient_proc = procedures[procedures["subject_id"] == patient_id].copy()
    patient_labs = labs[labs["subject_id"] == patient_id].copy()

    graph = nx.DiGraph(
        patient_id=int(patient_id),
        patient_gender=patient_row["gender"],
        date_of_birth=iso_date(patient_row["dob"]),
        mortality_flag=int(patient_row["expire_flag"]),
        raw_data_dir=str(tables["raw_dir"]),
        data_mode="hybrid_real_plus_synthetic_documents",
    )

    previous_encounter_id = None
    for encounter in patient_adm.itertuples(index=False):
        encounter_dx = patient_dx[patient_dx["hadm_id"] == encounter.hadm_id].sort_values("seq_num")
        encounter_proc = patient_proc[patient_proc["hadm_id"] == encounter.hadm_id].sort_values("seq_num")
        encounter_labs = _select_labs_for_encounter(pd.Series(encounter._asdict()), patient_labs)

        dx_titles = _dedupe_titles(
            encounter_dx["long_title"].fillna(encounter_dx["short_title"]).tolist(), limit=6
        )
        proc_titles = _dedupe_titles(
            encounter_proc["long_title"].fillna(encounter_proc["short_title"]).tolist(), limit=5
        )

        enc_id = f"enc:{patient_id}:{encounter.hadm_id}"
        graph.add_node(
            enc_id,
            node_type=NODE_TYPES["encounter"],
            patient_id=int(patient_id),
            encounter_id=int(encounter.hadm_id),
            date=iso_date(encounter.admittime),
            start_time=encounter.admittime.isoformat(),
            end_time=encounter.dischtime.isoformat(),
            type=str(encounter.admission_type).lower(),
            summary=_encounter_summary(pd.Series(encounter._asdict()), dx_titles),
            department=_infer_department(pd.Series(encounter._asdict())),
            diagnosis_text=str(encounter.diagnosis),
        )
        if previous_encounter_id is not None:
            graph.add_edge(previous_encounter_id, enc_id, edge_type="FOLLOWED_BY")
        previous_encounter_id = enc_id

        note_id = f"note:{patient_id}:{encounter.hadm_id}"
        note_full_text = _note_text(pd.Series(encounter._asdict()), dx_titles, proc_titles, encounter_labs)
        graph.add_node(
            note_id,
            node_type=NODE_TYPES["note"],
            patient_id=int(patient_id),
            note_id=note_id,
            encounter_id=int(encounter.hadm_id),
            date=iso_date(encounter.admittime),
            note_type="discharge",
            author="Synthetic summarizer",
            summary="Synthetic longitudinal note summarizing diagnoses, procedures, and abnormal labs.",
            full_text=note_full_text,
        )
        graph.add_edge(enc_id, note_id, edge_type="DOCUMENTED_IN")

        for dx_row in encounter_dx.itertuples(index=False):
            dx_id = f"dx:{patient_id}:{encounter.hadm_id}:{dx_row.seq_num}"
            description = dx_row.long_title or dx_row.short_title or dx_row.icd9_code
            graph.add_node(
                dx_id,
                node_type=NODE_TYPES["diagnosis"],
                patient_id=int(patient_id),
                dx_id=dx_id,
                encounter_id=int(encounter.hadm_id),
                date=iso_date(encounter.admittime),
                icd_code=str(dx_row.icd9_code),
                description=str(description),
                status="resolved" if encounter.hospital_expire_flag else "active",
            )
            graph.add_edge(enc_id, dx_id, edge_type="DIAGNOSED_WITH")

        procedure_nodes = []
        for proc_row in encounter_proc.itertuples(index=False):
            proc_id = f"proc:{patient_id}:{encounter.hadm_id}:{proc_row.seq_num}"
            proc_time = _procedure_date(pd.Series(encounter._asdict()), int(proc_row.seq_num))
            description = proc_row.long_title or proc_row.short_title or proc_row.icd9_code
            outcome_summary = "Procedure documented during the indexed encounter."
            if "drain" in canonicalize_text(str(description)):
                outcome_summary = "Procedure likely relieved a fluid collection or decompressed a compartment."
            if "insert" in canonicalize_text(str(description)) or "catheter" in canonicalize_text(str(description)):
                outcome_summary = "Procedure established access or respiratory support during acute care."
            graph.add_node(
                proc_id,
                node_type=NODE_TYPES["procedure"],
                patient_id=int(patient_id),
                proc_id=proc_id,
                encounter_id=int(encounter.hadm_id),
                date=iso_date(proc_time),
                cpt_code=str(proc_row.icd9_code),
                description=str(description),
                outcome_summary=outcome_summary,
            )
            graph.add_edge(enc_id, proc_id, edge_type="PERFORMED_DURING")
            procedure_nodes.append(proc_id)

        for lab_row in encounter_labs.itertuples(index=False):
            row_id = int(lab_row.row_id)
            lab_id = f"lab:{patient_id}:{row_id}"
            graph.add_node(
                lab_id,
                node_type=NODE_TYPES["lab"],
                patient_id=int(patient_id),
                lab_id=lab_id,
                encounter_id=int(encounter.hadm_id),
                date=iso_date(lab_row.charttime),
                test_name=str(lab_row.label),
                value=float(lab_row.valuenum),
                unit=str(lab_row.valueuom or ""),
                reference_range="",
                flag=str(lab_row.flag or "normal").lower(),
                summary=f"{lab_row.label} measured {lab_row.valuenum:g}{lab_row.valueuom or ''}.",
            )
            graph.add_edge(enc_id, lab_id, edge_type="ORDERED_DURING")

        imaging_payloads = _imaging_blueprints(pd.Series(encounter._asdict()), dx_titles, proc_titles)
        for index, payload in enumerate(imaging_payloads, start=1):
            imaging_id = f"img:{patient_id}:{encounter.hadm_id}:{index}"
            graph.add_node(
                imaging_id,
                node_type=NODE_TYPES["imaging"],
                patient_id=int(patient_id),
                report_id=imaging_id,
                encounter_id=int(encounter.hadm_id),
                date=iso_date(encounter.admittime + pd.Timedelta(hours=12 * index)),
                modality=payload["modality"],
                body_part=payload["body_part"],
                summary=payload["summary"],
                full_text=payload["full_text"],
            )
            graph.add_edge(enc_id, imaging_id, edge_type="ORDERED_DURING")
            for proc_id in procedure_nodes[:2]:
                graph.add_edge(imaging_id, proc_id, edge_type="RESULTED_IN")

    return graph


def get_available_patient_ids() -> list[int]:
    return select_patient_cohort()


def graph_stats() -> pd.DataFrame:
    rows = []
    for patient_id in get_available_patient_ids():
        graph = get_patient_graph(patient_id)
        node_counts = {}
        for _, data in graph.nodes(data=True):
            node_type = data["node_type"]
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        rows.append(
            {
                "patient_id": patient_id,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                **node_counts,
            }
        )
    return pd.DataFrame(rows).fillna(0).sort_values("patient_id")
