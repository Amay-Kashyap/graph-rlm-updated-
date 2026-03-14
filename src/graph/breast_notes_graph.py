from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import networkx as nx
import pandas as pd

from graph.graph_utils import NODE_TYPES, iso_date


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTES_ROOT = PROJECT_ROOT.parent

TREATMENT_PATTERNS = {
    "GEMCITABINE + CARBOPLATIN": [r"gemcitabine", r"carboplatin"],
    "PACLITAXEL": [r"\bpaclitaxel\b", r"\btaxol\b"],
    "TRASTUZUMAB": [r"\btrastuzumab\b", r"\bherceptin\b"],
    "ANASTROZOLE": [r"\banastrozole\b", r"\barimidex\b"],
    "CAPECITABINE": [r"\bcapecitabine\b"],
    "CMF": [r"\bcmf\b"],
    "RADIATION THERAPY": [r"radiation"],
    "LUMPECTOMY": [r"\blumpectomy\b"],
    "MASTECTOMY": [r"\bmastectomy\b"],
    "BIOPSY": [r"\bbiopsy\b"],
}

DIAGNOSIS_PATTERNS = {
    "STAGE IA": [r"\bstage ia\b"],
    "STAGE IIB": [r"\bstage iib\b"],
    "STAGE IV": [r"\bstage iv\b"],
    "DCIS": [r"\bdcis\b"],
    "INVASIVE DUCTAL CARCINOMA": [r"invasive ductal carcinoma"],
    "LOBULAR CARCINOMA": [r"lobular carcinoma"],
    "TRIPLE NEGATIVE": [r"triple negative", r"er/pr/her2[- ]negative"],
    "BREAST CANCER": [r"\bbreast cancer\b", r"\bbreast ca\b", r"malignant neoplasm.*breast"],
    "METASTATIC DISEASE": [r"\bmetastatic\b", r"\bm1\b", r"malignant involvement"],
}

IMAGING_PATTERNS = {
    "PET/CT": [r"\bpet/?ct\b", r"\bpet scan\b"],
    "CT": [r"\bct chest abdomen pelvis\b", r"\bct c/a/p\b", r"\bct chest\b", r"\bct abdomen\b"],
    "MRI": [r"\bmri\b"],
    "MAMMOGRAM": [r"\bmammogram\b"],
    "ULTRASOUND": [r"\bultrasound\b"],
    "DEXA": [r"\bdexa\b"],
    "ECHO": [r"\bechocardi", r"transthoracic echo"],
}


def discover_note_files() -> list[Path]:
    return sorted(NOTES_ROOT.glob("patient_*.csv"))


def get_available_patient_ids() -> list[int]:
    return [int(path.stem.split("_")[1]) for path in discover_note_files()]


@lru_cache(maxsize=16)
def load_patient_notes(patient_id: int) -> pd.DataFrame:
    path = NOTES_ROOT / f"patient_{patient_id}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for column in ["NOTE_DATETIME", "VISIT_OCCURRENCE.DATETIME"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    df["NOTE_TEXT"] = df["NOTE_TEXT"].fillna("")
    df["NOTE_TITLE"] = df["NOTE_TITLE"].fillna("")
    return df.sort_values(["VISIT_OCCURRENCE.DATETIME", "NOTE_DATETIME", "NOTE_ID"]).reset_index(drop=True)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _extract_entities(title: str, text: str) -> dict[str, list[str]]:
    merged = f"{title} {text}".lower()
    treatments = []
    for label, regexes in TREATMENT_PATTERNS.items():
        if label == "GEMCITABINE + CARBOPLATIN":
            if all(re.search(regex, merged) for regex in regexes):
                treatments.append(label)
        elif any(re.search(regex, merged) for regex in regexes):
            treatments.append(label)
    diagnoses = [
        label
        for label, regexes in DIAGNOSIS_PATTERNS.items()
        if any(re.search(regex, merged) for regex in regexes)
    ]
    imaging = [
        label
        for label, regexes in IMAGING_PATTERNS.items()
        if any(re.search(regex, merged) for regex in regexes)
    ]
    return {"treatments": treatments, "diagnoses": diagnoses, "imaging": imaging}


def _note_summary(title: str, text: str, entities: dict[str, list[str]]) -> str:
    excerpt = _clean_text(text)[:220]
    labels = entities["diagnoses"] + entities["treatments"] + entities["imaging"]
    prefix = ", ".join(labels[:4])
    if prefix:
        return f"{title}: {prefix}. {excerpt}".strip()
    return f"{title}: {excerpt}".strip()


def _encounter_key(row: pd.Series) -> tuple[str, str, str]:
    visit_dt = row.get("VISIT_OCCURRENCE.DATETIME")
    note_dt = row.get("NOTE_DATETIME")
    event_time = visit_dt if pd.notna(visit_dt) else note_dt
    event_date = iso_date(event_time)
    visit_type = str(row.get("VISIT_OCCURRENCE.TYPE", "") or "")
    dept = str(row.get("DEPARTMENT.NAME", "") or "")
    return event_date, visit_type, dept


@lru_cache(maxsize=16)
def get_patient_graph(patient_id: int) -> nx.DiGraph:
    notes = load_patient_notes(patient_id)
    graph = nx.DiGraph(patient_id=patient_id, source="free_text_breast_notes")

    encounter_map: dict[tuple[str, str, str], str] = {}
    diagnosis_counter = 0
    procedure_counter = 0
    imaging_counter = 0
    ordered_encounters: list[tuple[str, pd.Timestamp | None]] = []
    imaging_nodes: list[tuple[str, str]] = []
    procedure_nodes: list[tuple[str, str]] = []

    for row in notes.itertuples(index=False):
        row_dict = row._asdict()
        key = _encounter_key(pd.Series(row_dict))
        if key not in encounter_map:
            enc_id = f"enc:{patient_id}:{len(encounter_map)+1:03d}"
            date, visit_type, dept = key
            graph.add_node(
                enc_id,
                node_type=NODE_TYPES["encounter"],
                patient_id=patient_id,
                encounter_id=enc_id,
                date=date,
                type=visit_type.lower() if visit_type else "encounter",
                summary=f"{visit_type or 'Encounter'} in {dept or 'unspecified department'}.",
                department=dept,
            )
            encounter_map[key] = enc_id
            visit_dt = row_dict.get("VISIT_OCCURRENCE.DATETIME") or row_dict.get("NOTE_DATETIME")
            ordered_encounters.append((enc_id, visit_dt if pd.notna(visit_dt) else None))
        enc_id = encounter_map[key]

        title = _clean_text(row_dict.get("NOTE_TITLE", ""))
        text = _clean_text(row_dict.get("NOTE_TEXT", ""))
        entities = _extract_entities(title, text)
        note_id = f"note:{patient_id}:{row_dict['NOTE_ID']}"
        note_date = iso_date(row_dict.get("NOTE_DATETIME") or row_dict.get("VISIT_OCCURRENCE.DATETIME"))
        graph.add_node(
            note_id,
            node_type=NODE_TYPES["note"],
            patient_id=patient_id,
            note_id=note_id,
            encounter_id=enc_id,
            date=note_date,
            note_type=title or "Note",
            author=str(row_dict.get("PROVIDER.TYPE", "") or ""),
            summary=_note_summary(title, text, entities),
            full_text=text,
            department=str(row_dict.get("DEPARTMENT.NAME", "") or ""),
            source_value=str(row_dict.get("NOTE_SOURCE_VALUE", "") or ""),
        )
        graph.add_edge(enc_id, note_id, edge_type="DOCUMENTED_IN")

        for diagnosis_label in entities["diagnoses"]:
            diagnosis_counter += 1
            dx_id = f"dx:{patient_id}:{diagnosis_counter:04d}"
            graph.add_node(
                dx_id,
                node_type=NODE_TYPES["diagnosis"],
                patient_id=patient_id,
                dx_id=dx_id,
                encounter_id=enc_id,
                date=note_date,
                icd_code="",
                description=diagnosis_label,
                status="active",
                summary=f"Diagnosis mention: {diagnosis_label}",
            )
            graph.add_edge(enc_id, dx_id, edge_type="DIAGNOSED_WITH")
            graph.add_edge(note_id, dx_id, edge_type="DIAGNOSED_WITH")

        for treatment_label in entities["treatments"]:
            procedure_counter += 1
            proc_id = f"proc:{patient_id}:{procedure_counter:04d}"
            graph.add_node(
                proc_id,
                node_type=NODE_TYPES["procedure"],
                patient_id=patient_id,
                proc_id=proc_id,
                encounter_id=enc_id,
                date=note_date,
                cpt_code="",
                description=treatment_label,
                outcome_summary="Mentioned in free-text note.",
                summary=f"Treatment or procedure mention: {treatment_label}",
            )
            graph.add_edge(enc_id, proc_id, edge_type="PERFORMED_DURING")
            graph.add_edge(note_id, proc_id, edge_type="PERFORMED_DURING")
            procedure_nodes.append((proc_id, note_date))

        for imaging_label in entities["imaging"]:
            imaging_counter += 1
            imaging_id = f"img:{patient_id}:{imaging_counter:04d}"
            graph.add_node(
                imaging_id,
                node_type=NODE_TYPES["imaging"],
                patient_id=patient_id,
                report_id=imaging_id,
                encounter_id=enc_id,
                date=note_date,
                modality=imaging_label,
                body_part="breast/chest",
                summary=f"{imaging_label} referenced in note: {title or 'untitled note'}",
                full_text=text,
            )
            graph.add_edge(enc_id, imaging_id, edge_type="ORDERED_DURING")
            graph.add_edge(note_id, imaging_id, edge_type="DOCUMENTED_IN")
            imaging_nodes.append((imaging_id, note_date))

    ordered_encounters = sorted(ordered_encounters, key=lambda item: item[1] or pd.Timestamp.min)
    for first, second in zip(ordered_encounters, ordered_encounters[1:]):
        graph.add_edge(first[0], second[0], edge_type="FOLLOWED_BY")

    def _to_ts(value: str) -> pd.Timestamp:
        return pd.to_datetime(value, errors="coerce")

    procedure_nodes.sort(key=lambda item: _to_ts(item[1]))
    for imaging_id, imaging_date in sorted(imaging_nodes, key=lambda item: _to_ts(item[1])):
        img_ts = _to_ts(imaging_date)
        for proc_id, proc_date in procedure_nodes:
            proc_ts = _to_ts(proc_date)
            if pd.isna(img_ts) or pd.isna(proc_ts):
                continue
            day_delta = int((proc_ts - img_ts).days)
            if 0 <= day_delta <= 180:
                graph.add_edge(imaging_id, proc_id, edge_type="RESULTED_IN")
                break

    return graph


def graph_stats() -> pd.DataFrame:
    rows = []
    for patient_id in get_available_patient_ids():
        graph = get_patient_graph(patient_id)
        counts: dict[str, int] = {}
        for _, data in graph.nodes(data=True):
            node_type = data["node_type"]
            counts[node_type] = counts.get(node_type, 0) + 1
        rows.append(
            {
                "patient_id": patient_id,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                **counts,
            }
        )
    return pd.DataFrame(rows).fillna(0).sort_values("patient_id")
