from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from graph.graph_utils import NODE_TYPES, canonicalize_text, iso_date


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTES_ROOT = PROJECT_ROOT.parent

TREATMENT_PATTERNS = {
    "GEMCITABINE + CARBOPLATIN": [r"gemcitabine", r"carboplatin"],
    "PACLITAXEL": [r"\bpaclitaxel\b", r"\btaxol\b"],
    "TRASTUZUMAB": [r"\btrastuzumab\b", r"\bherceptin\b"],
    "ANASTROZOLE": [r"\banastrozole\b", r"\barimidex\b"],
    "CAPECITABINE": [r"\bcapecitabine\b"],
    "CMF": [r"\bcmf\b"],
    "RADIATION THERAPY": [r"\bradiation\b", r"\brt\b"],
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

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bdenies\b",
    r"\bnegative for\b",
]
HISTORICAL_PATTERNS = [
    r"\bhistory of\b",
    r"\bprior\b",
    r"\bprevious\b",
    r"\bstatus post\b",
    r"\bs/p\b",
    r"\bpast medical history\b",
]
PLANNED_PATTERNS = [
    r"\bplan(?:ned)?\b",
    r"\bwill\b",
    r"\bscheduled\b",
    r"\bto start\b",
    r"\brecommend(?:ed|ation)?\b",
]
FOLLOWUP_PATTERNS = [
    r"\bfollow[- ]up\b",
    r"\bafter\b",
    r"\bfollowing\b",
    r"\bsubsequent\b",
    r"\bnext step\b",
]


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


def _section_label(title: str, text: str) -> str:
    merged = f"{title}\n{text[:400]}".lower()
    if "assessment" in merged:
        return "assessment"
    if "impression" in merged:
        return "impression"
    if "plan" in merged:
        return "plan"
    if "history" in merged:
        return "history"
    return "general"


def _context_window(text: str, start: int, end: int, radius: int = 80) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)]


def _classify_context(window: str) -> tuple[str, str, float]:
    lower = window.lower()
    if any(re.search(p, lower) for p in NEGATION_PATTERNS):
        return "negated", "current", 0.05
    if any(re.search(p, lower) for p in HISTORICAL_PATTERNS):
        return "asserted", "historical", 0.35
    if any(re.search(p, lower) for p in PLANNED_PATTERNS):
        return "asserted", "planned", 0.55
    return "asserted", "current", 0.9


def _make_entity(label: str, entity_type: str, window: str, section: str) -> dict[str, Any]:
    assertion, temporality, confidence = _classify_context(window)
    if section in {"assessment", "impression"} and assertion == "asserted" and temporality == "current":
        confidence = min(1.0, confidence + 0.05)
    if section == "history" and temporality == "current":
        temporality = "historical"
        confidence = min(confidence, 0.45)
    return {
        "label": label,
        "entity_type": entity_type,
        "assertion": assertion,
        "temporality": temporality,
        "confidence": round(confidence, 2),
        "source_span": _clean_text(window)[:180],
        "section": section,
    }


def _extract_entities(title: str, text: str) -> dict[str, list[dict[str, Any]]]:
    merged = f"{title}\n{text}"
    lowered = merged.lower()
    section = _section_label(title, text)
    entities: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str, str]] = set()

    def register(label: str, entity_type: str, start: int, end: int) -> None:
        window = _context_window(merged, start, end)
        entity = _make_entity(label, entity_type, window, section)
        key = (entity_type, label, entity["assertion"], entity["temporality"])
        if key in seen:
            return
        seen.add(key)
        entities[entity_type].append(entity)

    for label, regexes in TREATMENT_PATTERNS.items():
        if label == "GEMCITABINE + CARBOPLATIN":
            matches = [re.search(regex, lowered) for regex in regexes]
            if all(matches):
                start = min(match.start() for match in matches if match)
                end = max(match.end() for match in matches if match)
                register(label, "treatments", start, end)
            continue
        for regex in regexes:
            match = re.search(regex, lowered)
            if match:
                register(label, "treatments", match.start(), match.end())
                break

    for label, regexes in DIAGNOSIS_PATTERNS.items():
        for regex in regexes:
            match = re.search(regex, lowered)
            if match:
                register(label, "diagnoses", match.start(), match.end())
                break

    for label, regexes in IMAGING_PATTERNS.items():
        for regex in regexes:
            match = re.search(regex, lowered)
            if match:
                register(label, "imaging", match.start(), match.end())
                break

    return {key: value for key, value in entities.items()}


def _note_summary(title: str, text: str, entities: dict[str, list[dict[str, Any]]]) -> str:
    excerpt = _clean_text(text)[:220]
    labels = [item["label"] for group in entities.values() for item in group if item["assertion"] == "asserted"]
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


def _should_materialize(entity: dict[str, Any]) -> bool:
    return entity["assertion"] == "asserted" and entity["temporality"] == "current" and entity["confidence"] >= 0.6


def _note_has_text_match(graph: nx.DiGraph, note_id: str, *terms: str) -> bool:
    if note_id not in graph:
        return False
    text = f"{graph.nodes[note_id].get('note_type', '')} {graph.nodes[note_id].get('full_text', '')}".lower()
    return all(term.lower() in text for term in terms if term)


def _supporting_note_ids(graph: nx.DiGraph, encounter_id: str) -> list[str]:
    return [
        target
        for _, target, edge in graph.out_edges(encounter_id, data=True)
        if edge.get("edge_type") == "DOCUMENTED_IN" and graph.nodes[target].get("node_type") == NODE_TYPES["note"]
    ]


@lru_cache(maxsize=16)
def get_patient_graph(patient_id: int) -> nx.DiGraph:
    notes = load_patient_notes(patient_id)
    graph = nx.DiGraph(patient_id=patient_id, source="free_text_breast_notes")

    encounter_map: dict[tuple[str, str, str], str] = {}
    diagnosis_counter = 0
    procedure_counter = 0
    imaging_counter = 0
    ordered_encounters: list[tuple[str, pd.Timestamp | None]] = []
    imaging_nodes: list[tuple[str, str, str]] = []
    procedure_nodes: list[tuple[str, str, str, float]] = []

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
            section_label=_section_label(title, text),
        )
        graph.add_edge(enc_id, note_id, edge_type="DOCUMENTED_IN")

        for entity in entities.get("diagnoses", []):
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
                description=entity["label"],
                status="active" if entity["temporality"] == "current" else entity["temporality"],
                summary=f"Diagnosis mention: {entity['label']}",
                assertion=entity["assertion"],
                temporality=entity["temporality"],
                confidence=entity["confidence"],
                source_span=entity["source_span"],
                section_label=entity["section"],
            )
            graph.add_edge(enc_id, dx_id, edge_type="DIAGNOSED_WITH", confidence=entity["confidence"])
            graph.add_edge(note_id, dx_id, edge_type="MENTIONED_IN", confidence=entity["confidence"])

        for entity in entities.get("treatments", []):
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
                description=entity["label"],
                outcome_summary="Mentioned in free-text note.",
                summary=f"Treatment or procedure mention: {entity['label']}",
                assertion=entity["assertion"],
                temporality=entity["temporality"],
                confidence=entity["confidence"],
                source_span=entity["source_span"],
                section_label=entity["section"],
            )
            graph.add_edge(enc_id, proc_id, edge_type="PERFORMED_DURING", confidence=entity["confidence"])
            graph.add_edge(note_id, proc_id, edge_type="MENTIONED_IN", confidence=entity["confidence"])
            if _should_materialize(entity):
                procedure_nodes.append((proc_id, note_date, note_id, entity["confidence"]))

        for entity in entities.get("imaging", []):
            imaging_counter += 1
            imaging_id = f"img:{patient_id}:{imaging_counter:04d}"
            graph.add_node(
                imaging_id,
                node_type=NODE_TYPES["imaging"],
                patient_id=patient_id,
                report_id=imaging_id,
                encounter_id=enc_id,
                date=note_date,
                modality=entity["label"],
                body_part="breast/chest",
                summary=f"{entity['label']} referenced in note: {title or 'untitled note'}",
                full_text=text,
                assertion=entity["assertion"],
                temporality=entity["temporality"],
                confidence=entity["confidence"],
                source_span=entity["source_span"],
                section_label=entity["section"],
            )
            graph.add_edge(enc_id, imaging_id, edge_type="ORDERED_DURING", confidence=entity["confidence"])
            graph.add_edge(note_id, imaging_id, edge_type="MENTIONED_IN", confidence=entity["confidence"])
            if _should_materialize(entity):
                imaging_nodes.append((imaging_id, note_date, note_id))

    ordered_encounters = sorted(ordered_encounters, key=lambda item: item[1] or pd.Timestamp.min)
    for first, second in zip(ordered_encounters, ordered_encounters[1:]):
        graph.add_edge(first[0], second[0], edge_type="FOLLOWED_BY")

    def _to_ts(value: str) -> pd.Timestamp:
        return pd.to_datetime(value, errors="coerce")

    procedure_nodes.sort(key=lambda item: _to_ts(item[1]))
    for imaging_id, imaging_date, source_note_id in sorted(imaging_nodes, key=lambda item: _to_ts(item[1])):
        img_ts = _to_ts(imaging_date)
        modality = str(graph.nodes[imaging_id].get("modality", ""))
        encounter_id = str(graph.nodes[imaging_id].get("encounter_id", ""))
        image_notes = _supporting_note_ids(graph, encounter_id)
        best_candidate: tuple[str, float, list[str], str] | None = None
        for proc_id, proc_date, proc_note_id, proc_conf in procedure_nodes:
            proc_ts = _to_ts(proc_date)
            if pd.isna(img_ts) or pd.isna(proc_ts):
                continue
            day_delta = int((proc_ts - img_ts).days)
            if not 0 <= day_delta <= 120:
                continue
            proc_label = str(graph.nodes[proc_id].get("description", ""))
            proc_enc_id = str(graph.nodes[proc_id].get("encounter_id", ""))
            proc_notes = _supporting_note_ids(graph, proc_enc_id)
            confidence = 0.15
            evidence_note_ids: list[str] = []
            edge_type = "FOLLOWED_BY_EVENT"

            for note_id in proc_notes:
                text = f"{graph.nodes[note_id].get('note_type', '')} {graph.nodes[note_id].get('full_text', '')}".lower()
                if proc_label.lower() in text:
                    confidence += 0.2
                    evidence_note_ids.append(note_id)
                if modality.lower() in text:
                    confidence += 0.2
                    evidence_note_ids.append(note_id)
                if any(re.search(pattern, text) for pattern in FOLLOWUP_PATTERNS):
                    confidence += 0.15
                    evidence_note_ids.append(note_id)

            for note_id in image_notes:
                text = f"{graph.nodes[note_id].get('note_type', '')} {graph.nodes[note_id].get('full_text', '')}".lower()
                if proc_label.lower() in text and any(re.search(pattern, text) for pattern in PLANNED_PATTERNS):
                    confidence += 0.2
                    evidence_note_ids.append(note_id)

            confidence += min(proc_conf, 0.2)
            if proc_enc_id == encounter_id:
                confidence += 0.15
            if day_delta <= 30:
                confidence += 0.1
            if day_delta > 90:
                confidence -= 0.15

            evidence_note_ids = list(dict.fromkeys(evidence_note_ids))
            if confidence >= 0.75:
                edge_type = "RESULTED_IN"
            elif confidence >= 0.55:
                edge_type = "RECOMMENDED_AFTER"
            else:
                continue

            candidate = (proc_id, round(confidence, 2), evidence_note_ids, edge_type)
            if best_candidate is None or candidate[1] > best_candidate[1]:
                best_candidate = candidate

        if best_candidate is not None:
            proc_id, confidence, evidence_note_ids, edge_type = best_candidate
            graph.add_edge(
                imaging_id,
                proc_id,
                edge_type=edge_type,
                confidence=confidence,
                evidence_note_ids=evidence_note_ids,
                evidence_source=source_note_id,
            )

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
