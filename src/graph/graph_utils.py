from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


NODE_TYPES = {
    "encounter": "Encounter",
    "imaging": "Imaging_Report",
    "note": "Clinical_Note",
    "lab": "Lab_Result",
    "diagnosis": "Diagnosis",
    "procedure": "Procedure",
}


def parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts


def iso_date(value: Any) -> str:
    ts = parse_timestamp(value)
    if ts is None:
        return ""
    return ts.strftime("%Y-%m-%d")


def approx_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.25))


def canonicalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    parts = []
    last_space = False
    for ch in text:
        keep = ch.isalnum() or ch in {" ", "%", ".", "/"}
        if not keep:
            ch = " "
        if ch == " ":
            if last_space:
                continue
            last_space = True
        else:
            last_space = False
        parts.append(ch)
    return "".join(parts).strip()


def summarize_node(data: dict[str, Any]) -> str:
    node_type = data.get("node_type", "")
    if node_type == NODE_TYPES["encounter"]:
        return (
            f"{data.get('date', '')} {data.get('type', '')}: "
            f"{data.get('summary', '')}"
        ).strip()
    if node_type == NODE_TYPES["lab"]:
        value = data.get("value", "")
        unit = data.get("unit", "")
        flag = data.get("flag", "")
        return (
            f"{data.get('date', '')} {data.get('test_name', '')} "
            f"{value}{unit} ({flag or 'unflagged'})"
        ).strip()
    if node_type == NODE_TYPES["diagnosis"]:
        return (
            f"{data.get('date', '')} {data.get('description', '')} "
            f"[{data.get('status', '')}]"
        ).strip()
    if node_type == NODE_TYPES["procedure"]:
        return (
            f"{data.get('date', '')} {data.get('description', '')}: "
            f"{data.get('outcome_summary', '')}"
        ).strip()
    return f"{data.get('date', '')} {data.get('summary', '')}".strip()


def trend_label(values: list[float]) -> str:
    if len(values) < 2:
        return "insufficient data"
    start = values[0]
    end = values[-1]
    spread = max(values) - min(values)
    if spread <= max(0.5, abs(start) * 0.05):
        return "stable"
    if end > start:
        return "increased"
    if end < start:
        return "decreased"
    return "fluctuated"


def date_distance_days(start: Any, end: Any) -> int | None:
    start_ts = parse_timestamp(start)
    end_ts = parse_timestamp(end)
    if start_ts is None or end_ts is None:
        return None
    return int((end_ts.normalize() - start_ts.normalize()).days)


def friendly_datetime(value: Any) -> str:
    ts = parse_timestamp(value)
    if ts is None:
        return ""
    return ts.strftime("%Y-%m-%d")


def now_stamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
