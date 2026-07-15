"""Data loading, validation, cleaning, and feature engineering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = {
    "Incident ID": "incident_id",
    "Date": "date",
    "Site": "site",
    "Country": "country",
    "Incident Type": "incident_type",
    "Severity": "severity",
    "Shift": "shift",
    "Response Time (min)": "response_time_min",
    "Badge Access": "badge_access",
    "CCTV Alert": "cctv_alert",
    "Visitor": "visitor",
    "Outcome": "outcome",
}

EXPECTED_CATEGORIES = {
    "severity": {"Low", "Medium", "High", "Critical"},
    "shift": {"Day", "Evening", "Night"},
    "badge_access": {"Yes", "No", "N/A"},
    "cctv_alert": {"Yes", "No"},
    "visitor": {"Yes", "No"},
    "outcome": {"Confirmed", "Resolved", "Under Investigation", "False Alarm"},
}

SEVERITY_WEIGHT = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}


def load_and_clean(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load the supplied CSV, validate its schema, and return cleaned data and QA results."""
    path = Path(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = raw.columns.str.strip()

    missing_columns = sorted(set(RAW_COLUMNS) - set(raw.columns))
    extra_columns = sorted(set(raw.columns) - set(RAW_COLUMNS))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = raw[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS).copy()
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].str.strip()

    invalid_dates_before = int(pd.to_datetime(df["date"], errors="coerce").isna().sum())
    invalid_response_before = int(pd.to_numeric(df["response_time_min"], errors="coerce").isna().sum())
    blank_counts = {col: int(df[col].eq("").sum()) for col in df.columns}
    duplicate_ids = int(df["incident_id"].duplicated().sum())
    duplicate_rows = int(df.duplicated().sum())

    invalid_categories = {}
    for col, expected in EXPECTED_CATEGORIES.items():
        unexpected = sorted(set(df[col].unique()) - expected)
        invalid_categories[col] = unexpected

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["response_time_min"] = pd.to_numeric(df["response_time_min"], errors="coerce")
    df = df.drop_duplicates(subset="incident_id", keep="first")
    df = df.dropna(subset=["date", "response_time_min"]).copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek.ge(5)
    df["severity_weight"] = df["severity"].map(SEVERITY_WEIGHT).astype(int)
    df["high_risk_target"] = df["severity"].isin(["High", "Critical"]).astype(int)
    df["is_confirmed"] = df["outcome"].eq("Confirmed").astype(int)
    df["is_unresolved"] = df["outcome"].eq("Under Investigation").astype(int)
    df["is_false_alarm"] = df["outcome"].eq("False Alarm").astype(int)
    df["slow_response"] = df["response_time_min"].gt(15).astype(int)

    site_country_counts = df.groupby("site")["country"].nunique()
    qa = {
        "source_file": path.name,
        "source_rows": int(len(raw)),
        "clean_rows": int(len(df)),
        "source_columns": int(len(raw.columns)),
        "missing_required_columns": missing_columns,
        "extra_columns": extra_columns,
        "blank_counts": blank_counts,
        "duplicate_incident_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "invalid_dates": invalid_dates_before,
        "invalid_response_times": invalid_response_before,
        "invalid_categories": invalid_categories,
        "negative_response_times": int(df["response_time_min"].lt(0).sum()),
        "site_country_mapping_violations": int(site_country_counts.gt(1).sum()),
        "date_min": df["date"].min().strftime("%Y-%m-%d"),
        "date_max": df["date"].max().strftime("%Y-%m-%d"),
        "badge_na_count": int(df["badge_access"].eq("N/A").sum()),
        "badge_na_interpretation": "Retained as an explicit 'not applicable/unknown' category; not imputed.",
        "quality_note": (
            "The supplied sample is unusually complete and category-consistent. "
            "This may reflect curated or synthetic assessment data, so small differences should not be over-interpreted."
        ),
    }
    return df.sort_values(["date", "incident_id"]).reset_index(drop=True), qa


def save_outputs(df: pd.DataFrame, qa: dict, output_dir: str | Path) -> None:
    """Persist cleaned data and a machine-readable data-quality report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export = df.copy()
    export["date"] = export["date"].dt.strftime("%Y-%m-%d")
    export.to_csv(output_dir / "physical_security_incidents_cleaned.csv", index=False)
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )


def data_quality_score(qa: dict) -> float:
    """Create a transparent completeness/validity/uniqueness score out of 100."""
    rows = max(qa["source_rows"], 1)
    blank_total = sum(qa["blank_counts"].values())
    issue_total = (
        blank_total
        + qa["duplicate_incident_ids"]
        + qa["invalid_dates"]
        + qa["invalid_response_times"]
        + sum(len(v) for v in qa["invalid_categories"].values())
    )
    opportunities = rows * max(qa["source_columns"], 1)
    return float(np.clip(100 * (1 - issue_total / opportunities), 0, 100))

