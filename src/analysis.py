"""Exploratory summaries, transparent risk scoring, and insight generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _minmax(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / span


def site_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return site KPIs and a documented 0-100 comparative risk score."""
    site = (
        df.groupby(["site", "country"], as_index=False)
        .agg(
            incidents=("incident_id", "count"),
            critical_incidents=("severity", lambda s: int((s == "Critical").sum())),
            high_incidents=("severity", lambda s: int((s == "High").sum())),
            high_risk_rate=("high_risk_target", "mean"),
            confirmed_rate=("is_confirmed", "mean"),
            unresolved_rate=("is_unresolved", "mean"),
            avg_response_min=("response_time_min", "mean"),
            p90_response_min=("response_time_min", lambda s: s.quantile(0.90)),
            slow_response_rate=("slow_response", "mean"),
        )
    )

    max_date = df["date"].max()
    recent_start = max_date - pd.Timedelta(days=89)
    prior_start = recent_start - pd.Timedelta(days=90)
    recent = df[df["date"].between(recent_start, max_date)].groupby("site").size()
    prior = df[df["date"].between(prior_start, recent_start - pd.Timedelta(days=1))].groupby("site").size()
    site["recent_90d_incidents"] = site["site"].map(recent).fillna(0).astype(int)
    site["prior_90d_incidents"] = site["site"].map(prior).fillna(0).astype(int)
    site["recent_trend_pct"] = np.where(
        site["prior_90d_incidents"].gt(0),
        100 * (site["recent_90d_incidents"] - site["prior_90d_incidents"]) / site["prior_90d_incidents"],
        0,
    )

    # Weighted comparative index, not a probability or causal safety measure.
    components = {
        "severity_component": 0.35 * _minmax(site["high_risk_rate"]),
        "critical_component": 0.20 * _minmax(site["critical_incidents"]),
        "response_component": 0.20 * _minmax(site["p90_response_min"]),
        "unresolved_component": 0.15 * _minmax(site["unresolved_rate"]),
        "trend_component": 0.10 * _minmax(site["recent_trend_pct"].clip(lower=0)),
    }
    for name, values in components.items():
        site[name] = values
    site["risk_score"] = 100 * sum(components.values())
    site["risk_rank"] = site["risk_score"].rank(method="min", ascending=False).astype(int)

    percent_cols = ["high_risk_rate", "confirmed_rate", "unresolved_rate", "slow_response_rate"]
    site[percent_cols] = site[percent_cols] * 100
    numeric_cols = site.select_dtypes(include="number").columns
    site[numeric_cols] = site[numeric_cols].round(2)
    return site.sort_values("risk_rank").reset_index(drop=True)


def build_summaries(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    monthly = (
        df.groupby("year_month", as_index=False)
        .agg(
            incidents=("incident_id", "count"),
            high_risk=("high_risk_target", "sum"),
            critical=("severity", lambda s: int((s == "Critical").sum())),
            avg_response_min=("response_time_min", "mean"),
        )
    )
    monthly["high_risk_rate"] = 100 * monthly["high_risk"] / monthly["incidents"]
    monthly[["avg_response_min", "high_risk_rate"]] = monthly[["avg_response_min", "high_risk_rate"]].round(2)

    incident_type = (
        df.groupby("incident_type", as_index=False)
        .agg(
            incidents=("incident_id", "count"),
            high_risk_rate=("high_risk_target", "mean"),
            critical=("severity", lambda s: int((s == "Critical").sum())),
            avg_response_min=("response_time_min", "mean"),
            confirmed_rate=("is_confirmed", "mean"),
        )
        .sort_values("incidents", ascending=False)
    )
    for col in ["high_risk_rate", "confirmed_rate"]:
        incident_type[col] = (100 * incident_type[col]).round(2)
    incident_type["avg_response_min"] = incident_type["avg_response_min"].round(2)

    shift = (
        df.groupby("shift", as_index=False)
        .agg(
            incidents=("incident_id", "count"),
            high_risk_rate=("high_risk_target", "mean"),
            critical=("severity", lambda s: int((s == "Critical").sum())),
            avg_response_min=("response_time_min", "mean"),
            p90_response_min=("response_time_min", lambda s: s.quantile(0.90)),
        )
    )
    shift["high_risk_rate"] = (100 * shift["high_risk_rate"]).round(2)
    shift[["avg_response_min", "p90_response_min"]] = shift[["avg_response_min", "p90_response_min"]].round(2)

    severity_response = (
        df.groupby("severity", as_index=False)
        .agg(
            incidents=("incident_id", "count"),
            avg_response_min=("response_time_min", "mean"),
            median_response_min=("response_time_min", "median"),
            p90_response_min=("response_time_min", lambda s: s.quantile(0.90)),
        )
    )
    severity_response = severity_response.round(2)

    return {
        "site_risk_summary": site_risk_summary(df),
        "monthly_summary": monthly,
        "incident_type_summary": incident_type,
        "shift_summary": shift,
        "severity_response_summary": severity_response,
    }


def generate_insights(df: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> dict:
    site = summaries["site_risk_summary"]
    incident = summaries["incident_type_summary"]
    shift = summaries["shift_summary"]
    monthly = summaries["monthly_summary"]
    top_risk = site.iloc[0]
    top_volume = site.sort_values("incidents", ascending=False).iloc[0]
    top_critical = site.sort_values("critical_incidents", ascending=False).iloc[0]
    slow_site = site.sort_values("avg_response_min", ascending=False).iloc[0]
    top_type = incident.iloc[0]
    type_risk = incident.sort_values(["high_risk_rate", "incidents"], ascending=False).iloc[0]
    slow_shift = shift.sort_values("avg_response_min", ascending=False).iloc[0]
    peak_month = monthly.sort_values("incidents", ascending=False).iloc[0]

    return {
        "overview": {
            "records": int(len(df)),
            "sites": int(df["site"].nunique()),
            "countries": int(df["country"].nunique()),
            "date_min": df["date"].min().strftime("%Y-%m-%d"),
            "date_max": df["date"].max().strftime("%Y-%m-%d"),
            "high_critical_count": int(df["high_risk_target"].sum()),
            "high_critical_rate": round(100 * df["high_risk_target"].mean(), 2),
            "critical_count": int(df["severity"].eq("Critical").sum()),
            "avg_response_min": round(float(df["response_time_min"].mean()), 2),
            "median_response_min": round(float(df["response_time_min"].median()), 2),
            "p90_response_min": round(float(df["response_time_min"].quantile(0.90)), 2),
            "confirmed_rate": round(100 * float(df["is_confirmed"].mean()), 2),
        },
        "findings": [
            f"{top_volume['site']} has the highest raw incident volume ({int(top_volume['incidents'])}), but volume is not a population-adjusted incident rate.",
            f"{top_critical['site']} records the most critical incidents ({int(top_critical['critical_incidents'])}).",
            f"{top_risk['site']} ranks first on the transparent comparative risk index ({top_risk['risk_score']:.1f}/100).",
            f"{slow_site['site']} has the slowest average response ({slow_site['avg_response_min']:.2f} minutes).",
            f"{top_type['incident_type']} is the most frequent incident type ({int(top_type['incidents'])} records).",
            f"{type_risk['incident_type']} has the highest observed high/critical share ({type_risk['high_risk_rate']:.1f}%).",
            f"{slow_shift['shift']} shift has the slowest average response ({slow_shift['avg_response_min']:.2f} minutes).",
            f"{peak_month['year_month']} has the highest monthly incident count ({int(peak_month['incidents'])}).",
        ],
        "limitations": [
            "No site population, employee count, visitor volume, floor area, or operating-hour exposure is supplied; raw site counts are not true incident rates.",
            "No precise event timestamp, dispatch timestamp, resolution duration, loss amount, narrative, or guard staffing data is available.",
            "The unusually complete and balanced sample may be curated or synthetic; small differences should not be treated as causal evidence.",
            "The risk index is a transparent prioritization aid, not a probability of harm or a replacement for security judgment.",
        ],
    }


def save_analysis_outputs(df: pd.DataFrame, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = build_summaries(df)
    for name, frame in summaries.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    insights = generate_insights(df, summaries)
    (output_dir / "insights.json").write_text(json.dumps(insights, indent=2), encoding="utf-8")
    return insights

