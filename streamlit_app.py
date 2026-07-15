"""Executive physical security analytics and AI triage dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from src.analysis import build_summaries
from src.data_preparation import data_quality_score, load_and_clean


ROOT = Path(__file__).resolve().parent
NAVY = "#10233F"
BLUE = "#2367A8"
GOLD = "#E0A82E"
RED = "#C94747"
GREEN = "#2B8A6E"
LIGHT = "#F4F7FB"
SEVERITY_COLORS = {"Low": GREEN, "Medium": GOLD, "High": "#E26A2C", "Critical": RED}

st.set_page_config(
    page_title="SentinelAI | Physical Security Intelligence",
    page_icon="ðŸ›¡ï¸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background: {LIGHT}; }}
    [data-testid="stSidebar"] {{ background: {NAVY}; }}
    [data-testid="stSidebar"] * {{ color: white; }}
    .hero {{ padding: 1.2rem 1.5rem; border-radius: 14px; background: linear-gradient(115deg, {NAVY}, {BLUE}); color: white; margin-bottom: 1rem; }}
    .hero h1 {{ margin: 0; font-size: 2rem; }}
    .hero p {{ margin: .35rem 0 0; opacity: .88; }}
    .callout {{ padding: .9rem 1rem; border-left: 4px solid {GOLD}; background: white; border-radius: 8px; margin: .5rem 0; }}
    .risk-high {{ padding: 1rem; border-radius: 12px; background: #FDECEC; border: 1px solid #F5B7B1; }}
    .risk-low {{ padding: 1rem; border-radius: 12px; background: #EAF7F2; border: 1px solid #A9DFBF; }}
    div[data-testid="stMetric"] {{ background: white; border: 1px solid #DFE7F1; padding: .8rem; border-radius: 12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, dict]:
    return load_and_clean(ROOT / "Physical_Security_Incidents.csv")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "incident_risk_model.joblib")


def pct(value: float) -> str:
    return f"{value:.1f}%"


def filtered_data(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("## Filters")
    dates = st.sidebar.date_input(
        "Incident date",
        value=(df["date"].min().date(), df["date"].max().date()),
        min_value=df["date"].min().date(),
        max_value=df["date"].max().date(),
    )
    sites = st.sidebar.multiselect("Site", sorted(df["site"].unique()))
    severities = st.sidebar.multiselect("Severity", ["Critical", "High", "Medium", "Low"])
    incident_types = st.sidebar.multiselect("Incident type", sorted(df["incident_type"].unique()))
    shifts = st.sidebar.multiselect("Shift", ["Day", "Evening", "Night"])
    outcomes = st.sidebar.multiselect("Outcome", sorted(df["outcome"].unique()))

    result = df.copy()
    if isinstance(dates, (tuple, list)) and len(dates) == 2:
        start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
        result = result[result["date"].between(start, end)]
    for column, values in {
        "site": sites,
        "severity": severities,
        "incident_type": incident_types,
        "shift": shifts,
        "outcome": outcomes,
    }.items():
        if values:
            result = result[result[column].isin(values)]
    return result


df, quality = load_data()
view = filtered_data(df)

st.sidebar.markdown("---")
st.sidebar.caption("Assessment dataset Â· 2,000 historical incidents")
st.sidebar.caption("Risk scores support prioritization; they do not replace security judgment.")

st.markdown(
    """
    <div class="hero">
      <h1>ðŸ›¡ï¸ SentinelAI Security Intelligence</h1>
      <p>Global incident trends, operational risk, response performance, and explainable AI triage</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if view.empty:
    st.warning("No incidents match the current filters. Adjust the sidebar selections.")
    st.stop()

overview_tab, site_tab, response_tab, ai_tab, governance_tab = st.tabs(
    ["Executive overview", "Site risk", "Response operations", "AI triage", "Data & governance"]
)

with overview_tab:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Incidents", f"{len(view):,}")
    c2.metric("High / critical", f"{int(view['high_risk_target'].sum()):,}", pct(100 * view["high_risk_target"].mean()))
    c3.metric("Critical", f"{int(view['severity'].eq('Critical').sum()):,}")
    c4.metric("Avg response", f"{view['response_time_min'].mean():.1f} min")
    c5.metric("Confirmed", pct(100 * view["is_confirmed"].mean()))

    left, right = st.columns([1.6, 1])
    with left:
        st.subheader("Incident volume and high-risk trend")
        monthly = (
            view.groupby("year_month", as_index=False)
            .agg(incidents=("incident_id", "count"), high_risk=("high_risk_target", "sum"))
        )
        base = alt.Chart(monthly).encode(x=alt.X("year_month:N", title="Month", sort=None))
        bars = base.mark_bar(color=BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            y=alt.Y("incidents:Q", title="Incident count"), tooltip=["year_month", "incidents", "high_risk"]
        )
        line = base.mark_line(color=RED, point=True, strokeWidth=2).encode(y=alt.Y("high_risk:Q", title="High / critical"))
        st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=330), width="stretch")
    with right:
        st.subheader("Severity mix")
        sev = view.groupby("severity", as_index=False).size()
        donut = (
            alt.Chart(sev)
            .mark_arc(innerRadius=70, outerRadius=120)
            .encode(
                theta="size:Q",
                color=alt.Color("severity:N", scale=alt.Scale(domain=list(SEVERITY_COLORS), range=list(SEVERITY_COLORS.values())), legend=alt.Legend(title=None)),
                tooltip=["severity", "size"],
            )
            .properties(height=330)
        )
        st.altair_chart(donut, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Most frequent incident types")
        types = view.groupby("incident_type", as_index=False).size().sort_values("size", ascending=False)
        chart = (
            alt.Chart(types)
            .mark_bar(color=BLUE)
            .encode(
                x=alt.X("size:Q", title="Incidents"),
                y=alt.Y("incident_type:N", sort="-x", title=None),
                tooltip=["incident_type", "size"],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, width="stretch")
    with right:
        st.subheader("Site Ã— severity heatmap")
        heat = view.groupby(["site", "severity"], as_index=False).size()
        chart = (
            alt.Chart(heat)
            .mark_rect(cornerRadius=2)
            .encode(
                x=alt.X("severity:N", sort=["Low", "Medium", "High", "Critical"], title=None),
                y=alt.Y("site:N", title=None),
                color=alt.Color("size:Q", scale=alt.Scale(scheme="blues"), title="Incidents"),
                tooltip=["site", "severity", "size"],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, width="stretch")

with site_tab:
    summaries = build_summaries(view)
    sites = summaries["site_risk_summary"]
    st.subheader("Comparative site risk index")
    st.caption("35% high-risk share Â· 20% critical volume Â· 20% p90 response Â· 15% unresolved share Â· 10% positive recent trend")
    risk_chart = (
        alt.Chart(sites)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("risk_score:Q", title="Comparative risk score (0â€“100)"),
            y=alt.Y("site:N", sort="-x", title=None),
            color=alt.Color("risk_score:Q", scale=alt.Scale(range=[GOLD, RED]), legend=None),
            tooltip=["site", "incidents", "critical_incidents", "high_risk_rate", "p90_response_min", "recent_trend_pct", "risk_score"],
        )
        .properties(height=340)
    )
    st.altair_chart(risk_chart, width="stretch")

    st.subheader("Risk versus response")
    scatter = (
        alt.Chart(sites)
        .mark_circle(opacity=0.82, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("avg_response_min:Q", title="Average response time (minutes)", scale=alt.Scale(zero=False)),
            y=alt.Y("high_risk_rate:Q", title="High / critical incidents (%)", scale=alt.Scale(zero=False)),
            size=alt.Size("incidents:Q", legend=alt.Legend(title="Incident volume")),
            color=alt.Color("risk_score:Q", scale=alt.Scale(range=[GOLD, RED]), title="Risk score"),
            tooltip=["site", "incidents", "high_risk_rate", "avg_response_min", "risk_score"],
        )
        .properties(height=380)
    )
    labels = scatter.mark_text(align="left", baseline="middle", dx=8, size=12).encode(text="site:N", size=alt.value(12), color=alt.value(NAVY))
    st.altair_chart(scatter + labels, width="stretch")
    st.dataframe(
        sites[["risk_rank", "site", "incidents", "critical_incidents", "high_risk_rate", "avg_response_min", "p90_response_min", "unresolved_rate", "recent_trend_pct", "risk_score"]],
        hide_index=True,
        width="stretch",
    )
    st.info("Site counts are not exposure-adjusted because employee population, visitor volume, floor area, and operating hours were not supplied.")

with response_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median response", f"{view['response_time_min'].median():.1f} min")
    c2.metric("90th percentile", f"{view['response_time_min'].quantile(.90):.1f} min")
    c3.metric(">15 minute responses", f"{int(view['slow_response'].sum()):,}")
    c4.metric("Within 15 min", pct(100 * (1 - view["slow_response"].mean())))
    st.caption("The 15-minute threshold is an analytical assumption, not a company-provided SLA.")

    left, right = st.columns(2)
    with left:
        st.subheader("Response distribution by severity")
        box = (
            alt.Chart(view)
            .mark_boxplot(size=38, extent="min-max")
            .encode(
                x=alt.X("severity:N", sort=["Low", "Medium", "High", "Critical"], title=None),
                y=alt.Y("response_time_min:Q", title="Response time (minutes)"),
                color=alt.Color("severity:N", scale=alt.Scale(domain=list(SEVERITY_COLORS), range=list(SEVERITY_COLORS.values())), legend=None),
            )
            .properties(height=350)
        )
        st.altair_chart(box, width="stretch")
    with right:
        st.subheader("Shift operating profile")
        shift = summaries["shift_summary"]
        chart = (
            alt.Chart(shift)
            .mark_bar(color=BLUE)
            .encode(
                x=alt.X("shift:N", title=None, sort=["Day", "Evening", "Night"]),
                y=alt.Y("avg_response_min:Q", title="Average response (minutes)", scale=alt.Scale(zero=False)),
                tooltip=["shift", "incidents", "critical", "high_risk_rate", "avg_response_min", "p90_response_min"],
            )
            .properties(height=350)
        )
        st.altair_chart(chart, width="stretch")

    st.subheader("Operational queue")
    queue = view[(view["high_risk_target"].eq(1)) | (view["slow_response"].eq(1)) | (view["is_unresolved"].eq(1))].copy()
    queue["priority"] = (
        4 * queue["severity"].eq("Critical").astype(int)
        + 2 * queue["severity"].eq("High").astype(int)
        + queue["slow_response"]
        + queue["is_unresolved"]
    )
    queue = queue.sort_values(["priority", "date"], ascending=[False, False])
    st.dataframe(queue[["incident_id", "date", "site", "incident_type", "severity", "response_time_min", "outcome"]].head(100), hide_index=True, width="stretch")

with ai_tab:
    metadata = json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(ROOT / "models" / "model_comparison.csv")
    importance = pd.read_csv(ROOT / "models" / "feature_importance.csv").head(15)
    metrics = metadata["test_metrics"]
    st.subheader("High / critical incident triage model")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Selected model", metadata["selected_model"])
    c2.metric("Recall", pct(100 * metrics["recall"]))
    c3.metric("Precision", pct(100 * metrics["precision"]))
    c4.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    c5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.markdown('<div class="callout"><b>Validation design:</b> Train on the earliest 60%, tune the decision threshold on the next 20%, and report once on the latest 20%. Severity, outcome, response time, and Incident ID are excluded from inputs.</div>', unsafe_allow_html=True)

    st.markdown("### Try an intake prediction")
    with st.form("risk_prediction"):
        cols = st.columns(3)
        site = cols[0].selectbox("Site", sorted(df["site"].unique()))
        incident_type = cols[1].selectbox("Incident type", sorted(df["incident_type"].unique()))
        shift = cols[2].selectbox("Shift", ["Day", "Evening", "Night"])
        badge = cols[0].selectbox("Badge access", ["Yes", "No", "N/A"])
        cctv = cols[1].selectbox("CCTV alert", ["Yes", "No"])
        visitor = cols[2].selectbox("Visitor involved", ["Yes", "No"])
        incident_date = cols[0].date_input("Incident date", value=df["date"].max().date())
        submitted = st.form_submit_button("Estimate triage risk", type="primary")
    if submitted:
        date = pd.Timestamp(incident_date)
        row = pd.DataFrame([{
            "site": site,
            "incident_type": incident_type,
            "shift": shift,
            "badge_access": badge,
            "cctv_alert": cctv,
            "visitor": visitor,
            "month": date.month,
            "day_of_week": date.day_name(),
            "is_weekend": date.dayofweek >= 5,
        }])
        probability = float(load_model().predict_proba(row)[0, 1])
        threshold = metadata["decision_threshold"]
        if probability >= threshold:
            st.markdown(f'<div class="risk-high"><b>Escalate for priority review</b><br>Estimated high/critical probability: <b>{probability:.1%}</b> Â· decision threshold: {threshold:.0%}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low"><b>Standard triage queue</b><br>Estimated high/critical probability: <b>{probability:.1%}</b> Â· decision threshold: {threshold:.0%}</div>', unsafe_allow_html=True)
        st.caption("This proof-of-concept output supports triage only. A security professional remains accountable for escalation decisions.")

    left, right = st.columns(2)
    with left:
        st.subheader("Model comparison on latest-period test set")
        st.dataframe(comparison, hide_index=True, width="stretch")
    with right:
        st.subheader("Most influential encoded features")
        fi = (
            alt.Chart(importance)
            .mark_bar(color=BLUE)
            .encode(x=alt.X("importance:Q", title="Model importance"), y=alt.Y("feature:N", sort="-x", title=None), tooltip=["feature", "importance"])
            .properties(height=390)
        )
        st.altair_chart(fi, width="stretch")
    st.warning(metadata["caution"])

with governance_tab:
    st.subheader("Data-quality scorecard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quality score", f"{data_quality_score(quality):.1f}/100")
    c2.metric("Blank cells", f"{sum(quality['blank_counts'].values()):,}")
    c3.metric("Duplicate IDs", quality["duplicate_incident_ids"])
    c4.metric("Invalid values", quality["invalid_dates"] + quality["invalid_response_times"])
    st.success("All 2,000 records passed schema, completeness, uniqueness, date, numeric, category, and site-country validation.")
    st.info("Badge Access contains 90 explicit N/A values. They are retained as a separate category because N/A may mean not applicable or unknown; replacing them with Yes/No would invent information.")

    st.subheader("Recommended AI operating model")
    recommendations = [
        ("1. Triage assist", "Score new incidents at intake and route high-probability cases for rapid human review."),
        ("2. SLA monitoring", "Create severity-specific response targets and alert supervisors before an incident breaches its target."),
        ("3. Site action plans", "Review Dublin's critical concentration and response profile; separate exposure volume from controllable performance."),
        ("4. Data enrichment", "Capture exact timestamps, zone, dispatch and closure times, staffing, footfall, loss impact, and short narratives."),
        ("5. Model governance", "Monitor recall, false-alert burden, calibration, site-level fairness, and drift; retrain only after approved review."),
    ]
    for title, body in recommendations:
        st.markdown(f"**{title}** â€” {body}")

    st.subheader("Responsible-use limitations")
    for item in [
        "This is a curated 2,000-row assessment sample, not a production deployment dataset.",
        "Incident type strongly encodes the severity label, inflating apparent predictability.",
        "Site counts cannot be normalized without population or activity exposure data.",
        "The model should supportâ€”not automateâ€”security escalation decisions.",
        "Prospective validation on future incidents is required before operational use.",
    ]:
        st.markdown(f"- {item}")


