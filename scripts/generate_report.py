"""Generate the interview-ready executive PDF from reproducible outputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.analysis import build_summaries, generate_insights
from src.data_preparation import data_quality_score, load_and_clean


NAVY = colors.HexColor("#10233F")
BLUE = colors.HexColor("#2367A8")
GOLD = colors.HexColor("#E0A82E")
RED = colors.HexColor("#C94747")
GREEN = colors.HexColor("#2B8A6E")
PALE = colors.HexColor("#F4F7FB")
GREY = colors.HexColor("#5A6878")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(18 * mm, 8 * mm, "Physical Security Intelligence Assessment · Shashank")
    canvas.drawRightString(A4[0] - 18 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def bar_chart(labels, values, title, color=BLUE, width=480, height=245):
    drawing = Drawing(width, height)
    chart = HorizontalBarChart()
    chart.x = 155
    chart.y = 25
    chart.height = height - 60
    chart.width = width - 180
    chart.data = [list(values)[::-1]]
    chart.categoryAxis.categoryNames = list(labels)[::-1]
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = 0
    chart.valueAxis.labels.fontSize = 8
    chart.bars[0].fillColor = color
    chart.bars[0].strokeColor = None
    drawing.add(chart)
    drawing.add(String(10, height - 15, title, fontName="Helvetica-Bold", fontSize=12, fillColor=NAVY))
    return drawing


def line_chart(labels, values, title, width=480, height=230):
    drawing = Drawing(width, height)
    chart = HorizontalLineChart()
    chart.x = 48
    chart.y = 40
    chart.height = height - 78
    chart.width = width - 70
    chart.data = [list(values)]
    chart.categoryAxis.categoryNames = list(labels)
    chart.categoryAxis.labels.angle = 45
    chart.categoryAxis.labels.fontSize = 6.5
    chart.valueAxis.valueMin = 0
    chart.lines[0].strokeColor = BLUE
    chart.lines[0].strokeWidth = 2
    chart.lines[0].symbol = None
    drawing.add(chart)
    drawing.add(String(10, height - 15, title, fontName="Helvetica-Bold", fontSize=12, fillColor=NAVY))
    return drawing


def styled_table(data, widths=None, header=True, font_size=8):
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CCD8E5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    table.setStyle(TableStyle(commands))
    return table


def bullets(items, styles):
    return [Paragraph(f"• {item}", styles["Body"]) for item in items]


def generate():
    df, qa = load_and_clean(ROOT / "Physical_Security_Incidents.csv")
    summaries = build_summaries(df)
    insights = generate_insights(df, summaries)
    model = json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(ROOT / "models" / "model_comparison.csv")
    site = summaries["site_risk_summary"]
    incident = summaries["incident_type_summary"]
    monthly = summaries["monthly_summary"]

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    pdf_path = out / "Physical_Security_AI_Analysis_Shashank.pdf"

    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=25, leading=31, textColor=NAVY, alignment=TA_LEFT, spaceAfter=12),
        "Subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontSize=12, leading=18, textColor=GREY, spaceAfter=12),
        "H1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=NAVY, spaceBefore=4, spaceAfter=10),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=BLUE, spaceBefore=8, spaceAfter=5),
        "Body": ParagraphStyle("Body", parent=base["BodyText"], fontSize=9.5, leading=14, textColor=colors.HexColor("#25364A"), spaceAfter=6),
        "Small": ParagraphStyle("Small", parent=base["BodyText"], fontSize=8, leading=11, textColor=GREY),
        "Center": ParagraphStyle("Center", parent=base["BodyText"], fontSize=9, leading=13, textColor=NAVY, alignment=TA_CENTER),
        "Callout": ParagraphStyle("Callout", parent=base["BodyText"], fontSize=10, leading=15, textColor=NAVY, borderColor=GOLD, borderWidth=1, borderPadding=9, backColor=colors.HexColor("#FFF9EA"), spaceAfter=10),
    }

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm, title="Physical Security AI & Data Analytics Assessment",
        author="Shashank",
    )
    story = []

    story += [Spacer(1, 25 * mm), Paragraph("PHYSICAL SECURITY", styles["Subtitle"]), Paragraph("AI & Data Analytics Assessment", styles["Title"])]
    story += [Paragraph("Global incident intelligence · Risk prioritization · Response operations · Explainable machine learning", styles["Subtitle"]), Spacer(1, 10 * mm)]
    cover = [
        ["Prepared by", "Shashank"],
        ["Dataset", "2,000 physical security incidents across 8 global sites"],
        ["Coverage", f"{insights['overview']['date_min']} to {insights['overview']['date_max']}"],
        ["Deliverables", "Interactive Streamlit dashboard, reproducible Python pipeline, model, and report"],
    ]
    story += [styled_table(cover, widths=[35 * mm, 115 * mm], header=False, font_size=9.5), Spacer(1, 30 * mm)]
    story += [Paragraph("Purpose", styles["H2"]), Paragraph("Convert a supplied assessment dataset into defensible operational insight and demonstrate how a governed AI triage model could support—not replace—physical security decision-making.", styles["Callout"]), PageBreak()]

    story += [Paragraph("1. Executive summary", styles["H1"])]
    kpis = [
        ["Incidents", "High / critical", "Critical", "Avg response", "Confirmed"],
        [f"{len(df):,}", f"{insights['overview']['high_critical_count']:,} ({insights['overview']['high_critical_rate']:.1f}%)", f"{insights['overview']['critical_count']:,}", f"{insights['overview']['avg_response_min']:.1f} min", f"{insights['overview']['confirmed_rate']:.1f}%"],
    ]
    story += [styled_table(kpis, widths=[30 * mm] * 5, font_size=9), Spacer(1, 6 * mm)]
    story += bullets(insights["findings"], styles)
    story += [Paragraph("Leadership interpretation", styles["H2"]), Paragraph("Dublin warrants the first management review because it combines the highest critical count, the highest comparative risk score, and the slowest average response. Austin has the highest volume, but without site-exposure denominators it should not automatically be labelled the least safe location.", styles["Callout"])]
    story += [Paragraph("Recommended action", styles["H2"]), Paragraph("Pilot an AI-assisted intake queue with human approval, establish severity-specific response SLAs, investigate site-level drivers, enrich the incident schema, and prospectively validate performance before operational deployment.", styles["Body"]), PageBreak()]

    story += [Paragraph("2. Data quality and preprocessing", styles["H1"])]
    qa_rows = [
        ["Check", "Result", "Treatment"],
        ["Schema", "12/12 expected fields", "Standardized names; original CSV preserved"],
        ["Completeness", "0 blank cells", "No imputation required"],
        ["Uniqueness", "0 duplicate IDs / rows", "Incident ID retained only for traceability"],
        ["Validity", "0 invalid dates or response times", "Typed date and numeric response fields"],
        ["Categories", "0 unexpected values", "Explicit category validation"],
        ["Badge Access N/A", "90 records", "Retained as a distinct category; meaning is ambiguous"],
        ["Quality score", f"{data_quality_score(qa):.1f}/100", "Transparent issue-count score, not a universal standard"],
    ]
    story += [styled_table(qa_rows, widths=[34 * mm, 42 * mm, 78 * mm], font_size=8.5), Spacer(1, 6 * mm)]
    story += [Paragraph("Quality caution", styles["H2"]), Paragraph("The file is unusually complete, consistent, and balanced for operational incident data. It may be curated or synthetic for assessment purposes. The analysis therefore emphasizes method, transparency, and limitations rather than treating small observed differences as causal evidence.", styles["Callout"])]
    story += [Paragraph("Engineered fields", styles["H2"])]
    story += bullets(["Calendar features: year, month, year-month, weekday, and weekend indicator.", "Operational flags: high/critical target, confirmed, under investigation, false alarm, and response above 15 minutes.", "Severity weights used only in descriptive prioritization—not as model inputs."], styles)
    story += [PageBreak()]

    story += [Paragraph("3. Exploratory findings", styles["H1"])]
    story += [line_chart(monthly["year_month"], monthly["incidents"], "Monthly incident volume"), Spacer(1, 4 * mm)]
    story += [bar_chart(incident["incident_type"], incident["incidents"], "Incident volume by type"), Spacer(1, 4 * mm)]
    story += [Paragraph("Interpretation", styles["H2"]), Paragraph("Lost Badge is the largest incident category. Suspicious Package and Door Forced Open are entirely labelled high/critical in this sample, while several categories never receive those labels. This strong label structure is operationally useful for rule-based escalation but also explains much of the model's predictive power.", styles["Body"]), PageBreak()]

    story += [Paragraph("4. Site risk and response operations", styles["H1"])]
    story += [bar_chart(site["site"], site["risk_score"], "Transparent comparative site risk score", RED), Spacer(1, 4 * mm)]
    site_rows = [["Rank", "Site", "Incidents", "Critical", "High-risk %", "Avg response", "P90 response", "Score"]]
    for _, r in site.iterrows():
        site_rows.append([int(r.risk_rank), r.site, int(r.incidents), int(r.critical_incidents), f"{r.high_risk_rate:.1f}%", f"{r.avg_response_min:.1f}", f"{r.p90_response_min:.1f}", f"{r.risk_score:.1f}"])
    story += [styled_table(site_rows, widths=[12 * mm, 25 * mm, 21 * mm, 18 * mm, 22 * mm, 23 * mm, 22 * mm, 16 * mm], font_size=7.7), Spacer(1, 5 * mm)]
    story += [Paragraph("Risk-index formula", styles["H2"]), Paragraph("35% high/critical share + 20% critical count + 20% 90th-percentile response + 15% under-investigation share + 10% positive recent trend. Each component is min-max normalized across the eight sites. It is a comparative prioritization index—not a calibrated probability of harm.", styles["Body"])]
    story += [Paragraph("Response finding", styles["H2"]), Paragraph(f"Overall response averages {df['response_time_min'].mean():.2f} minutes; median is {df['response_time_min'].median():.0f} and the 90th percentile is {df['response_time_min'].quantile(.90):.0f}. A 15-minute threshold is shown only as an analytical scenario because no official SLA was supplied.", styles["Callout"]), PageBreak()]

    story += [Paragraph("5. AI/ML approach", styles["H1"])]
    story += [Paragraph("Use case", styles["H2"]), Paragraph("At incident intake, estimate the probability that the case will be classified High or Critical and prioritize cases above an approved decision threshold for human review.", styles["Body"])]
    model_rows = [["Design choice", "Implementation"]]
    model_rows += [
        ["Target", "1 = High/Critical; 0 = Low/Medium"],
        ["Inputs", "Site, incident type, shift, badge, CCTV, visitor, month, weekday, weekend"],
        ["Leakage exclusions", "Severity, outcome, response time, and Incident ID"],
        ["Validation", "Earliest 60% train; next 20% validation; latest 20% untouched test"],
        ["Threshold", f"{model['decision_threshold']:.2f}; maximize validation recall subject to precision ≥ 0.70"],
        ["Selected model", model["selected_model"]],
        ["Human role", "Model recommends queue priority; security professional owns escalation"],
    ]
    story += [styled_table(model_rows, widths=[38 * mm, 116 * mm], font_size=8.5), Spacer(1, 6 * mm)]
    comparison_rows = [["Model", "Recall", "Precision", "F1", "ROC-AUC", "PR-AUC", "Brier"]]
    for _, r in comparison.iterrows():
        comparison_rows.append([r["model"], f"{r['recall']:.3f}", f"{r['precision']:.3f}", f"{r['f1']:.3f}", f"{r['roc_auc']:.3f}", f"{r['pr_auc']:.3f}", f"{r['brier_score']:.3f}"])
    story += [styled_table(comparison_rows, widths=[38 * mm, 19 * mm, 20 * mm, 17 * mm, 21 * mm, 20 * mm, 19 * mm], font_size=8), Spacer(1, 6 * mm)]
    cm = model["test_metrics"]["confusion_matrix"]
    story += [Paragraph("Latest-period test result", styles["H2"]), Paragraph(f"The selected model identifies {cm['tp']} of {cm['tp'] + cm['fn']} high-risk incidents (recall {model['test_metrics']['recall']:.1%}) while {cm['fp']} lower-risk cases are also escalated. Precision is {model['test_metrics']['precision']:.1%}; PR-AUC is {model['test_metrics']['pr_auc']:.3f}; ROC-AUC is {model['test_metrics']['roc_auc']:.3f}.", styles["Callout"]), PageBreak()]

    story += [Paragraph("6. Validation, governance, and limitations", styles["H1"])]
    story += [Paragraph("Why these metrics?", styles["H2"])]
    story += bullets(["Recall measures how many truly high-risk cases are caught; a missed critical incident is costly.", "Precision measures alert burden and guards against alert fatigue.", "F1 balances recall and precision at the operating threshold.", "PR-AUC evaluates ranking quality for the positive operational class.", "ROC-AUC measures overall separation; Brier score evaluates probability accuracy/calibration.", "The confusion matrix translates model scores into operational case volumes."], styles)
    story += [Paragraph("Required production controls", styles["H2"])]
    story += bullets(["Prospective shadow-mode validation on future incidents.", "Monthly monitoring of recall, precision, calibration, alert volume, and drift.", "Site-level performance checks to identify uneven error rates.", "Versioned data, model, threshold, and approvals with an audit trail.", "Human override, escalation documentation, and periodic threshold review."], styles)
    story += [Paragraph("Limitations", styles["H2"])]
    story += bullets(insights["limitations"], styles)
    story += [Paragraph("Most important modelling caveat", styles["Callout"]), Paragraph(model["caution"], styles["Body"]), PageBreak()]

    story += [Paragraph("7. Actionable roadmap", styles["H1"])]
    roadmap = [
        ["Horizon", "Action", "Success measure"],
        [Paragraph("0-30 days", styles["Small"]), Paragraph("Define severity-specific SLAs; review Dublin critical cases; standardize closure reasons.", styles["Small"]), Paragraph("Approved SLA and site action plan", styles["Small"])],
        [Paragraph("30-60 days", styles["Small"]), Paragraph("Run AI triage in shadow mode; compare model queue with analyst decisions.", styles["Small"]), Paragraph("Recall, precision, alert load and overrides", styles["Small"])],
        [Paragraph("60-90 days", styles["Small"]), Paragraph("Add timestamps, zones, staffing, footfall, impact, narratives and exposure denominators.", styles["Small"]), Paragraph("Improved data completeness and normalized rates", styles["Small"])],
        [Paragraph("90+ days", styles["Small"]), Paragraph("Controlled pilot with human approval, drift monitoring and quarterly governance review.", styles["Small"]), Paragraph("Faster high-risk response without alert fatigue", styles["Small"])],
    ]
    story += [styled_table(roadmap, widths=[23 * mm, 78 * mm, 53 * mm], font_size=8.3), Spacer(1, 8 * mm)]
    story += [Paragraph("Final recommendation", styles["H2"]), Paragraph("Treat the model as a prioritization layer within an accountable operating process. The strongest near-term value comes from combining transparent analytics, consistent response targets, richer incident capture, and human-reviewed risk scoring—not from autonomous security decisions.", styles["Callout"])]
    story += [Paragraph("Supporting artefacts", styles["H2"])]
    story += bullets(["Interactive Streamlit dashboard with filters and intake prediction.", "Reproducible cleaning, analysis, model training, and report scripts.", "Cleaned dataset, quality report, site summaries, model artefacts, and test predictions.", "README, notebook walkthrough, and reproduction instructions."], styles)

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Generated {pdf_path}")


if __name__ == "__main__":
    generate()
