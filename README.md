# Physical Security AI & Data Analytics Assessment

An end-to-end assessment of 2,000 physical security incidents across eight global sites. The project combines data-quality validation, exploratory analytics, an explainable site-risk index, time-aware machine-learning evaluation, and an interactive Streamlit dashboard.

- **Live dashboard:** https://physical-security-ai-shashank.streamlit.app/
- **Executive report:** `reports/Physical_Security_AI_Analysis_Shashank.pdf`

## Key results

- 2,000 valid records, 12 source fields, no blank cells, duplicate IDs, invalid dates, or invalid response times.
- 776 High/Critical incidents (38.8%); 165 Critical incidents.
- Austin has the highest raw volume (269), while Dublin has the most Critical incidents (30) and ranks first on the comparative risk index.
- Average response time is 11.59 minutes; median is 12 minutes; 90th percentile is 18 minutes.
- Selected triage model: Random Forest. Latest-period test recall 81.8%, precision 70.4%, F1 75.6%, ROC-AUC 0.917, PR-AUC 0.878.
- Important caution: incident type strongly encodes severity in this curated sample. Model performance is a proof of concept, not evidence of production generalization.

## How to read the dashboard

### Executive overview

- **Incidents:** Number of records remaining after the selected filters.
- **High / critical:** Number of incidents labelled High or Critical. The accompanying percentage is the share of filtered incidents, not a change over time. With no filters, `776 / 2,000 = 38.8%`.
- **Critical:** Number of Critical incidents.
- **Average response:** Arithmetic mean of response time in minutes.
- **Confirmed:** Percentage of incidents with outcome `Confirmed`.

The trend chart shows monthly incident volume and monthly High/Critical volume. The severity chart shows the Low, Medium, High and Critical mix. The heatmap compares site and severity counts.

### Site risk

- **High-risk rate:** Percentage of a site's incidents classified High or Critical.
- **Critical incidents:** Count of Critical incidents at the site.
- **P90 response:** The response time within which 90% of incidents were handled; the slowest 10% took longer.
- **Risk score:** A transparent comparative index, not an ML probability or a true exposure-adjusted safety rate.

The Risk versus Response chart uses bubble size for incident volume and color for the comparative risk score. Site counts cannot be normalized without employee population, visitor volume, floor area or operating-hour data.

### Response operations

- **Median response:** Middle response time after sorting all response times.
- **90th percentile:** Overall P90 response time. In this sample it is 18 minutes, meaning 90% of incidents were handled within 18 minutes.
- **Above 15 minutes:** An illustrative slow-response threshold. Fifteen minutes was chosen as a round scenario between the 12-minute median and 18-minute P90; it is not a company-provided SLA.

In production, security leadership should define severity- and incident-specific SLAs.

### AI triage

The model estimates whether a newly reported incident is likely to be High/Critical. It uses site, incident type, shift, badge access, CCTV alert, visitor involvement and calendar information. The selected date supplies only month, weekday and weekend/weekday features; the model does not infer ten-day cycles or predict when the next incident will occur.

- **Threshold:** Probability boundary used to recommend priority review. Lower thresholds catch more High/Critical incidents but create more false alerts.
- **Accuracy:** Percentage of all test predictions that were correct.
- **Precision:** Of all escalated incidents, the percentage actually High/Critical.
- **Recall:** Of all genuinely High/Critical incidents, the percentage successfully identified.
- **F1:** Harmonic balance of precision and recall.
- **ROC-AUC:** Ability to rank a random High/Critical case above a random Low/Medium case across thresholds.
- **PR-AUC:** Precision-recall performance focused on the High/Critical class.
- **Brier score:** Mean squared error of predicted probabilities; lower is better.

Random Forest was selected under the operational objective of maximizing High/Critical recall while maintaining approximately 70% precision. It caught 121 of 148 High/Critical test incidents versus 120 for Logistic Regression. Logistic Regression was slightly stronger on precision, F1, ROC-AUC, PR-AUC and calibration, so Random Forest is preferred for this specific recall-first security objective rather than universally superior.

### Encoded features and importance

Categorical fields are converted into one-hot Yes/No indicators such as `incident_type_Suspicious Package`. Random Forest importance measures how much each encoded feature reduced classification uncertainty across the trees. Values approximately sum to 1. Importance shows what the model relied on, not causation. Incident type dominates because the supplied sample strongly links some incident types with severity.

### Data and governance

This section records quality checks, data limitations, responsible-use controls and the proposed implementation roadmap. The model is intended to support a human-reviewed queue, not autonomously decide security escalation.

## Repository structure

```text
.
|-- Physical_Security_Incidents.csv       # Original data, preserved unchanged
|-- streamlit_app.py                      # Interactive dashboard entrypoint
|-- requirements.txt                      # Pinned deployment dependencies
|-- data/processed/                       # Cleaned data and quality report
|-- models/                               # Trained pipeline, metrics, predictions
|-- notebooks/                            # Guided reproducible walkthrough
|-- outputs/analysis/                     # EDA and risk-score summaries
|-- reports/                              # Executive PDF report
|-- scripts/                              # Pipeline, training, report generation
`-- src/                                  # Reusable preparation, analysis, modelling
```

## Reproduce the project

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\run_pipeline.py
python scripts\train_model.py
python scripts\generate_report.py
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

## Model design

The binary target is `1 = High/Critical` and `0 = Low/Medium`. Inputs are limited to fields reasonably available at intake: site, incident type, shift, badge access, CCTV alert, visitor involvement, and calendar features. Severity, outcome, response time, and Incident ID are excluded to prevent target leakage.

Records are ordered chronologically: the earliest 60% train candidate models, the next 20% selects the operating threshold, and the latest 20% is held out for final evaluation. The threshold maximizes recall while requiring at least 70% validation precision.

## Risk-index definition

The site index weights normalized components: 35% high-risk share, 20% Critical count, 20% 90th-percentile response time, 15% under-investigation share, and 10% positive recent trend. It is a comparative prioritization aid, not a probability or causal safety measure.

## Limitations

- No site population, footfall, building size, operating hours, or staffing data is supplied, so raw site counts are not normalized incident rates.
- No exact time, zone, dispatch/closure timestamps, loss impact, narrative, or intervention data is available.
- The assessment sample is unusually clean and may be curated or synthetic.
- The model must be prospectively validated before operational use and should never autonomously determine security escalation.
