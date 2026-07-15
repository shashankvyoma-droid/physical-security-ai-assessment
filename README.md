# Physical Security AI & Data Analytics Assessment

An end-to-end assessment of 2,000 physical security incidents across eight global sites. The project combines data-quality validation, exploratory analytics, an explainable site-risk index, time-aware machine-learning evaluation, and an interactive Streamlit dashboard.

## Key results

- 2,000 valid records, 12 source fields, no blank cells, duplicate IDs, invalid dates, or invalid response times.
- 776 High/Critical incidents (38.8%); 165 Critical incidents.
- Austin has the highest raw volume (269), while Dublin has the most Critical incidents (30) and ranks first on the comparative risk index.
- Average response time is 11.59 minutes; median is 12 minutes; 90th percentile is 18 minutes.
- Selected triage model: Random Forest. Latest-period test recall 81.8%, precision 70.4%, F1 75.6%, ROC-AUC 0.917, PR-AUC 0.878.
- Important caution: incident type strongly encodes severity in this curated sample. Model performance is a proof of concept, not evidence of production generalization.

## Repository structure

```text
.
├── Physical_Security_Incidents.csv       # Original data, preserved unchanged
├── streamlit_app.py                      # Interactive dashboard entrypoint
├── requirements.txt                      # Pinned deployment dependencies
├── data/processed/                       # Cleaned data and quality report
├── models/                               # Trained pipeline, metrics, predictions
├── notebooks/                            # Guided reproducible walkthrough
├── outputs/analysis/                     # EDA and risk-score summaries
├── reports/                              # Executive PDF and interview guide
├── scripts/                              # Pipeline, training, report generation
└── src/                                  # Reusable preparation, analysis, modelling
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

## Deployment

1. Push this folder to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Create an app from the repository and select `streamlit_app.py`.
4. Select a compatible Python version and deploy.
5. Verify every tab and the prediction form before sharing the URL.

The sample contains no credentials or secrets. Review organizational policy before publicly hosting any real security dataset.

## Limitations

- No site population, footfall, building size, operating hours, or staffing data is supplied, so raw site counts are not normalized incident rates.
- No exact time, zone, dispatch/closure timestamps, loss impact, narrative, or intervention data is available.
- The assessment sample is unusually clean and may be curated or synthetic.
- The model must be prospectively validated before operational use and should never autonomously determine security escalation.

