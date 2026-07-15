# Interview preparation guide

## Your 60-second explanation

“I treated this as an operational decision-support problem, not just a charting exercise. I first built a reproducible quality pipeline and preserved the original data. I then analysed incident frequency, severity, response performance, outcomes, shifts, and site differences. Because site exposure data was absent, I avoided calling raw counts incident rates. I created a transparent comparative risk index and built a leakage-aware model that predicts High/Critical risk using only intake-time variables. I validated chronologically, selected the decision threshold on validation data, and evaluated once on the latest period. The result is delivered through a Streamlit dashboard, but I position the model as human-reviewed triage support rather than autonomous decision-making.”

## Concepts you should know

### Why a chronological split?

A random split lets future patterns leak indirectly into training and does not simulate deployment. A chronological split trains on earlier records and tests on later records, closer to how a real model would encounter future incidents.

### What is data leakage?

Leakage means giving the model information unavailable when the prediction must be made. Severity is the target, while outcome and response time occur after intake; including them would produce misleading performance. Incident ID is also excluded because it is an identifier, not a causal feature.

### Why binary classification?

The operational question is whether a case deserves priority review. Combining High and Critical produces a clear positive class with 776 examples, enough for a proof of concept. Four-class prediction would be harder to validate reliably with only 165 Critical examples.

### Why compare several models?

Dummy classification establishes the no-skill benchmark. Logistic Regression is an interpretable linear baseline. Random Forest and Gradient Boosting capture nonlinear category interactions. Selection is based on measured out-of-time performance and the operating objective—not model complexity.

### Why was Random Forest selected?

Under the pre-declared threshold policy, Random Forest achieved the highest test recall while maintaining approximately 70% precision. Logistic Regression is competitive and could be preferred if maximum transparency or calibration were more important.

### Recall versus precision

- Recall: Of all High/Critical incidents, how many were caught?
- Precision: Of all incidents escalated by the model, how many were actually High/Critical?

Physical security usually values recall because missed critical incidents are costly, but excessively low precision creates alert fatigue. The threshold balances those costs.

### ROC-AUC versus PR-AUC

ROC-AUC measures overall ranking across positive and negative cases. PR-AUC focuses on performance for the positive High/Critical class and is usually more operationally informative when the positive class is the priority.

### What is the Brier score?

It is the mean squared error of predicted probabilities. Lower is better. It evaluates whether probability estimates—not only classifications—are sensible.

### Why is the model performance suspiciously strong?

Incident type strongly encodes severity. Suspicious Package and Door Forced Open are always High/Critical in this sample, while several categories are never High/Critical. This may reflect policy rules or synthetic data generation. Therefore, the result is a proof of concept and requires prospective validation.

### Is the site-risk score an ML prediction?

No. It is a transparent weighted index for comparative prioritization. Every component and weight is disclosed. It is not a probability of harm and should be sensitivity-tested with security leadership.

### Why not call Austin the riskiest site?

Austin has the most records, but the dataset lacks employee population, visitor volume, floor area, and operating hours. A larger site naturally produces more incidents. Dublin ranks higher on the chosen severity/response index, but neither result is a true exposure-adjusted incident rate.

### Why no generative AI or LLM?

The supplied data is structured tabular data with no incident narratives. Classical supervised ML is more appropriate, cheaper, easier to validate, and easier to explain. An LLM would become relevant after collecting narratives—for summarization, entity extraction, and analyst-assist—not for inventing information from the current fields.

## Numbers to remember

- 2,000 records; 8 sites; 8 countries; 11 incident types.
- Date coverage: 1 January 2025 to 30 June 2026.
- 776 High/Critical (38.8%); 165 Critical.
- Average response 11.59 minutes; median 12; 90th percentile 18.
- Austin: highest raw volume, 269.
- Dublin: most Critical incidents, 30; first comparative risk rank.
- Random Forest latest-period test: recall 81.8%, precision 70.4%, F1 75.6%, ROC-AUC 0.917, PR-AUC 0.878.
- Confusion matrix: 121 true positives, 51 false positives, 27 false negatives, 201 true negatives.

## Difficult questions

**Would you deploy this model now?**  
No. I would run it in shadow mode on future incidents, measure analyst agreement and operational burden, check drift and site-level errors, and obtain governance approval before it influences queues.

**How would you improve it?**  
Add exact timestamps, site exposure, location zone, access event history, staffing, dispatch/closure times, loss impact, recurrence, short narratives, and analyst actions. Then validate on a larger prospective dataset.

**What if leadership changes the cost of false negatives?**  
Change the threshold using a documented cost or capacity framework; do not necessarily retrain the model. Show the expected false positives and false negatives at candidate thresholds.

**How would you monitor it?**  
Track recall, precision, calibration, alert volumes, override rates, feature/score drift, site-level performance, response-time impact, and model/version lineage.

**What is the strongest business recommendation?**  
Combine a human-reviewed triage queue with severity-specific SLAs and richer data capture. The operational process and data quality are more important than model complexity.

