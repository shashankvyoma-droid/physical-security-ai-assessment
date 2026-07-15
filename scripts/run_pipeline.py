"""Run the reproducible data preparation pipeline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_preparation import data_quality_score, load_and_clean, save_outputs
from src.analysis import save_analysis_outputs


if __name__ == "__main__":
    source = ROOT / "Physical_Security_Incidents.csv"
    output = ROOT / "data" / "processed"
    frame, report = load_and_clean(source)
    report["quality_score"] = round(data_quality_score(report), 2)
    save_outputs(frame, report, output)
    insights = save_analysis_outputs(frame, ROOT / "outputs" / "analysis")
    print(f"Prepared {len(frame):,} records")
    print(f"Data quality score: {report['quality_score']:.2f}/100")
    print(f"Output: {output}")
    print("Key findings:")
    for finding in insights["findings"]:
        print(f"- {finding}")
