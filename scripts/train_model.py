"""Train and export the chronological incident triage model."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_preparation import load_and_clean
from src.modeling import train_and_save


if __name__ == "__main__":
    frame, _ = load_and_clean(ROOT / "Physical_Security_Incidents.csv")
    metadata = train_and_save(frame, ROOT / "models")
    print(f"Selected model: {metadata['selected_model']}")
    print(f"Decision threshold: {metadata['decision_threshold']:.2f}")
    print("Latest-period test metrics:")
    for name, value in metadata["test_metrics"].items():
        print(f"- {name}: {value}")
