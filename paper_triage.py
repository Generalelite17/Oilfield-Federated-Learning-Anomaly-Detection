from pathlib import Path
import pandas as pd

SCORED = Path("outputs/scored_output.csv")
OUT = Path("paper_outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(SCORED)

total = len(df)
anom = int(df["is_anomaly"].sum())
rate = anom / total if total else 0
reduction_factor = total / anom if anom else None

summary = pd.DataFrame([{
    "total_flows": total,
    "flagged_anomalies": anom,
    "anomaly_rate": rate,
    "triage_reduction_factor_(total/anomalies)": reduction_factor
}])

summary.to_csv(OUT / "table_triage_reduction.csv", index=False)

print(summary.to_string(index=False))
print("[SUCCESS] Saved:", (OUT / "table_triage_reduction.csv").resolve())
