from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Use whichever scored file you have
SCORED_PATH = Path("outputs/scored_output.csv")  # baseline
# If you later produce a feature-engineered scored file, switch to it:
# SCORED_PATH = Path("outputs/scored_output_feature_engineered.csv")

OUT_DIR = Path("paper_outputs")
OUT_DIR.mkdir(exist_ok=True)

def parse_flow_id(flow_id: pd.Series) -> pd.DataFrame:
    # Example: 192.168.137.41-157.249.81.141-51746-80-6
    parts = flow_id.astype(str).str.split("-", expand=True)
    out = pd.DataFrame(index=flow_id.index)
    out["src_port"] = pd.to_numeric(parts.iloc[:, -3], errors="coerce")
    out["dst_port"] = pd.to_numeric(parts.iloc[:, -2], errors="coerce")
    out["proto"] = pd.to_numeric(parts.iloc[:, -1], errors="coerce")
    return out

# -------------------
# Load scored results
# -------------------
if not SCORED_PATH.exists():
    raise FileNotFoundError(f"Could not find {SCORED_PATH.resolve()}")

df = pd.read_csv(SCORED_PATH)

required_cols = {"is_anomaly", "anomaly_score"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns in scored file: {missing}")

# -------------------
# Summary metrics
# -------------------
total = len(df)
anom = int(df["is_anomaly"].sum())
rate = anom / total if total else 0.0

summary = pd.DataFrame([{
    "total_flows": total,
    "anomalies": anom,
    "anomaly_rate": rate,
    "score_min": float(df["anomaly_score"].min()),
    "score_p01": float(df["anomaly_score"].quantile(0.01)),
    "score_median": float(df["anomaly_score"].median()),
    "score_p99": float(df["anomaly_score"].quantile(0.99)),
    "score_max": float(df["anomaly_score"].max()),
}])

summary_path = OUT_DIR / "table_summary_metrics.csv"
summary.to_csv(summary_path, index=False)

# -------------------
# Score distribution figure
# -------------------
plt.figure()
plt.hist(df["anomaly_score"].dropna(), bins=60)
plt.title("Anomaly Score Distribution (Isolation Forest)")
plt.xlabel("anomaly_score (lower = more anomalous)")
plt.ylabel("count")
fig1_path = OUT_DIR / "fig_score_distribution.png"
plt.savefig(fig1_path, dpi=200, bbox_inches="tight")
plt.close()

# -------------------
# Top anomalies table (human readable)
# -------------------
cols_prefer = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label", "anomaly_score", "is_anomaly"]
cols_use = [c for c in cols_prefer if c in df.columns]

top = df.sort_values("anomaly_score").head(20).copy()

# Add parsed ports/proto if Flow ID exists
if "Flow ID" in top.columns:
    parsed = parse_flow_id(top["Flow ID"])
    top = pd.concat([top, parsed], axis=1)

top_path = OUT_DIR / "table_top20_anomalies.csv"
top[cols_use + [c for c in ["src_port", "dst_port", "proto"] if c in top.columns]].to_csv(top_path, index=False)

# -------------------
# (Optional) Port distribution among anomalies
# -------------------
if "Flow ID" in df.columns:
    parsed_all = parse_flow_id(df["Flow ID"])
    df2 = pd.concat([df[["is_anomaly"]], parsed_all], axis=1)

    anom_ports = df2[df2["is_anomaly"] == True]["dst_port"].dropna()
    if len(anom_ports) > 0:
        top_ports = anom_ports.value_counts().head(15)

        plt.figure()
        plt.bar(top_ports.index.astype(str), top_ports.values)
        plt.title("Top Destination Ports Among Flagged Anomalies")
        plt.xlabel("dst_port")
        plt.ylabel("count")
        plt.xticks(rotation=45, ha="right")
        fig2_path = OUT_DIR / "fig_top_dst_ports_anomalies.png"
        plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
        plt.close()

print("[SUCCESS] Wrote paper artifacts to:", OUT_DIR.resolve())
print(" -", summary_path.name)
print(" -", fig1_path.name)
print(" -", top_path.name)
