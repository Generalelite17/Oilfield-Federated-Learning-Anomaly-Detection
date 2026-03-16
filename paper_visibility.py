from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCORED = Path("outputs/scored_output.csv")  # change if needed
OUT = Path("paper_outputs")
OUT.mkdir(exist_ok=True)

def parse_flow_id(flow_id: pd.Series) -> pd.DataFrame:
    parts = flow_id.astype(str).str.split("-", expand=True)
    out = pd.DataFrame(index=flow_id.index)
    out["src_port"] = pd.to_numeric(parts.iloc[:, -3], errors="coerce")
    out["dst_port"] = pd.to_numeric(parts.iloc[:, -2], errors="coerce")
    out["proto"] = pd.to_numeric(parts.iloc[:, -1], errors="coerce")
    return out

df = pd.read_csv(SCORED)

# Ensure required columns exist
for c in ["is_anomaly", "anomaly_score"]:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

# Parse ports/proto from Flow ID if available
if "Flow ID" in df.columns:
    df = pd.concat([df, parse_flow_id(df["Flow ID"])], axis=1)

anom = df[df["is_anomaly"] == True].copy()

# ---- Top destination ports among anomalies ----
if "dst_port" in anom.columns:
    top_ports = anom["dst_port"].dropna().astype(int).value_counts().head(15)
    top_ports.to_csv(OUT / "table_top_dst_ports_anomalies.csv", header=["count"])

    plt.figure()
    plt.bar(top_ports.index.astype(str), top_ports.values)
    plt.title("Top Destination Ports Among Flagged Anomalies")
    plt.xlabel("dst_port")
    plt.ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(OUT / "fig_top_dst_ports_anomalies.png", dpi=200, bbox_inches="tight")
    plt.close()

# ---- Top destination IPs among anomalies ----
if "Dst IP" in anom.columns:
    top_dips = anom["Dst IP"].astype(str).value_counts().head(15)
    top_dips.to_csv(OUT / "table_top_dst_ips_anomalies.csv", header=["count"])

print("[SUCCESS] Wrote visibility artifacts to:", OUT.resolve())
