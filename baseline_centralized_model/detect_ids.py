import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = Path("models/isoforest_benign_feature_engineered.joblib")
INPUT_CSV = Path("csv_data/BenignTraffic.pcap_Flow.csv")  # change this when testing other files

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Feature engineering helpers (same as train_ids.py)
# -----------------------------
def ip_to_octets(series: pd.Series, prefix: str) -> pd.DataFrame:
    parts = series.astype(str).str.split(".", expand=True)
    if parts.shape[1] != 4:
        parts = pd.DataFrame([[0, 0, 0, 0]] * len(series))
    parts = parts.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    parts.columns = [f"{prefix}_oct1", f"{prefix}_oct2", f"{prefix}_oct3", f"{prefix}_oct4"]
    return parts

def parse_flow_id(flow_id: pd.Series) -> pd.DataFrame:
    parts = flow_id.astype(str).str.split("-", expand=True)
    out = pd.DataFrame(index=flow_id.index)
    out["flow_src_port"] = pd.to_numeric(parts.iloc[:, -3], errors="coerce")
    out["flow_dst_port"] = pd.to_numeric(parts.iloc[:, -2], errors="coerce")
    out["flow_proto"] = pd.to_numeric(parts.iloc[:, -1], errors="coerce")
    return out.fillna(0)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.select_dtypes(include=[np.number]).copy()

    if "Src IP" in df.columns:
        X = pd.concat([X, ip_to_octets(df["Src IP"], "srcip")], axis=1)
    if "Dst IP" in df.columns:
        X = pd.concat([X, ip_to_octets(df["Dst IP"], "dstip")], axis=1)

    if "Flow ID" in df.columns:
        X = pd.concat([X, parse_flow_id(df["Flow ID"])], axis=1)

    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        X["ts_hour"] = ts.dt.hour.fillna(0).astype(int)
        X["ts_minute"] = ts.dt.minute.fillna(0).astype(int)
        X["ts_second"] = ts.dt.second.fillna(0).astype(int)
        X["ts_weekday"] = ts.dt.weekday.fillna(0).astype(int)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

# -----------------------------
# Load model artifact
# -----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH.resolve()}\nRun train_ids.py first.")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
scaler = artifact["scaler"]
feature_cols = artifact["feature_columns"]

# -----------------------------
# Load input CSV and score
# -----------------------------
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found at: {INPUT_CSV.resolve()}")

df = pd.read_csv(INPUT_CSV)

X = build_features(df)

# Align columns to the training feature set (super important)
X = X.reindex(columns=feature_cols, fill_value=0)

X_scaled = scaler.transform(X)

pred = model.predict(X_scaled)               # 1 normal, -1 anomaly
score = model.decision_function(X_scaled)    # lower = more anomalous

df_out = df.copy()
df_out["is_anomaly"] = (pred == -1)
df_out["anomaly_score"] = score

print("[INFO] Total rows:", len(df_out))
print("[INFO] Anomalies:", int(df_out["is_anomaly"].sum()))

# Show worst 20
print("\n[TOP 20 MOST ANOMALOUS ROWS]")
print(df_out.sort_values("anomaly_score").head(20)[["anomaly_score", "is_anomaly"]].to_string(index=False))

out_file = OUT_DIR / "scored_output_feature_engineered.csv"
df_out.to_csv(out_file, index=False)
print("[SUCCESS] Saved:", out_file.resolve())
