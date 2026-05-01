import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = Path("csv_data/BenignTraffic.pcap_Flow.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

CONTAMINATION = 0.01
RANDOM_STATE = 42

def ip_to_octets(series: pd.Series, prefix: str) -> pd.DataFrame:
    parts = series.astype(str).str.split(".", expand=True)
    if parts.shape[1] != 4:
        # handle weird/missing IPs
        parts = pd.DataFrame([[0, 0, 0, 0]] * len(series))
    parts = parts.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    parts.columns = [f"{prefix}_oct1", f"{prefix}_oct2", f"{prefix}_oct3", f"{prefix}_oct4"]
    return parts

def parse_flow_id(flow_id: pd.Series) -> pd.DataFrame:
    # Example: 192.168.137.41-157.249.81.141-51746-80-6
    parts = flow_id.astype(str).str.split("-", expand=True)
    out = pd.DataFrame(index=flow_id.index)

    # src_port, dst_port, proto are the last 3 parts (robust-ish)
    out["flow_src_port"] = pd.to_numeric(parts.iloc[:, -3], errors="coerce")
    out["flow_dst_port"] = pd.to_numeric(parts.iloc[:, -2], errors="coerce")
    out["flow_proto"] = pd.to_numeric(parts.iloc[:, -1], errors="coerce")

    return out.fillna(0)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Start with numeric columns
    X = df.select_dtypes(include=[np.number]).copy()

    # Add IP octets
    if "Src IP" in df.columns:
        X = pd.concat([X, ip_to_octets(df["Src IP"], "srcip")], axis=1)
    if "Dst IP" in df.columns:
        X = pd.concat([X, ip_to_octets(df["Dst IP"], "dstip")], axis=1)

    # Parse Flow ID ports/proto
    if "Flow ID" in df.columns:
        X = pd.concat([X, parse_flow_id(df["Flow ID"])], axis=1)

    # Timestamp features
    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        X["ts_hour"] = ts.dt.hour.fillna(0).astype(int)
        X["ts_minute"] = ts.dt.minute.fillna(0).astype(int)
        X["ts_second"] = ts.dt.second.fillna(0).astype(int)
        X["ts_weekday"] = ts.dt.weekday.fillna(0).astype(int)

    # Clean up any inf/nan
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X

print("[INFO] IDS training script started (feature-engineered)")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH.resolve()}")

df = pd.read_csv(DATA_PATH)
print(f"[INFO] Loaded dataset: rows={df.shape[0]}, cols={df.shape[1]}")

X = build_features(df)
print(f"[INFO] Final feature matrix: rows={X.shape[0]}, features={X.shape[1]}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=400,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
model.fit(X_scaled)
print("[SUCCESS] Isolation Forest trained (with IP/port/time features)")

pred = model.predict(X_scaled)
anom_rate = (pred == -1).mean()
unique, counts = np.unique(pred, return_counts=True)
print("[INFO] Prediction distribution (1=normal, -1=anomaly):", dict(zip(unique.tolist(), counts.tolist())))
print(f"[INFO] Observed anomaly rate: {anom_rate:.4%}")

artifact = {
    "model": model,
    "scaler": scaler,
    "feature_columns": X.columns.tolist(),
    "contamination": CONTAMINATION,
    "data_path": str(DATA_PATH),
}
out_path = MODEL_DIR / "isoforest_benign_feature_engineered.joblib"
joblib.dump(artifact, out_path)
print(f"[SUCCESS] Saved model artifact to: {out_path.resolve()}")
