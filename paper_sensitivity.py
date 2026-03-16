from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA = Path("csv_data/BenignTraffic.pcap_Flow.csv")
OUT = Path("paper_outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# Numeric-only (fast + stable for a deadline)
X = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

contaminations = [0.001, 0.005, 0.01]
rows = []

for c in contaminations:
    model = IsolationForest(n_estimators=300, contamination=c, random_state=42, n_jobs=-1)
    model.fit(Xs)
    pred = model.predict(Xs)
    anom = int((pred == -1).sum())
    rows.append({
        "contamination": c,
        "anomalies": anom,
        "anomaly_rate": anom / len(df)
    })

result = pd.DataFrame(rows)
result.to_csv(OUT / "table_sensitivity_contamination.csv", index=False)
print(result.to_string(index=False))
print("[SUCCESS] Saved:", (OUT / "table_sensitivity_contamination.csv").resolve())

