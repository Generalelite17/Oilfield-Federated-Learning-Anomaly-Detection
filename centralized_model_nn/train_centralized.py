import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import json

DATA_PATH = Path("../csv_data/CIC/BenignTraffic.pcap_Flow.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CONTAMINATION = 0.01
RANDOM_STATE = 42
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001


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


class CentralizedNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)


print("[INFO] Centralized NN benchmark started")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH.resolve()}")

df = pd.read_csv(DATA_PATH)
print(f"[INFO] Loaded dataset: rows={df.shape[0]}, cols={df.shape[1]}")

X = build_features(df)
print(f"[INFO] Feature matrix: rows={X.shape[0]}, features={X.shape[1]}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: Generate pseudo-labels using Isolation Forest
iso = IsolationForest(
    n_estimators=400,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

iso_pred = iso.fit_predict(X_scaled)

# Convert Isolation Forest output:
# 1 = normal -> 0
# -1 = anomaly -> 1
y = np.where(iso_pred == -1, 1, 0)

unique, counts = np.unique(y, return_counts=True)
print("[INFO] Pseudo-label distribution (0=normal, 1=anomaly):", dict(zip(unique.tolist(), counts.tolist())))

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Step 3: Train centralized neural network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CentralizedNN(input_dim=X_train.shape[1]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[INFO] Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# Step 4: Evaluate centralized model

model.eval()

with torch.no_grad():
    outputs = model(X_test_tensor.to(device))
    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print("\n[SUCCESS] Centralized NN Results")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n[INFO] Classification Report")
print(classification_report(y_test, predictions, zero_division=0))

results = {
    "model": "Centralized Neural Network",
    "dataset": str(DATA_PATH),
    "pseudo_label_method": "Isolation Forest",
    "contamination": CONTAMINATION,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "rows": int(df.shape[0]),
    "features": int(X.shape[1]),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE
}

with open(RESULTS_DIR / "centralized_nn_results.json", "w") as f:
    json.dump(results, f, indent=4)

torch.save(model.state_dict(), RESULTS_DIR / "centralized_nn_model.pth")

print(f"[SUCCESS] Results saved to: {(RESULTS_DIR / 'centralized_nn_results.json').resolve()}")
print(f"[SUCCESS] Model saved to: {(RESULTS_DIR / 'centralized_nn_model.pth').resolve()}")