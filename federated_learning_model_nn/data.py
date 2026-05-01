import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch

from config import (
    DATA_PATH,
    RANDOM_STATE,
    CONTAMINATION,
    NUM_CLIENTS,
    TEST_SIZE,
    BATCH_SIZE,
)


def ip_to_octets(series: pd.Series, prefix: str) -> pd.DataFrame:
    parts = series.astype(str).str.split(".", expand=True)

    if parts.shape[1] != 4:
        parts = pd.DataFrame([[0, 0, 0, 0]] * len(series), index=series.index)

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


def load_dataset():
    print("[INFO] Loading dataset for federated NN benchmark")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded dataset: rows={df.shape[0]}, cols={df.shape[1]}")

    X = build_features(df)
    print(f"[INFO] Feature matrix: rows={X.shape[0]}, features={X.shape[1]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=400,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    iso_pred = iso.fit_predict(X_scaled)

    # 0 = normal, 1 = pseudo-anomaly
    y = np.where(iso_pred == -1, 1, 0)

    unique, counts = np.unique(y, return_counts=True)
    print(
        "[INFO] Pseudo-label distribution (0=normal, 1=anomaly):",
        dict(zip(unique.tolist(), counts.tolist())),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
        shuffle=True,
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    input_dim = X_train.shape[1]

    return train_dataset, test_dataset, input_dim, {
        "rows": int(df.shape[0]),
        "features": int(X.shape[1]),
        "pseudo_label_distribution": dict(zip(unique.tolist(), counts.tolist())),
    }


def create_client_loaders(train_dataset):
    total_size = len(train_dataset)
    base_size = total_size // NUM_CLIENTS
    lengths = [base_size] * NUM_CLIENTS
    lengths[-1] += total_size - sum(lengths)

    generator = torch.Generator().manual_seed(RANDOM_STATE)
    client_datasets = random_split(train_dataset, lengths, generator=generator)

    client_loaders = [
        DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)
        for client_dataset in client_datasets
    ]

    return client_loaders


def create_test_loader(test_dataset):
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)