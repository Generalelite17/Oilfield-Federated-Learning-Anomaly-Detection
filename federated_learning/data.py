import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

from config import DATASET, NUM_CLIENTS


def load_cic_data():
    df = pd.read_csv("../csv_data/CIC/BenignTraffic.pcap_Flow.csv", low_memory=False)

    # Benign-only baseline experiment
    df["label"] = 0

    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


def load_edge_data():
    benign = pd.read_csv("csv_data/EDGE/Modbus.csv", low_memory=False)
    attack = pd.read_csv("csv_data/EDGE/DDoS_TCP_SYN_Flood_attack.csv", low_memory=False)
    port = pd.read_csv("csv_data/EDGE/Port_Scanning_attack.csv", low_memory=False)
    sql = pd.read_csv("csv_data/EDGE/SQL_injection_attack.csv", low_memory=False)

    benign["label"] = 0
    attack["label"] = 1
    port["label"] = 2
    sql["label"] = 3

    n_samples = min(len(benign), len(attack), len(port), len(sql), 3500)

    benign = benign.sample(n=n_samples, random_state=42)
    attack = attack.sample(n=n_samples, random_state=42)
    port = port.sample(n=n_samples, random_state=42)
    sql = sql.sample(n=n_samples, random_state=42)

    df = pd.concat([benign, attack, port, sql], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    return df


def load_and_prepare_data():
    if DATASET == "CIC":
        df = load_cic_data()
    elif DATASET == "EDGE":
        df = load_edge_data()
    else:
        raise ValueError("DATASET must be either 'CIC' or 'EDGE'")

    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    df = df.drop_duplicates().reset_index(drop=True)    

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)

    print("Dataset:", DATASET)
    print("Feature count:", X.shape[1])
    print("Total rows:", X.shape[0])

    for c in np.unique(y):
        print(f"Class {c} rows:", (y == c).sum())

    return X, y


def split_clients(X, y, num_clients=NUM_CLIENTS):
    clients = []
    rng = np.random.default_rng(42)

    classes = np.unique(y)
    class_indices = {}

    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        class_indices[c] = np.array_split(idx, num_clients)

    for i in range(num_clients):
        selected = []

        for c in classes:
            chunk = class_indices[c][i]

            if i % 2 == 0:
                keep_ratio = 1.0 if c in [0, 1] else 0.35
            else:
                keep_ratio = 1.0 if c in [2, 3] else 0.35

            keep_n = max(1, int(len(chunk) * keep_ratio))
            selected.append(chunk[:keep_n])

        idx = np.concatenate(selected)
        rng.shuffle(idx)

        X_client = X[idx]
        y_client = y[idx]

        print(f"Client {i}: ", end="")
        for c in classes:
            print(f"class{c}={(y_client == c).sum()} ", end="")
        print()

        X_train, X_test, y_train, y_test = train_test_split(
            X_client,
            y_client,
            test_size=0.2,
            random_state=42,
            stratify=y_client,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            ),
            batch_size=32,
            shuffle=True,
        )

        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long),
            ),
            batch_size=32,
            shuffle=False,
        )

        clients.append((train_loader, test_loader))

    return clients