import json
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from data import load_and_prepare_data
from model import Net
from local_train import train_local

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.utils import shuffle


RESULTS_DIR = Path("federated_learning/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


X, y = load_and_prepare_data()

df = pd.DataFrame(X)
df["label"] = y

df = df.drop_duplicates()

y = df["label"].values
X = df.drop(columns=["label"]).values

print("Rows after removing duplicates:", len(y))

print("\n=== DATA CHECKS ===")
df_check = pd.DataFrame(X)
df_check["label"] = y

print("Feature count:", X.shape[1])
print("Total rows:", len(y))
print("Class 0 rows:", (y == 0).sum())
print("Class 1 rows:", (y == 1).sum())
print("Duplicate full rows:", df_check.duplicated().sum())
print("Duplicate feature rows:", pd.DataFrame(X).duplicated().sum())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\n=== SPLIT CHECK ===")
print("Train rows:", len(y_train))
print("Test rows:", len(y_test))
print("Train class 0:", (y_train == 0).sum())
print("Train class 1:", (y_train == 1).sum())
print("Test class 0:", (y_test == 0).sum())
print("Test class 1:", (y_test == 1).sum())


trainloader = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    ),
    batch_size=32,
    shuffle=True,
)

model = Net(input_size=X.shape[1])

train_local(model, trainloader, epochs=10, lr=0.005)


model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    outputs = model(X_test_tensor)
    _, preds = torch.max(outputs, 1)

y_pred = preds.numpy()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== CENTRALIZED MODEL RESULTS ===")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))


print("\n=== SHUFFLE-LABEL SANITY TEST ===")

y_train_shuffled = shuffle(y_train, random_state=42)

trainloader_bad = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_shuffled, dtype=torch.long),
    ),
    batch_size=32,
    shuffle=True,
)

bad_model = Net(input_size=X.shape[1])
train_local(bad_model, trainloader_bad, epochs=10, lr=0.005)

bad_model.eval()
with torch.no_grad():
    outputs_bad = bad_model(torch.tensor(X_test, dtype=torch.float32))
    _, preds_bad = torch.max(outputs_bad, 1)

shuffle_accuracy = accuracy_score(y_test, preds_bad.numpy())

print("Shuffle-label accuracy:", shuffle_accuracy)

if shuffle_accuracy > 0.70:
    print("WARNING: Shuffle-label accuracy is too high. Possible leakage or duplicate structure.")
else:
    print("Good sign: shuffle-label accuracy dropped close to random guessing.")


results = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "shuffle_label_accuracy": float(shuffle_accuracy),
}

with open(RESULTS_DIR / "experiment2_centralized_results.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to: {(RESULTS_DIR / 'experiment2_centralized_results.json').resolve()}")