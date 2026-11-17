import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # rutas relativas a la carpeta de este script

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix, matthews_corrcoef
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SPLITS = 5
EPOCHS = 100
LR = 1e-3

# --------------------------------------------------
# MLP
# --------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------
# MÉTRICAS
# --------------------------------------------------
def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    return {
        "acc": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "auc": roc_auc_score(y_true, probs),
        "precision": precision_score(y_true, preds, zero_division=0),
        "sens": tp / (tp + fn) if (tp + fn) else 0,
        "spec": tn / (tn + fp) if (tn + fp) else 0,
        "mcc": matthews_corrcoef(y_true, preds)
    }

def find_best_threshold(y_true, probs):
    thresholds = np.linspace(0.1, 0.9, 100)
    f1s = [f1_score(y_true, (probs > t).astype(int)) for t in thresholds]
    return thresholds[np.argmax(f1s)]

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("resultados/representation_train_cv.csv")

y = df["mRS_bin"].values
X = df.drop(columns=["mRS_bin"]).values.astype(np.float32)

print("Shape:", X.shape)

# --------------------------------------------------
# CV TRAINING
# --------------------------------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

all_probs = []
all_y = []
results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n=== Fold {fold+1} ===")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    X_train = torch.tensor(X_train).to(DEVICE)
    X_val   = torch.tensor(X_val).to(DEVICE)

    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1).to(DEVICE)
    y_val   = torch.tensor(y_val,   dtype=torch.float32).view(-1,1).to(DEVICE)

    model = MLP(input_dim=X.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # entrenamiento
    for epoch in range(EPOCHS):
        model.train()

        logits = model(X_train)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validación
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    thr = find_best_threshold(y_val.cpu().numpy(), probs)
    metrics = compute_metrics(y_val.cpu().numpy(), probs, thr)

    results.append(metrics)
    all_probs.extend(probs)
    all_y.extend(y_val.cpu().numpy())

    print(metrics)

# --------------------------------------------------
# RESULTADOS CV
# --------------------------------------------------
print("\n==== RESULTADOS CV ====")
for k in results[0].keys():
    vals = [r[k] for r in results]
    print(f"{k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

thr_global = find_best_threshold(np.array(all_y), np.array(all_probs))
print("Threshold global:", thr_global)

# --------------------------------------------------
# VALIDACIÓN EXTERNA
# --------------------------------------------------
df_ext = pd.read_csv("resultados/representation_val_externa.csv")

y_ext = df_ext["mRS_bin"].values
X_ext = df_ext.drop(columns=["mRS_bin"]).values.astype(np.float32)

X_ext = torch.tensor(X_ext).to(DEVICE)

# usar último modelo (simple baseline)
model.eval()
with torch.no_grad():
    probs_ext = torch.sigmoid(model(X_ext)).cpu().numpy().flatten()

metrics_ext = compute_metrics(y_ext, probs_ext, thr_global)

print("\n==== VALIDACIÓN EXTERNA ====")
for k, v in metrics_ext.items():
    print(k, ":", v)