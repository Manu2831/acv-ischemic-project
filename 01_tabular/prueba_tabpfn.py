import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # rutas relativas a la carpeta de este script

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    brier_score_loss, roc_curve,
    confusion_matrix, matthews_corrcoef,
    classification_report
)
from tabpfn import TabPFNClassifier
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

N_SPLITS    = 5
RANDOM_SEED = 42

FEATURES = ["Edad", "NIHSS ingreso", "Diabetes","Fibrilación Auricular", "Hipertensión"]
TARGET   = "mRS egreso"

# ------------------------------------------------------------------
# MÉTRICAS
# ------------------------------------------------------------------

def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    return {
        "acc": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds, zero_division=0),
        "auc": roc_auc_score(y_true, probs),
        "prec": precision_score(y_true, preds, zero_division=0),
        "rec": recall_score(y_true, preds, zero_division=0),
        "sens": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "spec": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "mcc": matthews_corrcoef(y_true, preds),
        "brier": brier_score_loss(y_true, probs),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

def find_best_threshold_f1(y_true, probs):
    thresholds = np.linspace(0.1, 0.9, 100)
    f1s = [f1_score(y_true, (probs > t).astype(int)) for t in thresholds]
    return thresholds[np.argmax(f1s)]

def find_best_threshold_clinical(y_true, probs, min_sens=0.8):
    thresholds = np.linspace(0.05, 0.95, 200)
    best_t = 0.5
    best_spec = 0

    for t in thresholds:
        preds = (probs > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        if sens >= min_sens and spec > best_spec:
            best_spec = spec
            best_t = t

    return best_t

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":

    df = pd.read_csv("../datos/df_train_acv.csv")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna()
    df["mRS_bin"] = (df[TARGET] > 2).astype(int)

    print(f"\nPacientes: {len(df)}")
    print(df["mRS_bin"].value_counts())

    X = df[FEATURES].values.astype(np.float32)
    y = df["mRS_bin"].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    results = []
    all_probs = []
    all_y = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n==== Fold {fold+1} ====")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Escalado (aunque TabPFN no siempre lo necesita, ayuda estabilidad)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        # Modelo TabPFN

        clf = TabPFNClassifier(device="cuda", ignore_pretraining_limits=True)
        clf.fit(X_train, y_train)
        

        probs = clf.predict_proba(X_val)[:, 1]

        best_thr = find_best_threshold_f1(y_val, probs)
        metrics = compute_metrics(y_val, probs, threshold=best_thr)

        results.append(metrics)
        all_probs.extend(probs)
        all_y.extend(y_val)

        print(f"F1={metrics['f1']:.3f} | AUC={metrics['auc']:.3f}")
        print(f"Sens={metrics['sens']:.3f} | Spec={metrics['spec']:.3f}")
        print(f"PPV={metrics['ppv']:.3f} | NPV={metrics['npv']:.3f}")

    # ------------------------------------------------------------------
    # RESULTADOS FINALES
    # ------------------------------------------------------------------

    print("\n" + "="*50)
    print("TABPFN BASELINE")
    print("="*50)

    for key in ["acc","f1","auc","sens","spec","ppv","npv","mcc","brier"]:
        vals = [r[key] for r in results]
        print(f"{key.upper():6s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    thr_global = find_best_threshold_f1(all_y, np.array(all_probs))

    print(f"\nThreshold global: {thr_global:.3f}")

    preds = (np.array(all_probs) > thr_global).astype(int)

    print("\nClassification report:")
    print(classification_report(all_y, preds))
    
    print("\n" + "="*60)
print("        VALIDACIÓN EXTERNA TABPFN")
print("="*60)

# cargar dataset externo
df_val = pd.read_csv("../datos/df_val_acv.csv")
df_val[TARGET] = pd.to_numeric(df_val[TARGET], errors="coerce")
df_val = df_val.dropna()
df_val["mRS_bin"] = (df_val[TARGET] > 2).astype(int)

X_ext = df_val[FEATURES].values.astype(np.float32)
y_ext = df_val["mRS_bin"].values

probs_ensemble = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

    # reconstruir train del fold
    X_train = X[train_idx]
    y_train = y[train_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_ext_scaled = scaler.transform(X_ext)

    clf = TabPFNClassifier(device="cuda", ignore_pretraining_limits=True)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_ext_scaled)[:, 1]
    probs_ensemble.append(probs)

# ensemble promedio
probs_final = np.mean(probs_ensemble, axis=0)

# usar threshold global aprendido antes
metrics_ext = compute_metrics(y_ext, probs_final, threshold=thr_global)

print("\nResultados VALIDACIÓN EXTERNA:")
for k, v in metrics_ext.items():
    print(f"{k}: {v}")