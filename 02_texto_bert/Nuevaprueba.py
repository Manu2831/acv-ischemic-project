# ---------------------------------------------------------------------------
# PubMedBERT (congelado) + tabular directo + fusion simple
#
# Cambios principales respecto a cuda.py, dado que el dataset es muy chico
# (109 train / 24 val / 24 test):
#   - Se quita TabPFN del branch tabular: expandía 5 features a ~1536-dim
#     y el embedding de train se calculaba con el propio X_train en el
#     contexto (fuga de información). Ahora el branch tabular es un encoder
#     lineal pequeño sobre las 5 features escaladas.
#   - Se quita el cross-attention bidireccional (millones de parametros)
#     y se reemplaza por fusion por concatenacion + dropout fuerte.
#   - BERT se usa con mean-pooling (mejor que el CLS crudo para un encoder
#     que nunca se fine-tuneo) y se precalcula UNA sola vez (esta congelado,
#     no cambia por fold).
#   - Se quita FINETUNE_BERT: esa rama estaba rota (usaba una variable
#     que solo existe si FINETUNE_BERT=False) y ademas fine-tunear 110M
#     parametros con 87 ejemplos no es viable.
#   - pos_weight se calcula por fold y ahora si se usa (antes se calculaba
#     y se descartaba).
#   - Umbral de decision: antes se elegia maximizando F1 global, lo que con
#     clase minoritaria (~1:3.7) empuja el umbral hacia abajo y genera falsos
#     positivos a proposito para subir el recall. Ahora el umbral operativo
#     por defecto es el de Youden (maximiza sens+spec-1), y se reportan
#     tambien el de F1 y el de precision>=0.8 solo como referencia.
#   - df_test_acv.csv nunca se evaluaba porque le falta la columna
#     "Diabetes" (tiene "Sexo" en su lugar) -> KeyError si se intentaba.
#     Se recupera "Diabetes" desde ../dataset_report.csv (esta disponible
#     para los 24 pacientes de test) y se agrega evaluacion final en test.
#   - Se agrega bootstrap CI (95%) en val/test dado que n=24 hace que las
#     metricas puntuales sean muy ruidosas (un solo caso mueve sens ~20pts).
#   - Se agrega un baseline SOLO TABULAR (regresión logística sobre las 5
#     features) al final, con las mismas folds/umbral/pos_weight que el
#     modelo multimodal. En pruebas locales este baseline tuvo MEJOR AUC y
#     MENOS falsos positivos (mayor especificidad) que el multimodal en val
#     y test -> con n=109 el texto vía BERT parece aportar más ruido que
#     señal. Queda como comparación explícita para decidir con datos reales.
# ---------------------------------------------------------------------------


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # rutas relativas a la carpeta de este script

import os
import copy
import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    brier_score_loss, roc_curve, precision_recall_curve,
    confusion_matrix, matthews_corrcoef,
    classification_report
)
import warnings
warnings.filterwarnings("ignore")

load_dotenv(pathlib.Path(__file__).resolve().parents[1] / "license.env")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

device          = "cuda" if torch.cuda.is_available() else "cpu"
model_name      = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MAX_LEN         = 256
N_SPLITS        = 5
EPOCHS          = 100
BATCH_SIZE      = 16
PATIENCE        = 10
LR_MODEL        = 5e-4
WEIGHT_DECAY    = 1e-2       # regularizacion fuerte: pocos datos, hay que evitar overfit
TAB_PROJ_DIM    = 8          # 5 features -> 8: casi sin capacidad extra
TEXT_PROJ_DIM   = 32         # 768 (BERT) -> 32: "linear probe" sobre features congeladas
DROPOUT         = 0.5
RANDOM_SEED     = 42
FEATURES        = ["Edad", "NIHSS ingreso", "Diabetes", "Fibrilación Auricular", "Hipertensión"]
TARGET          = "mRS egreso"
N_BOOT          = 2000

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Device        : {device}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Patience      : {PATIENCE}")
print(f"tab_proj_dim  : {TAB_PROJ_DIM} | text_proj_dim: {TEXT_PROJ_DIM} | dropout: {DROPOUT}")

# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS (tabular + texto), con fix de la columna faltante en test
# ---------------------------------------------------------------------------

def load_clinical_texts(base_path):
    records = []
    for patient in os.listdir(base_path):
        fp = os.path.join(base_path, patient, "ses-01",
            "pretreatment_clinical_summary.txt")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                records.append({"Paciente_id": patient, "text": f.read().strip()})
    return pd.DataFrame(records)


def load_split(csv_path, rawdata_path, report_path="../../dataset_report.csv"):
    df = pd.read_csv(csv_path)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df["mRS_bin"] = (df[TARGET] > 2).astype(int)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        report = pd.read_csv(report_path)
        lookup = report[["patient"] + [c.lower() for c in missing if c.lower() in report.columns]]
        rename_map = {"patient": "Paciente_id"}
        rename_map.update({c.lower(): c for c in missing})
        lookup = lookup.rename(columns=rename_map)
        df = df.merge(lookup, on="Paciente_id", how="left")
        still_missing = [f for f in FEATURES if f not in df.columns]
        if still_missing:
            raise ValueError(f"Faltan columnas incluso tras el merge con {report_path}: {still_missing}")

    df = df.dropna(subset=FEATURES)

    text_df = load_clinical_texts(rawdata_path)
    df = df.merge(text_df, on="Paciente_id")
    return df


df      = load_split("../datos/df_train_acv.csv", "../../acv_split/train/rawdata")
df_val  = load_split("../datos/df_val_acv.csv",   "../../acv_split/val/rawdata")
df_test = load_split("../datos/df_test_acv.csv",  "../../acv_split/test/rawdata")

print(f"\nPacientes -> train: {len(df)} | val: {len(df_val)} | test: {len(df_test)}")
print(f"Distribución mRS_bin (train):\n{df['mRS_bin'].value_counts()}")
ratio = (df["mRS_bin"] == 0).sum() / (df["mRS_bin"] == 1).sum()
print(f"Ratio desbalance (train): {ratio:.2f}:1")

# ---------------------------------------------------------------------------
# 2. BERT (congelado — con 87 ejemplos/fold, fine-tunear 110M params no es viable)
# ---------------------------------------------------------------------------

print(f"\nCargando {model_name}...")
tokenizer  = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name).to(device)
for param in bert_model.parameters():
    param.requires_grad = False
bert_model.eval()
print("BERT completamente congelado.")


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def precompute_bert_embeddings(input_ids, attention_mask, token_type_ids=None, batch_size=16):
    bert_model.eval()
    pooled_chunks = []
    n = input_ids.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            enc = {"input_ids": input_ids[i:i + batch_size].to(device),
                   "attention_mask": attention_mask[i:i + batch_size].to(device)}
            if token_type_ids is not None:
                enc["token_type_ids"] = token_type_ids[i:i + batch_size].to(device)
            out = bert_model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            pooled_chunks.append(pooled.cpu())
    return torch.cat(pooled_chunks, dim=0)


def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")


print("\nTokenizando y precalculando embeddings de BERT (una sola vez, esta congelado)...")
tokens_train = tokenize(df["text"].tolist())
tokens_val   = tokenize(df_val["text"].tolist())
tokens_test  = tokenize(df_test["text"].tolist())

emb_train_fixed = precompute_bert_embeddings(
    tokens_train["input_ids"], tokens_train["attention_mask"],
    tokens_train.get("token_type_ids"))
emb_val_fixed = precompute_bert_embeddings(
    tokens_val["input_ids"], tokens_val["attention_mask"],
    tokens_val.get("token_type_ids")).to(device)
emb_test_fixed = precompute_bert_embeddings(
    tokens_test["input_ids"], tokens_test["attention_mask"],
    tokens_test.get("token_type_ids")).to(device)

# ---------------------------------------------------------------------------
# 3. MODELO — fusion simple (concat), sin cross-attention
# ---------------------------------------------------------------------------

class MultimodalModel(nn.Module):
    def __init__(self, n_tab_features, text_dim=768,
                 tab_proj_dim=TAB_PROJ_DIM, text_proj_dim=TEXT_PROJ_DIM,
                 dropout=DROPOUT):
        super().__init__()
        self.tab_encoder = nn.Sequential(
            nn.Linear(n_tab_features, tab_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, text_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(tab_proj_dim + text_proj_dim, 1),
        )

    def forward(self, tab_feats, text_emb):
        tab_h  = self.tab_encoder(tab_feats)
        text_h = self.text_encoder(text_emb)
        fused  = torch.cat([tab_h, text_h], dim=1)
        return self.classifier(fused)


_n_params = sum(p.numel() for p in
                MultimodalModel(len(FEATURES)).parameters() if p.requires_grad)
print(f"\nParámetros entrenables del modelo: {_n_params:,} "
      f"(vs. varios millones con TabPFN-emb + cross-attention)")

# ---------------------------------------------------------------------------
# 4. MÉTRICAS Y UMBRALES
# ---------------------------------------------------------------------------

def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs > threshold).astype(int)
    cm    = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5

    return {
        "acc":        accuracy_score(y_true, preds),
        "f1":         f1_score(y_true, preds, zero_division=0),
        "auc":        auc,
        "prec":       precision_score(y_true, preds, zero_division=0),
        "rec":        recall_score(y_true, preds, zero_division=0),
        "sens":       sensitivity,
        "spec":       specificity,
        "ppv":        ppv,
        "npv":        npv,
        "mcc":        matthews_corrcoef(y_true, preds),
        "brier":      brier_score_loss(y_true, probs),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def find_best_threshold_f1(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_true, (probs > t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[np.argmax(f1s)])


def find_best_threshold_precision(y_true, probs, min_precision=0.8):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_rec = 0.5, -1
    for t in thresholds:
        preds = (probs > t).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec  = recall_score(y_true, preds, zero_division=0)
        if prec >= min_precision and rec > best_rec:
            best_rec, best_t = rec, t
    return float(best_t)


def find_best_threshold_youden(y_true, probs):
    fpr, tpr, thr = roc_curve(y_true, probs)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def bootstrap_ci(y_true, probs, threshold, n_boot=N_BOOT, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    probs  = np.asarray(probs)
    n = len(y_true)
    keys = ["auc", "sens", "spec", "ppv", "f1"]
    stats = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, pr = y_true[idx], probs[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = compute_metrics(yt, pr, threshold=threshold)
        for k in keys:
            stats[k].append(m[k])
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)
            for k, v in stats.items()}


def make_batches(n, batch_size, seed=None):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [idx[i: i + batch_size] for i in range(0, n, batch_size)]

# ---------------------------------------------------------------------------
# 5. MAIN — K-Fold CV en train, ensemble sobre val y test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    X_tab = df[FEATURES].values.astype(np.float32)
    y     = df["mRS_bin"].values

    X_val_tab  = df_val[FEATURES].values.astype(np.float32)
    y_val      = df_val["mRS_bin"].values
    X_test_tab = df_test[FEATURES].values.astype(np.float32)
    y_test     = df_test["mRS_bin"].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    results, all_f1, all_auc = [], [], []
    all_tr_loss, all_val_loss = [], []
    all_probs_oof, all_y_oof  = [], []
    all_epochs_run = []
    saved_models, saved_scalers = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tab, y)):
        print(f"\n========== Fold {fold + 1}/{N_SPLITS} ==========")

        X_train, X_val_fold = X_tab[train_idx], X_tab[val_idx]
        y_train, y_val_fold = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_s     = scaler.fit_transform(X_train)
        X_val_fold_s  = scaler.transform(X_val_fold)

        tab_tr_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
        tab_vl_t = torch.tensor(X_val_fold_s, dtype=torch.float32).to(device)

        y_tr_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        y_vl_t = torch.tensor(y_val_fold, dtype=torch.float32).view(-1, 1).to(device)

        emb_tr_fixed = emb_train_fixed[train_idx].to(device)
        emb_vl_fixed = emb_train_fixed[val_idx].to(device)

        ratio_fold = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        # sqrt del ratio en vez del ratio crudo: la inversa de frecuencia completa
        # (~3.7x) sobrecorrige con tan pocos positivos y dispara los falsos
        # positivos (el loss termina premiando predecir "mal outcome" casi siempre).
        pos_weight = torch.tensor([ratio_fold ** 0.5], dtype=torch.float32).to(device)

        model = MultimodalModel(n_tab_features=len(FEATURES)).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MODEL, weight_decay=WEIGHT_DECAY)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        fold_f1, fold_auc, fold_tr_loss, fold_val_loss = [], [], [], []

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            batches = make_batches(len(train_idx), BATCH_SIZE, seed=epoch)

            for batch_idx in batches:
                optimizer.zero_grad()
                out_b = model(tab_tr_t[batch_idx], emb_tr_fixed[batch_idx])
                loss_b = criterion(out_b, y_tr_t[batch_idx])
                loss_b.backward()
                optimizer.step()
                epoch_loss += loss_b.item()

            avg_train_loss = epoch_loss / len(batches)
            fold_tr_loss.append(avg_train_loss)

            model.eval()
            with torch.no_grad():
                val_out  = model(tab_vl_t, emb_vl_fixed)
                val_loss = criterion(val_out, y_vl_t)
                probs    = torch.sigmoid(val_out).cpu().numpy().flatten()

            fold_f1.append(f1_score(y_val_fold, (probs > 0.5).astype(int), zero_division=0))
            fold_auc.append(roc_auc_score(y_val_fold, probs)
                             if len(np.unique(y_val_fold)) > 1 else 0.5)
            fold_val_loss.append(val_loss.item())

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stop en epoch {epoch + 1}")
                    break

        all_epochs_run.append(epoch + 1)

        model.load_state_dict(best_model_state)
        saved_models.append(copy.deepcopy(model.state_dict()))
        saved_scalers.append(copy.deepcopy(scaler))
        model.eval()

        with torch.no_grad():
            final_out = model(tab_vl_t, emb_vl_fixed)
            final_probs = torch.sigmoid(final_out).cpu().numpy().flatten()

        best_thr = find_best_threshold_youden(y_val_fold, final_probs)
        fold_metrics = compute_metrics(y_val_fold, final_probs, threshold=best_thr)
        fold_metrics["thresh_used"] = best_thr
        results.append(fold_metrics)
        all_f1.append(fold_f1)
        all_auc.append(fold_auc)
        all_tr_loss.append(fold_tr_loss)
        all_val_loss.append(fold_val_loss)
        all_probs_oof.extend(final_probs.tolist())
        all_y_oof.extend(y_val_fold.tolist())

        print(f"  Epochs corridos : {epoch + 1}")
        print(f"  Acc={fold_metrics['acc']:.3f} | F1={fold_metrics['f1']:.3f} | "
              f"AUC={fold_metrics['auc']:.3f}")
        print(f"  Sens={fold_metrics['sens']:.3f} | Spec={fold_metrics['spec']:.3f} | "
              f"PPV={fold_metrics['ppv']:.3f} | NPV={fold_metrics['npv']:.3f}")
        print(f"  MCC={fold_metrics['mcc']:.3f} | Brier={fold_metrics['brier']:.3f}")
        print(f"  Matriz: TP={fold_metrics['tp']} TN={fold_metrics['tn']} "
              f"FP={fold_metrics['fp']} FN={fold_metrics['fn']}")

    # -------------------------------------------------------------------
    # RESULTADOS CV (out-of-fold)
    # -------------------------------------------------------------------

    print(f"\n{'=' * 55}")
    print("  PubMedBERT (frozen) + tabular directo + fusion simple")
    print(f"  Epochs promedio: {np.mean(all_epochs_run):.1f} ± {np.std(all_epochs_run):.1f}")
    print(f"{'=' * 55}")

    metricas_reporte = [
        ("ACC", "acc"), ("F1", "f1"), ("AUC", "auc"),
        ("SENS", "sens"), ("SPEC", "spec"),
        ("PPV", "ppv"), ("NPV", "npv"),
        ("MCC", "mcc"), ("BRIER", "brier"),
    ]
    for label, key in metricas_reporte:
        vals = [r[key] for r in results]
        print(f"  {label:6s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    all_y_oof = np.array(all_y_oof)
    all_probs_oof = np.array(all_probs_oof)

    thr_youden = find_best_threshold_youden(all_y_oof, all_probs_oof)
    thr_f1     = find_best_threshold_f1(all_y_oof, all_probs_oof)
    thr_prec   = find_best_threshold_precision(all_y_oof, all_probs_oof, min_precision=0.8)

    print(f"\n  Umbral Youden (operativo, balancea sens/spec) : {thr_youden:.3f}")
    print(f"  Umbral F1-óptimo (referencia, favorece recall): {thr_f1:.3f}")
    print(f"  Umbral precisión>=0.8 (referencia, anti-FP)   : {thr_prec:.3f}")

    preds_oof = (all_probs_oof > thr_youden).astype(int)
    print(f"\n  Classification report OOF (umbral Youden={thr_youden:.3f}):")
    print(classification_report(all_y_oof, preds_oof,
                                 target_names=["buen outcome (0)", "mal outcome (1)"]))

    # -------------------------------------------------------------------
    # VALIDACIÓN EXTERNA (ensemble de los N_SPLITS modelos) — val y test
    # -------------------------------------------------------------------

    def ensemble_predict(X_tab_ext, emb_ext_fixed):
        probs_ensemble = []
        for state, scaler in zip(saved_models, saved_scalers):
            X_scaled = scaler.transform(X_tab_ext)
            tab_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)

            model_ext = MultimodalModel(n_tab_features=len(FEATURES)).to(device)
            model_ext.load_state_dict(state)
            model_ext.eval()

            with torch.no_grad():
                out   = model_ext(tab_t, emb_ext_fixed)
                probs = torch.sigmoid(out).cpu().numpy().flatten()
            probs_ensemble.append(probs)
        return np.mean(probs_ensemble, axis=0)


    def report_external(name, y_true, probs, threshold):
        print("\n" + "=" * 60)
        print(f"        {name}  (n={len(y_true)})")
        print("=" * 60)
        for label, thr in [("Youden", thr_youden), ("F1-óptimo", thr_f1), ("Precisión>=0.8", thr_prec)]:
            m = compute_metrics(y_true, probs, threshold=thr)
            print(f"  [{label:14s} thr={thr:.3f}] Acc={m['acc']:.3f} F1={m['f1']:.3f} "
                  f"AUC={m['auc']:.3f} Sens={m['sens']:.3f} Spec={m['spec']:.3f} "
                  f"PPV={m['ppv']:.3f} NPV={m['npv']:.3f}  TP={m['tp']} FP={m['fp']} "
                  f"TN={m['tn']} FN={m['fn']}")

        ci = bootstrap_ci(y_true, probs, threshold, n_boot=N_BOOT)
        print(f"\n  IC 95% (bootstrap, n={len(y_true)} -> ¡ojo, muy amplio!) con umbral Youden:")
        for k, (lo, hi) in ci.items():
            print(f"    {k:5s}: [{lo:.3f}, {hi:.3f}]")


    probs_val_ext  = ensemble_predict(X_val_tab, emb_val_fixed)
    probs_test_ext = ensemble_predict(X_test_tab, emb_test_fixed)

    report_external("VALIDACIÓN EXTERNA (val)", y_val, probs_val_ext, thr_youden)
    report_external("VALIDACIÓN EXTERNA (test, held-out)", y_test, probs_test_ext, thr_youden)

    # -------------------------------------------------------------------
    # BASELINE DE REFERENCIA: SOLO TABULAR (sin texto)
    #
    # Mismas folds, mismo pos_weight suavizado, mismo criterio de umbral
    # (Youden sobre OOF) que el modelo multimodal, para que la comparación
    # sea justa. Sirve para saber si BERT realmente está aportando señal
    # con este tamaño de muestra, o si solo agrega varianza/ruido.
    # -------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("   BASELINE DE REFERENCIA: SOLO TABULAR (LogisticRegression)")
    print("=" * 60)

    tab_oof_probs = np.zeros(len(y))
    tab_models, tab_scalers = [], []

    for train_idx, val_idx in skf.split(X_tab, y):
        X_train, X_val_fold = X_tab[train_idx], X_tab[val_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_s    = scaler.fit_transform(X_train)
        X_val_fold_s = scaler.transform(X_val_fold)

        ratio_fold = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        clf = LogisticRegression(class_weight={0: 1.0, 1: ratio_fold ** 0.5}, max_iter=1000)
        clf.fit(X_train_s, y_train)

        tab_oof_probs[val_idx] = clf.predict_proba(X_val_fold_s)[:, 1]
        tab_models.append(clf)
        tab_scalers.append(scaler)

    tab_thr_youden = find_best_threshold_youden(y, tab_oof_probs)
    tab_oof_metrics = compute_metrics(y, tab_oof_probs, threshold=tab_thr_youden)
    print(f"  OOF (umbral Youden={tab_thr_youden:.3f}): AUC={tab_oof_metrics['auc']:.3f} "
          f"Sens={tab_oof_metrics['sens']:.3f} Spec={tab_oof_metrics['spec']:.3f} "
          f"PPV={tab_oof_metrics['ppv']:.3f} TP={tab_oof_metrics['tp']} FP={tab_oof_metrics['fp']} "
          f"TN={tab_oof_metrics['tn']} FN={tab_oof_metrics['fn']}")

    def tab_ensemble_predict(X_ext):
        probs = [clf.predict_proba(scaler.transform(X_ext))[:, 1]
                 for clf, scaler in zip(tab_models, tab_scalers)]
        return np.mean(probs, axis=0)

    tab_probs_val  = tab_ensemble_predict(X_val_tab)
    tab_probs_test = tab_ensemble_predict(X_test_tab)

    for name, y_true, probs in [("VAL", y_val, tab_probs_val), ("TEST", y_test, tab_probs_test)]:
        m = compute_metrics(y_true, probs, threshold=tab_thr_youden)
        print(f"  {name:5s} (n={len(y_true)}): AUC={m['auc']:.3f} Sens={m['sens']:.3f} "
              f"Spec={m['spec']:.3f} PPV={m['ppv']:.3f} TP={m['tp']} FP={m['fp']} "
              f"TN={m['tn']} FN={m['fn']}")

    print("\n  Compará estos números contra el modelo multimodal de arriba: si el")
    print("  baseline tabular iguala o supera la especificidad/AUC del multimodal,")
    print("  el texto no está justificando su complejidad con n=109.")

    # -------------------------------------------------------------------
    # GRÁFICAS
    # -------------------------------------------------------------------

    min_ep = min(len(x) for x in all_tr_loss)
    mean_tr  = np.mean([x[:min_ep] for x in all_tr_loss],  axis=0)
    mean_val = np.mean([x[:min_ep] for x in all_val_loss], axis=0)
    mean_f1  = np.mean([x[:min_ep] for x in all_f1],       axis=0)
    mean_auc = np.mean([x[:min_ep] for x in all_auc],      axis=0)
    ep       = range(min_ep)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("PubMedBERT (frozen) + tabular directo + fusion simple", fontsize=12)

    axes[0, 0].plot(ep, mean_tr,  label="Train loss")
    axes[0, 0].plot(ep, mean_val, label="Val loss")
    axes[0, 0].set(title="Loss (promedio K-Fold)", xlabel="Epoch", ylabel="Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(ep, mean_f1,  label="F1")
    axes[0, 1].plot(ep, mean_auc, label="AUC")
    axes[0, 1].set(title="Métricas val (promedio K-Fold)", xlabel="Epoch", ylabel="Score")
    axes[0, 1].legend()

    fpr_g, tpr_g, _ = roc_curve(all_y_oof, all_probs_oof)
    auc_global = roc_auc_score(all_y_oof, all_probs_oof)
    axes[1, 0].plot(fpr_g, tpr_g, label=f"ROC OOF (AUC={auc_global:.3f})")
    axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1, 0].set(title="Curva ROC (OOF)", xlabel="1 - Especificidad", ylabel="Sensibilidad")
    axes[1, 0].legend()

    prec_g, rec_g, _ = precision_recall_curve(all_y_oof, all_probs_oof)
    axes[1, 1].plot(rec_g, prec_g, label="PR OOF")
    axes[1, 1].axhline((all_y_oof == 1).mean(), color="k", ls="--", alpha=0.4, label="Baseline (prevalencia)")
    axes[1, 1].set(title="Curva Precisión-Recall (OOF)", xlabel="Recall", ylabel="Precisión")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("resultados/curvas_nuevaprueba.png", dpi=150)
    plt.show()
