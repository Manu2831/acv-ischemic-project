


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # rutas relativas a la carpeta de este script

import os
import json
import pathlib
tabpfn_cfg_dir  = pathlib.Path.home() / ".tabpfn"
tabpfn_cfg_dir.mkdir(exist_ok=True)
tabpfn_cfg_file = tabpfn_cfg_dir / "config.json"
if not tabpfn_cfg_file.exists():
    with open(tabpfn_cfg_file, "w") as f:
        json.dump({"terms_accepted": True}, f)
os.environ["TABPFN_ACCEPT_TERMS"] = "1"
os.environ["TABPFN_TOKEN"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiMTZiZWFkMDctMWRkNy00ZjU1LWE4YTUtN2M2NmRlZDU1OWRkIiwiZXhwIjoxODA3MjQ1MzM1fQ.a57geeP_IfY3M6yUDxW6iqPS0tPB6L3u-LB49pCRdgA"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # ← primera línea absoluta
os.environ["TORCH_USE_CUDA_DSA"]   = "1"
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    brier_score_loss, roc_curve,
    confusion_matrix, matthews_corrcoef,
    classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG — todos los hiperparámetros en un solo lugar
# ---------------------------------------------------------------------------

device          = "cuda" if torch.cuda.is_available() else "cpu"
model_name      = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MAX_LEN         = 256
N_SPLITS        = 5
EPOCHS          = 100
BATCH_SIZE      = 16        # mini-batches para mejor generalización
PATIENCE        = 5        # early stopping: epochs sin mejora antes de parar
LR_BERT         = 1e-5      # lr para BERT (solo si FINETUNE_BERT=True)
LR_MODEL        = 5e-4      # lr para el resto del modelo
POS_WEIGHT_MULT = 1.0       # multiplicador sobre el ratio natural (1.0 = sin cambio)
FINETUNE_BERT   = False     # True = fine-tune layer 11 | False = BERT congelado
RANDOM_SEED     = 42
FEATURES        = ["Edad", "NIHSS ingreso", "Fibrilación Auricular", "Hipertensión"]
TARGET          = "mRS egreso"
TABPFN_EMB_DIM = None 

print(f"Device        : {device}")
print(f"Fine-tune BERT: {FINETUNE_BERT}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Patience      : {PATIENCE}")
print(f"pos_weight x  : {POS_WEIGHT_MULT}")

# ---------------------------------------------------------------------------
# 1. CARGA DE TEXTOS
# ---------------------------------------------------------------------------

def load_clinical_texts(base_path):
    records = []
    for patient in os.listdir(base_path):
        fp = os.path.join(base_path, patient, "ses-01",
            "pretreatment_clinical_imaging_summary.txt")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                records.append({"Paciente_id": patient, "text": f.read().strip()})
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# 2. BERT
# ---------------------------------------------------------------------------

print(f"\nCargando {model_name}...")
tokenizer  = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name).to(device)

if FINETUNE_BERT:
    # Solo última capa entrenable
    for name, param in bert_model.named_parameters():
        param.requires_grad = "encoder.layer.11" in name
    print(f"Parámetros entrenables BERT: "
        f"{sum(p.numel() for p in bert_model.parameters() if p.requires_grad):,}")
else:
    # Completamente congelado — más estable con datasets pequeños
    for param in bert_model.parameters():
        param.requires_grad = False
    print("BERT completamente congelado.")

bert_original_state = copy.deepcopy(bert_model.state_dict())

# ---------------------------------------------------------------------------
# 3. ARQUITECTURA TABULAR
# ---------------------------------------------------------------------------

def get_tabpfn_embeddings(X_train_np, y_train_np, X_query_np, device_str="cpu"):
    clf = TabPFNClassifier(
        device=device_str,
        ignore_pretraining_limits=True,
    )
    clf.fit(X_train_np, y_train_np)
    
    emb = clf.get_embeddings(X_query_np)

    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()

    if emb.ndim == 3:
        # Shape original: (n_ensemble, n_samples, emb_dim)
        # Necesitamos:    (n_samples, n_ensemble * emb_dim)
        emb = emb.transpose(1, 0, 2)          # → (n_samples, n_ensemble, emb_dim)
        emb = emb.reshape(emb.shape[0], -1)   # → (n_samples, n_ensemble * emb_dim)

    # Verificación
    assert emb.shape[0] == len(X_query_np), \
        f"Error: esperaba {len(X_query_np)} muestras, got {emb.shape[0]}"

    return torch.tensor(emb, dtype=torch.float32), clf

# =============================================
#    Funcion Perdida
# =============================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
# ---------------------------------------------------------------------------
# 4. MODELO MULTIMODAL
# ---------------------------------------------------------------------------

class MultimodalModel(nn.Module):
    def __init__(self, tab_emb_dim, text_dim=768):
        super().__init__()

        self.cross_attn = CrossAttention(
            dim_q=tab_emb_dim,
            dim_kv=text_dim
        )

        fusion_dim = tab_emb_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, tab_emb, text_tokens):

        # Cross-attention
        text_context = self.cross_attn(tab_emb, text_tokens)

        # Fusion
        fusion = torch.cat([tab_emb, text_context], dim=1)

        return self.classifier(fusion)

# ---------------------------------------------------------------------------
# 5. MÉTRICAS CLÍNICAS
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
def find_best_threshold_pre(y_true, probs, min_precision=0.5):
    thresholds = np.linspace(0.1, 0.9, 100)
    best_t = 0.5
    best_rec = 0

    for t in thresholds:
        preds = (probs > t).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec  = recall_score(y_true, preds)

        if prec >= min_precision and rec > best_rec:
            best_rec = rec
            best_t = t

    return best_t
def find_best_threshold_f1(y_true, probs):
    thresholds = np.linspace(0.1, 0.9, 100)
    f1s = [f1_score(y_true, (probs > t).astype(int)) for t in thresholds]
    return thresholds[np.argmax(f1s)]
# ---------------------------------------------------------------------------
# 6. UTILIDAD — mini-batches
# ---------------------------------------------------------------------------

def make_batches(n, batch_size, seed=None):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [idx[i: i + batch_size] for i in range(0, n, batch_size)]

class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=4):
        super().__init__()

        self.query_proj = nn.Linear(dim_q, dim_q)   # 🔥 NUEVO

        self.key_proj   = nn.Linear(dim_kv, dim_q)
        self.value_proj = nn.Linear(dim_kv, dim_q)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim_q,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, tab_emb, text_tokens):

        Q = self.query_proj(tab_emb).unsqueeze(1)   # 🔥 FIX
        K = self.key_proj(text_tokens)
        V = self.value_proj(text_tokens)

        attn_output, _ = self.attn(Q, K, V)

        return tab_emb + attn_output.squeeze(1)

def precompute_bert_embeddings(input_ids_list, attention_mask_list,
        token_type_ids_list=None):
    bert_model.eval()
    all_emb = []
    batch_size = 16
    n = input_ids_list.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            enc = {"input_ids":      input_ids_list[i:i+batch_size],
                "attention_mask": attention_mask_list[i:i+batch_size]}
            if token_type_ids_list is not None:
                enc["token_type_ids"] = token_type_ids_list[i:i+batch_size]
            out = bert_model(**enc)
            all_emb.append(out.last_hidden_state.cpu())
    return torch.cat(all_emb, dim=0)

def get_fold_tokens(idx):
    return {k: v[idx].to(device) for k, v in all_tokens.items()}
# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    df = pd.read_csv("../datos/df_train_acv.csv")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna()
    df["mRS_bin"] = (df[TARGET] > 2).astype(int)
    
    text_val_df = load_clinical_texts("../../acv_split/val/rawdata")
    df_val   = pd.read_csv("../datos/df_val_acv.csv")
    df_val[TARGET] = pd.to_numeric(df_val[TARGET], errors="coerce")
    df_val = df_val.dropna()
    df_val["mRS_bin"] = (df_val[TARGET] > 2).astype(int)

    df_val = df_val.merge(text_val_df, on="Paciente_id")

    text_df = load_clinical_texts("../../acv_split/train/rawdata")
    df      = df.merge(text_df, on="Paciente_id")
    
    
    
    print(f"\nPacientes finales: {len(df)}")
    print(f"Distribución mRS_bin:\n{df['mRS_bin'].value_counts()}")
    ratio = (df["mRS_bin"] == 0).sum() / (df["mRS_bin"] == 1).sum()
    print(f"Ratio desbalance: {ratio:.2f}:1")

    X_tab = df[FEATURES].values.astype(np.float32)
    y     = df["mRS_bin"].values
    texts = df["text"].tolist()
    # Tokenización global (una sola vez)
    print("\nTokenizando textos...")
    all_tokens = tokenizer(texts, padding=True, truncation=True,
        max_length=MAX_LEN, return_tensors="pt")
    tokens_val = tokenizer(
    df_val["text"].tolist(),
    padding=True,
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
    )
    # ============================================================
    # PRECOMPUTE VAL EXTERNO (una sola vez)
    # ============================================================

    X_val_ext_tab = df_val[FEATURES].values.astype(np.float32)
    y_val_ext     = df_val["mRS_bin"].values

    tokens_val_ext = {k: v.to(device) for k, v in tokens_val.items()}

    emb_val_fixed = precompute_bert_embeddings(
        tokens_val_ext["input_ids"],
        tokens_val_ext["attention_mask"],
        tokens_val_ext.get("token_type_ids", None)
    ).to(device)

    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    results    = []
    all_f1     = []
    all_auc    = []
    all_tr_loss  = []
    all_val_loss = []
    all_probs  = []
    all_y_val  = []
    all_epochs_run = []
    saved_models = []
    saved_scalers = []
    saved_tabpfn_clfs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tab, y)):
        print(f"\n========== Fold {fold+1}/{N_SPLITS} ==========")

        # Reinicio de BERT
        if FINETUNE_BERT:
            bert_model.load_state_dict(copy.deepcopy(bert_original_state))

        # Split tabular
        X_train, X_val = X_tab[train_idx], X_tab[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        
        tab_emb_tr, tabpfn_clf = get_tabpfn_embeddings(
        X_train, y_train, X_train, device_str="cuda"
        )
        tab_emb_vl, _ = get_tabpfn_embeddings(
            X_train, y_train, X_val, device_str="cuda"
        )
        X_val_ext_scaled = scaler.transform(X_val_ext_tab)
        tab_emb_val_ext, _ = get_tabpfn_embeddings(
            X_train, y_train, X_val_ext_scaled, device_str="cuda"
        )
        if TABPFN_EMB_DIM is None:
            TABPFN_EMB_DIM = tab_emb_tr.shape[1]
            print(f"  TabPFN emb_dim detectado: {TABPFN_EMB_DIM}")

        tab_emb_tr      = tab_emb_tr.to(device)
        tab_emb_vl      = tab_emb_vl.to(device)
        tab_emb_val_ext = tab_emb_val_ext.to(device)
        saved_tabpfn_clfs.append(tabpfn_clf)

        y_tr_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        y_vl_t = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1).to(device)

        ratio_fold = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        pos_weight = torch.tensor(
            [ratio_fold * POS_WEIGHT_MULT], dtype=torch.float32).to(device)

        tokens_tr = get_fold_tokens(train_idx)
        tokens_vl = get_fold_tokens(val_idx)

        # Si BERT está congelado, pre-calculamos embeddings una sola vez
        if not FINETUNE_BERT:
            emb_tr_fixed = precompute_bert_embeddings(
                tokens_tr["input_ids"], tokens_tr["attention_mask"],
                tokens_tr.get("token_type_ids")).to(device)
            emb_vl_fixed = precompute_bert_embeddings(
                tokens_vl["input_ids"], tokens_vl["attention_mask"],
                tokens_vl.get("token_type_ids")).to(device)

        model = MultimodalModel(tab_emb_dim=TABPFN_EMB_DIM).to(device)
        criterion = FocalLoss(alpha=0.6, gamma=1.5)
        
        if FINETUNE_BERT:
            optimizer = torch.optim.AdamW([
                {"params": filter(lambda p: p.requires_grad, bert_model.parameters()),
                "lr": LR_BERT},
                {"params": model.parameters(), "lr": LR_MODEL}
            ], weight_decay=1e-3)
        else:
            optimizer = torch.optim.AdamW(model.parameters(),
                lr=LR_MODEL, weight_decay=5e-3)

        # Early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        fold_f1, fold_auc, fold_tr_loss, fold_val_loss = [], [], [], []
        
        #print(f"  tab_emb_tr : {tab_emb_tr.shape}  device={tab_emb_tr.device}")
        #print(f"  emb_tr_fixed: {emb_tr_fixed.shape} device={emb_tr_fixed.device}")
        #print(f"  y_tr_t     : {y_tr_t.shape}  device={y_tr_t.device}")
        #print(f"  TABPFN_EMB_DIM={TABPFN_EMB_DIM}")

        for epoch in range(EPOCHS):

            # ── TRAIN (mini-batches) ────────────────────────────────────────
            model.train()
            if FINETUNE_BERT:
                bert_model.train()

            epoch_loss = 0.0
            batches = make_batches(len(train_idx), BATCH_SIZE, seed=epoch)

            for batch_idx in batches:
                optimizer.zero_grad()

                text_tokens_b = emb_tr_fixed[batch_idx]   # (B, L, 768)
                out_b = model(tab_emb_tr[batch_idx], text_tokens_b)
                loss_b = criterion(out_b,
                    y_tr_t[batch_idx])
                loss_b.backward()
                optimizer.step()
                epoch_loss += loss_b.item()

            avg_train_loss = epoch_loss / len(batches)
            fold_tr_loss.append(avg_train_loss)

            # ── VALIDACIÓN ─────────────────────────────────────────────────
            model.eval()
            if FINETUNE_BERT:
                bert_model.eval()

            with torch.no_grad():
                text_tokens_vl = emb_vl_fixed   # (N_val, L, 768)
                val_out = model(tab_emb_vl, text_tokens_vl)
                val_loss = criterion(val_out, y_vl_t)
                probs    = torch.sigmoid(val_out).cpu().numpy().flatten()

            fold_f1.append(f1_score(y_val, (probs > 0.5).astype(int), zero_division=0))
            fold_auc.append(roc_auc_score(y_val, probs)
                            if len(np.unique(y_val)) > 1 else 0.5)
            fold_val_loss.append(val_loss.item())

            # ── EARLY STOPPING ─────────────────────────────────────────────
            if val_loss.item() < best_val_loss:
                best_val_loss    = val_loss.item()
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stop en epoch {epoch+1}")
                    break

        all_epochs_run.append(epoch + 1)

        # Restauramos el mejor modelo del fold
        model.load_state_dict(best_model_state)
        saved_models.append(copy.deepcopy(model.state_dict()))
        saved_scalers.append(copy.deepcopy(scaler))
        model.eval()
        if FINETUNE_BERT:
            bert_model.eval()

        with torch.no_grad():
            text_tokens_vl = emb_vl_fixed
            final_out = model(tab_emb_vl, text_tokens_vl)
            final_probs = torch.sigmoid(final_out).cpu().numpy().flatten()

        best_thr = find_best_threshold_f1(y_val, final_probs)
        fold_metrics = compute_metrics(y_val, final_probs, threshold=best_thr)
        fold_metrics["thresh_used"] = best_thr
        results.append(fold_metrics)
        all_f1.append(fold_f1)
        all_auc.append(fold_auc)
        all_tr_loss.append(fold_tr_loss)
        all_val_loss.append(fold_val_loss)
        all_probs.extend(final_probs.tolist())
        all_y_val.extend(y_val.tolist())

        print(f"  Epochs corridos : {epoch+1} ")
        print(f"  Acc={fold_metrics['acc']:.3f} | F1={fold_metrics['f1']:.3f} | "
            f"AUC={fold_metrics['auc']:.3f}")
        print(f"  Sens={fold_metrics['sens']:.3f} | Spec={fold_metrics['spec']:.3f} | "
            f"PPV={fold_metrics['ppv']:.3f} | NPV={fold_metrics['npv']:.3f}")
        print(f"  MCC={fold_metrics['mcc']:.3f} | Brier={fold_metrics['brier']:.3f}")
        print(f"  Matriz: TP={fold_metrics['tp']} TN={fold_metrics['tn']} "
            f"FP={fold_metrics['fp']} FN={fold_metrics['fn']}")

    # ---------------------------------------------------------------------------
    # RESULTADOS FINALES
    # ---------------------------------------------------------------------------

    print(f"\n{'='*55}")
    print(f"  PubMedBERT + TabPFN v2.6")
    print(f"  BERT congelado={not FINETUNE_BERT} | "
        f"pos_weight x{POS_WEIGHT_MULT} | patience={PATIENCE}")
    print(f"  Epochs promedio: {np.mean(all_epochs_run):.1f} ± {np.std(all_epochs_run):.1f}")
    print(f"{'='*55}")

    metricas_reporte = [
        ("ACC",   "acc"),   ("F1",    "f1"),   ("AUC",  "auc"),
        ("SENS",  "sens"),  ("SPEC",  "spec"),
        ("PPV",   "ppv"),   ("NPV",   "npv"),
        ("MCC",   "mcc"),   ("BRIER", "brier"),
    ]

    for label, key in metricas_reporte:
        vals = [r[key] for r in results]
        print(f"  {label:6s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    thr_global = find_best_threshold_f1(all_y_val, np.array(all_probs))
    print(f"\n  Umbral  usado: {thr_global}")

    preds_global = (np.array(all_probs) > thr_global).astype(int)
    print(f"\n  Classification report global (umbral={thr_global:.3f}):")
    print(classification_report(all_y_val, preds_global,
                                target_names=["buen outcome (0)", "mal outcome (1)"]))
    #======================
    #VALIDACIÓN EXTERNA
    #==========================
    print("\n" + "="*60)
    print("        VALIDACIÓN EXTERNA (ENSEMBLE)")
    print("="*60)

    probs_ensemble = []

    for state, scaler, tabpfn_clf in zip(saved_models, saved_scalers, saved_tabpfn_clfs):
        X_val_scaled = scaler.transform(X_val_ext_tab)

    # Re-extraer embeddings TabPFN del fold
        emb_raw = tabpfn_clf.get_embeddings(X_val_scaled)
        if isinstance(emb_raw, torch.Tensor):
            emb_raw = emb_raw.detach().cpu().numpy()
        if emb_raw.ndim == 3:
            emb_raw = emb_raw.transpose(1, 0, 2)
            emb_raw = emb_raw.reshape(emb_raw.shape[0], -1)
        emb_tab_ext = torch.tensor(emb_raw, dtype=torch.float32).to(device)
    # ─────────────────────────────────────────────────────────────────

        model_ext = MultimodalModel(tab_emb_dim=TABPFN_EMB_DIM).to(device)
        model_ext.load_state_dict(state)
        model_ext.eval()

        with torch.no_grad():
            out   = model_ext(emb_tab_ext, emb_val_fixed)
            probs = torch.sigmoid(out).cpu().numpy().flatten()

        probs_ensemble.append(probs)

    # Promedio ensemble
    probs_final = np.mean(probs_ensemble, axis=0)

    # Métricas
    metrics_ext = compute_metrics(y_val_ext, probs_final, threshold=thr_global)

    print("\nResultados VALIDACIÓN EXTERNA:")
    for k, v in metrics_ext.items():
        print(f"{k}: {v}")
    # ---------------------------------------------------------------------------
    # GRÁFICAS
    # ---------------------------------------------------------------------------

    # Alineamos curvas al fold más corto (por early stopping)
    min_ep = min(len(x) for x in all_tr_loss)
    mean_tr  = np.mean([x[:min_ep] for x in all_tr_loss],  axis=0)
    mean_val = np.mean([x[:min_ep] for x in all_val_loss], axis=0)
    mean_f1  = np.mean([x[:min_ep] for x in all_f1],       axis=0)
    mean_auc = np.mean([x[:min_ep] for x in all_auc],      axis=0)
    ep       = range(min_ep)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"PubMedBERT + TabPFN v2.6 "
                f"(BERT_frozen={not FINETUNE_BERT}, pw_mult={POS_WEIGHT_MULT})",
                fontsize=12)

    axes[0].plot(ep, mean_tr,  label="Train loss")
    axes[0].plot(ep, mean_val, label="Val loss")
    axes[0].set(title="Loss (promedio K-Fold)", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(ep, mean_f1,  label="F1")
    axes[1].plot(ep, mean_auc, label="AUC")
    axes[1].set(title="Métricas val (promedio K-Fold)", xlabel="Epoch", ylabel="Score")
    axes[1].legend()

    fpr_g, tpr_g, _ = roc_curve(all_y_val, all_probs)
    auc_global = roc_auc_score(all_y_val, all_probs)
    axes[2].plot(fpr_g, tpr_g, label=f"ROC global (AUC={auc_global:.3f})")
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[2].set(title="Curva ROC global", xlabel="1 - Especificidad",
                ylabel="Sensibilidad")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("resultados/curvas_multimodal_v3.png", dpi=150)
    plt.show()