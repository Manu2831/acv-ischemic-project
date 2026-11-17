"""
tabpfn_embeddings.py
──────────────────────────────────────────────────────────────────────────────
TabPFN v2.6 — extracción de representación pre-clasificación.

Por qué los hooks directos no funcionan en v2.6
────────────────────────────────────────────────
TabPFN v2 NO usa un Linear(d_model → n_classes) clásico para clasificar.
Calcula probabilidades comparando las muestras de test contra el contexto
de entrenamiento mediante atención sobre las etiquetas — no hay una capa
"softmax head" estándar que hookear.

Solución: log-odds (siempre funciona, matemáticamente equivalente)
──────────────────────────────────────────────────────────────────
Para clasificación binaria, si TabPFN devuelve p = P(y=1|x):

    logit(p) = log(p / (1-p))

Este valor ES la diferencia de logits pre-softmax: logit_1 - logit_0.
Como los dos logits siempre colapsan a un solo grado de libertad bajo
softmax, el log-odds captura TODA la información discriminativa disponible.

                     predict_proba → p
                          ↓
    logit(p) = log(p/(1-p))  ← esto es nuestro "embedding pre-softmax"

Además se incluye `discover_layers()` que imprime todas las capas del
modelo con su output shape para que puedas identificar capas intermedias
más ricas si las necesitas para la fusión con imágenes.
──────────────────────────────────────────────────────────────────────────────
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # rutas relativas a la carpeta de este script


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    brier_score_loss, confusion_matrix,
    matthews_corrcoef, classification_report
)
from tabpfn import TabPFNClassifier
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

N_SPLITS    = 5
RANDOM_SEED = 42

FEATURES  = ["Edad", "NIHSS ingreso", "Diabetes", "Fibrilación Auricular", "Hipertensión"]
TARGET    = "mRS egreso"
N_CLASSES = 2


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class TabPFNEmbedder:


    def __init__(self, device: str = "cpu"):
        self.device = device
        self.clf    = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
        self._buffer: dict = {}
        self._hooks: list  = []
        self._n_train: int = 0

    # ── acceso al modelo interno ────────────────────────────────────────────

    def _internal_model(self) -> nn.Module:
        for attr in ("model_", "model", "base_model", "network"):
            m = getattr(self.clf, attr, None)
            if isinstance(m, nn.Module):
                return m
        raise RuntimeError(
            "No se encontró el modelo interno de TabPFN.\n"
            "Asegúrate de usar device='cuda' o 'cpu' (modo LOCAL).\n"
            "El modo CLOUD (API key remota) no permite hooks."
        )

    # ── API pública principal ───────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.clf.fit(X_train, y_train)
        self._n_train = X_train.shape[0]
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Probabilidades originales de TabPFN. Sin cambios."""
        return self.clf.predict_proba(X_test)

    def get_logodds(self, X_test: np.ndarray) -> np.ndarray:
        """
        Log-odds de la clase positiva: logit(P(y=1|x)) = log(p / (1-p)).

        Para clasificación binaria esto es matemáticamente equivalente
        a logit_1 - logit_0 (diferencia de logits pre-softmax).
        Captura TODA la información discriminativa en un escalar por muestra.

        Returns
        -------
        np.ndarray  shape (n_test, 1)
            Listo para concatenar con embeddings de imagen u otras features.
        """
        probs = self.predict_proba(X_test)[:, 1]
        probs = np.clip(probs, 1e-7, 1 - 1e-7)   # evitar log(0)
        logodds = np.log(probs / (1 - probs))
        return logodds.reshape(-1, 1)

    # ── descubridor de capas (diagnóstico) ─────────────────────────────────

    def discover_layers(
        self,
        X_test: np.ndarray,
        min_dim0: int = None,
        max_dim0: int = None,
        verbose: bool = True,
    ) -> dict:
        """
        Registra hooks en TODOS los submódulos, ejecuta un forward pass
        y devuelve un dict {layer_name: output_shape}.

        Útil para identificar capas internas con representaciones ricas
        (ej. embeddings del transformer de dimensión alta).

        Parameters
        ----------
        X_test    : muestras de validación (ya escaladas)
        min_dim0  : filtrar capas cuyo output.shape[0] >= min_dim0
        max_dim0  : filtrar capas cuyo output.shape[0] <= max_dim0
        verbose   : imprimir tabla de resultados

        Returns
        -------
        dict  {layer_name: tuple(shape)}
        """
        model   = self._internal_model()
        results = {}
        hooks   = []

        n_test = X_test.shape[0]

        def make_hook(name):
            def _h(module, inp, out):
                if isinstance(out, torch.Tensor):
                    results[name] = tuple(out.shape)
                elif isinstance(out, (tuple, list)):
                    results[name] = tuple(out[0].shape) if isinstance(out[0], torch.Tensor) else None
            return _h

        for name, module in model.named_modules():
            hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            _ = self.clf.predict_proba(X_test)

        for h in hooks:
            h.remove()

        # Filtrar y mostrar
        if verbose:
            n_train = self._n_train
            print(f"\n{'CAPA':<55} {'SHAPE'}")
            print("─" * 80)
            for name, shape in results.items():
                if shape is None:
                    continue
                # Resaltar capas cuyo dim-0 == n_test o n_train+n_test
                flag = ""
                if len(shape) >= 1:
                    d0 = shape[0]
                    if d0 == n_test:
                        flag = "  ← solo test"
                    elif d0 == n_train + n_test:
                        flag = "  ← train+test (tomar [-n_test:])"
                    elif d0 == n_train:
                        flag = "  ← solo train"

                skip = False
                if min_dim0 is not None and shape[0] < min_dim0:
                    skip = True
                if max_dim0 is not None and shape[0] > max_dim0:
                    skip = True
                if not skip:
                    print(f"  {name:<53} {str(shape):<25}{flag}")
            print()

        return results

    def get_embedding(
        self,
        X_test: np.ndarray,
        layer_name: str,
    ) -> np.ndarray:

        model  = self._internal_model()
        buffer = {}
        hook   = None
        n_test = X_test.shape[0]

        target = dict(model.named_modules()).get(layer_name)
        if target is None:
            raise ValueError(
                f"Capa '{layer_name}' no encontrada.\n"
                f"Usa discover_layers() para ver los nombres disponibles."
            )

        def _h(module, inp, out):
            if isinstance(out, torch.Tensor):
                buffer["emb"] = out.detach().cpu()
            elif isinstance(out, (tuple, list)) and isinstance(out[0], torch.Tensor):
                buffer["emb"] = out[0].detach().cpu()

        hook = target.register_forward_hook(_h)

        with torch.no_grad():
            _ = self.clf.predict_proba(X_test)

        hook.remove()

        raw = buffer.get("emb")
        if raw is None:
            raise RuntimeError(f"El hook en '{layer_name}' no capturó nada.")

        # Normalizar a (n_test, features)
        if raw.dim() == 3:
            # (seq, batch, d) → tomar últimos n_test
            raw = raw[-n_test:].squeeze(1)
        elif raw.dim() == 2 and raw.shape[0] > n_test:
            raw = raw[-n_test:]

        return raw.numpy()


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "acc":   accuracy_score(y_true, preds),
        "f1":    f1_score(y_true, preds, zero_division=0),
        "auc":   roc_auc_score(y_true, probs),
        "prec":  precision_score(y_true, preds, zero_division=0),
        "rec":   recall_score(y_true, preds, zero_division=0),
        "sens":  tp / (tp + fn) if (tp + fn) > 0 else 0,
        "spec":  tn / (tn + fp) if (tn + fp) > 0 else 0,
        "ppv":   tp / (tp + fp) if (tp + fp) > 0 else 0,
        "npv":   tn / (tn + fn) if (tn + fn) > 0 else 0,
        "mcc":   matthews_corrcoef(y_true, preds),
        "brier": brier_score_loss(y_true, probs),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

def find_best_threshold_f1(y_true, probs):
    thresholds = np.linspace(0.1, 0.9, 100)
    f1s = [f1_score(y_true, (probs > t).astype(int)) for t in thresholds]
    return thresholds[np.argmax(f1s)]

def find_best_threshold_clinical(y_true, probs, min_sens=0.8):
    thresholds = np.linspace(0.05, 0.95, 200)
    best_t, best_spec = 0.5, 0
    for t in thresholds:
        preds = (probs > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        if sens >= min_sens and spec > best_spec:
            best_spec = spec
            best_t    = t
    return best_t


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

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

    results      = []
    all_probs    = []
    all_y        = []
    all_logodds  = []   # ← representación pre-clasificación
    all_val_idx  = []

    # ── Descubrimiento de capas (solo Fold 1, una sola vez) ────────────────
    DISCOVER_LAYERS = True   # ← ponlo en False tras identificar la capa correcta

    # Nombre de la capa que quieres hookear (dejar None para usar log-odds)
    # Después de correr discover_layers(), pega aquí el nombre de la capa
    # que tenga shape compatible con (n_test, d) o (n_train+n_test, d).
    # Ejemplo: CUSTOM_LAYER = "transformer.layers.11.norm2"
    CUSTOM_LAYER = None

    # ── CV principal ───────────────────────────────────────────────────────
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n==== Fold {fold+1} ====")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        embedder = TabPFNEmbedder(device="cuda")
        embedder.fit(X_train, y_train)

        # Descubrir capas solo en el primer fold
        if DISCOVER_LAYERS and fold == 0:
            print("\n  [discover_layers] Mapeando arquitectura de TabPFN v2.6...")
            layer_map = embedder.discover_layers(
                X_val,
                min_dim0=len(val_idx) - 5,   # capas cuyo dim0 ≈ n_test
                max_dim0=len(X_train) + len(val_idx) + 5,
            )
            print("  Pega el nombre de la capa deseada en CUSTOM_LAYER "
                  "y pon DISCOVER_LAYERS=False para el run final.\n")

        # ── Representación pre-clasificación ───────────────────────────────
        if CUSTOM_LAYER is not None:
            # Usar la capa identificada por discover_layers
            representation = embedder.get_embedding(X_val, CUSTOM_LAYER)
            print(f"  Embedding '{CUSTOM_LAYER}' shape: {representation.shape}")
        else:
            # Log-odds: siempre funciona, matemáticamente equivalente
            representation = embedder.get_logodds(X_val)   # (n_val, 1)
            print(f"  Log-odds shape: {representation.shape}  "
                  f"(rango: [{representation.min():.2f}, {representation.max():.2f}])")

        all_logodds.append(representation)
        all_val_idx.extend(val_idx.tolist())

        # Probabilidades y métricas (igual que antes)
        probs    = embedder.predict_proba(X_val)[:, 1]
        best_thr = find_best_threshold_f1(y_val, probs)
        metrics  = compute_metrics(y_val, probs, threshold=best_thr)

        results.append(metrics)
        all_probs.extend(probs)
        all_y.extend(y_val)

        print(f"F1={metrics['f1']:.3f} | AUC={metrics['auc']:.3f}")
        print(f"Sens={metrics['sens']:.3f} | Spec={metrics['spec']:.3f}")
        print(f"PPV={metrics['ppv']:.3f}  | NPV={metrics['npv']:.3f}")

    # ── resultados finales ─────────────────────────────────────────────────
    print("\n" + "="*50)
    print("TABPFN BASELINE")
    print("="*50)

    for key in ["acc", "f1", "auc", "sens", "spec", "ppv", "npv", "mcc", "brier"]:
        vals = [r[key] for r in results]
        print(f"{key.upper():6s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    thr_global = find_best_threshold_f1(all_y, np.array(all_probs))
    print(f"\nThreshold global: {thr_global:.3f}")
    preds = (np.array(all_probs) > thr_global).astype(int)
    print("\nClassification report:")
    print(classification_report(all_y, preds))

    # ── exportar representación ────────────────────────────────────────────
    all_repr_np = np.vstack(all_logodds)
    order       = np.argsort(all_val_idx)
    all_repr_np = all_repr_np[order]
    all_y_ord   = np.array(all_y)[order]

    rep_cols  = ([f"emb_{i}" for i in range(all_repr_np.shape[1])]
                 if all_repr_np.shape[1] > 1 else ["logodds"])
    df_repr   = pd.DataFrame(all_repr_np, columns=rep_cols)
    df_repr.insert(0, "mRS_bin", all_y_ord)
    df_repr.to_csv("resultados/representation_train_cv.csv", index=False)
    print(f"\nRepresentación CV exportada → representation_train_cv.csv  "
          f"shape: {all_repr_np.shape}")


    # ══════════════════════════════════════════════════════════════════════
    # VALIDACIÓN EXTERNA
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "="*60)
    print("        VALIDACIÓN EXTERNA TABPFN")
    print("="*60)

    df_val_ext = pd.read_csv("../datos/df_val_acv.csv")
    df_val_ext[TARGET] = pd.to_numeric(df_val_ext[TARGET], errors="coerce")
    df_val_ext = df_val_ext.dropna()
    df_val_ext["mRS_bin"] = (df_val_ext[TARGET] > 2).astype(int)

    X_ext = df_val_ext[FEATURES].values.astype(np.float32)
    y_ext = df_val_ext["mRS_bin"].values

    probs_ensemble = []
    repr_ensemble  = []

    for fold, (train_idx, _) in enumerate(skf.split(X, y)):
        X_train = X[train_idx]
        y_train = y[train_idx]

        scaler       = StandardScaler()
        X_train      = scaler.fit_transform(X_train)
        X_ext_scaled = scaler.transform(X_ext)

        embedder = TabPFNEmbedder(device="cuda")
        embedder.fit(X_train, y_train)

        probs_ensemble.append(embedder.predict_proba(X_ext_scaled)[:, 1])

        if CUSTOM_LAYER is not None:
            repr_ensemble.append(embedder.get_embedding(X_ext_scaled, CUSTOM_LAYER))
        else:
            repr_ensemble.append(embedder.get_logodds(X_ext_scaled))

    probs_final = np.mean(probs_ensemble, axis=0)
    repr_final  = np.mean(repr_ensemble,  axis=0)   # ensemble (N_ext, d)

    metrics_ext = compute_metrics(y_ext, probs_final, threshold=thr_global)
    print("\nResultados VALIDACIÓN EXTERNA:")
    for k, v in metrics_ext.items():
        print(f"{k}: {v}")

    rep_cols_ext  = ([f"emb_{i}" for i in range(repr_final.shape[1])]
                    if repr_final.shape[1] > 1 else ["logodds"])
    df_repr_ext   = pd.DataFrame(repr_final, columns=rep_cols_ext)
    df_repr_ext.insert(0, "mRS_bin", y_ext)
    df_repr_ext.to_csv("resultados/representation_val_externa.csv", index=False)
    print(f"\nRepresentación externa → representation_val_externa.csv  "
          f"shape: {repr_final.shape}")