"""
Entrena UN experimento: CV estratificada (5-fold) sobre train con early stopping
en el fold de validación, y ensemble de los modelos de cada fold sobre val y test.
Protocolo igual a Nuevaprueba.py: umbral de Youden sobre OOF, bootstrap IC95%.

Para correr todo, usá ejecutar_todo.py. Este archivo entrena UNA configuración.

Codificadores de imagen (--encoder):
  lesion : SwinUNETR preentrenado SEGMENTANDO la lesión, congelado (cache_lesion/, de lesion_encoder.py)
  dinov2 : DINOv2 2D preentrenado y congelado, corte por corte (cache_dinov2/, de extract_dinov2.py)
  vit3d  : ViT 3D desde cero sobre el volumen (cache/, de preprocess.py) -- solo como referencia

Ejemplos (desde text/):
  python 04_imagen/train.py --images adc
  python 04_imagen/train.py --images adc dwi --tabular --fusion cross_attn
  python 04_imagen/train.py --images adc dwi --tabular --encoder dinov2
  python 04_imagen/train.py --images dwi --tabular --control shuffle_image   (control: imagen de otro paciente)
  python 04_imagen/train.py --images adc --tabular --quick                     (prueba rápida)

Salida en 04_imagen/results/<grupo>/.../<experimento>/ (ver finalize_config):
metrics.json, predictions.csv, embeddings.npz (para fusionar texto después),
attention.npz (mapas espaciales), log.csv, config.json, modelos/fold*.pt
"""

import argparse
import copy
import json
import math
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from models import MultimodalNet
from rutas import (CACHE_DIR, DINO_ROOT, LESION_ROOT, REPORT_CSV, RESULTS_DIR, SPLIT_ROOT, TABLA_LIMPIA, TABULAR_DIR,
                   amp_dtype, nifti)

ENCODER_FOLDER = {"lesion": "03_SwinUNETR_lesion", "dinov2": "02_DINOv2_preentrenado",
                  "vit3d": "01_ViT3D_desde_cero"}
TOKEN_ENCODERS = ("dinov2", "lesion")
SPLITS = ("train", "val", "test")
FEATURES = ["Edad", "NIHSS ingreso", "Diabetes", "Fibrilación Auricular", "Hipertensión"]
TARGET = "mRS egreso"


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Imagen (ADC/DWI) + tabular para mRS dicotomizado")
    p.add_argument("--name", default=None, help="nombre del experimento (auto si se omite)")
    p.add_argument("--images", nargs="*", default=[], choices=["adc", "dwi"])
    p.add_argument("--tabular", action="store_true")
    p.add_argument("--fusion", default="concat", choices=list(MultimodalNet.FUSIONS))
    p.add_argument("--no_lesion_mask", action="store_true", help="no usar la máscara de lesión")
    p.add_argument("--control", default="none", choices=["none", "shuffle_image"])
    # arquitectura
    p.add_argument("--encoder", default="lesion", choices=list(ENCODER_FOLDER),
                   help="lesion: SwinUNETR preentrenado segmentando la lesión | dinov2 | vit3d (desde cero)")
    p.add_argument("--backbone", default="vit_small_patch14_reg4_dinov2.lvd142m", help="solo --encoder dinov2")
    p.add_argument("--lesion_model", default="swinunetr_fs24", help="solo --encoder lesion (carpeta en cache_lesion/)")
    p.add_argument("--tokenizer", default="conv", choices=["conv", "patch"], help="solo --encoder vit3d")
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--depth", type=int, default=None, help="bloques transformer de imagen (vit3d: 4, dinov2: 2)")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--tab_depth", type=int, default=1)
    p.add_argument("--cross_depth", type=int, default=1)
    p.add_argument("--drop", type=float, default=0.1)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--head_drop", type=float, default=0.3)
    # entrenamiento
    p.add_argument("--epochs", type=int, default=85)
    p.add_argument("--min_epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--aux_weight", type=float, default=0.3, help="peso de las pérdidas por modalidad (solo gated)")
    # aumentaciones (en GPU)
    p.add_argument("--aug_rot", type=float, default=10.0, help="grados, rotación en el plano axial")
    p.add_argument("--aug_scale", type=float, default=0.1)
    p.add_argument("--aug_shift", type=float, default=6.0, help="vóxeles en el plano")
    p.add_argument("--aug_intensity", type=float, default=0.1)
    p.add_argument("--aug_noise", type=float, default=0.03)
    p.add_argument("--lr_flip", action="store_true", help="flip izq-der (apagado: la lateralidad importa)")
    p.add_argument("--feat_noise", type=float, default=0.1, help="dinov2: ruido relativo sobre los tokens")
    p.add_argument("--token_drop", type=float, default=0.1, help="dinov2: fracción de tokens anulados al entrenar")
    # protocolo
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--no_save_models", action="store_true")
    p.add_argument("--quick", action="store_true", help="2 folds x 3 épocas, para verificar que corre")
    p.add_argument("--results_dir", default=str(RESULTS_DIR))
    return p


FUSION_FOLDER = {f: f"{i + 1}_{f}" for i, f in enumerate(MultimodalNet.FUSIONS)}


def finalize_config(cfg):
    """Fija cfg.name (id único dentro de su carpeta) y cfg.out_dir (carpeta ordenada por tipo):

    results/
      00_referencia_tabular/        regresion_logistica, tabular_transformer
      01_ViT3D_desde_cero/          \\  mismas subcarpetas en los tres:
      02_DINOv2_preentrenado/        |  01_solo_imagen/        ADC, DWI, ADC+DWI
      03_SwinUNETR_lesion/          /   02_imagen+tabular/     1_concat/ 2_film/ 3_cross_attn/ 4_gated/
                                        03_controles/          imagen barajada, sin máscara de lesión
      05_personalizados/            corridas con --name explícito
    """
    if getattr(cfg, "out_dir", None):
        return cfg
    if not cfg.images and not cfg.tabular:
        raise ValueError("Elegí al menos --images o --tabular")
    if cfg.depth is None:
        cfg.depth = 4 if cfg.encoder == "vit3d" else 2
    if cfg.quick:
        cfg.epochs, cfg.folds, cfg.min_epochs, cfg.patience, cfg.warmup, cfg.n_boot = 3, 2, 1, 2, 1, 200
        cfg.results_dir = str(Path(cfg.results_dir).with_name("results_quick"))

    mods = "+".join(m.upper() for m in cfg.images)
    custom = cfg.name is not None
    if not custom:
        name = "+".join(p for p in (mods, "TAB" if cfg.tabular else "") if p)
        if cfg.images and cfg.tabular:
            name += f"_{cfg.fusion}"
        if cfg.images and cfg.no_lesion_mask:
            name += "_sin_mascara"
        if cfg.control == "shuffle_image":
            name += "_imagen_barajada"
        cfg.name = name

    enc = Path(ENCODER_FOLDER[cfg.encoder])
    if custom:
        rel = Path("05_personalizados") / cfg.name
    elif not cfg.images:
        rel = Path("00_referencia_tabular") / "tabular_transformer"
    elif cfg.control != "none" or cfg.no_lesion_mask:
        rel = enc / "03_controles" / cfg.name
    elif not cfg.tabular:
        rel = enc / "01_solo_imagen" / mods
    else:
        rel = enc / "02_imagen+tabular" / FUSION_FOLDER[cfg.fusion] / mods
    cfg.out_dir = str(Path(cfg.results_dir) / rel)
    return cfg


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

_AVISADO = set()


def _avisar(msg):
    if msg not in _AVISADO:
        _AVISADO.add(msg)
        print(msg, flush=True)


def load_tabular():
    """Tabla clínica. Usa la tabla limpia (07_tabla_limpia/construir_tabla.py: valores corregidos con la nota de
    ingreso y sin los pacientes excluidos) si existe; con ACV_TABLA=original, o si falta, usa los df_*_acv.csv."""
    if os.environ.get("ACV_TABLA") != "original" and TABLA_LIMPIA.exists():
        df = pd.read_csv(TABLA_LIMPIA, encoding="utf-8-sig")
        n = len(df)
        df = df[df.excluido.isna() | (df.excluido.astype(str).str.strip() == "")]
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.dropna(subset=FEATURES + [TARGET])
        df["y"] = (df[TARGET] > 2).astype(int)
        _avisar(f"[tabular] tabla LIMPIA: {TABLA_LIMPIA} ({len(df)} de {n} pacientes, {int(df.y.sum())} eventos)")
        return df[["Paciente_id", "split", "y"] + FEATURES].reset_index(drop=True)
    _avisar(f"[tabular] tabla ORIGINAL (df_*_acv.csv en {TABULAR_DIR}): sin las correcciones de las notas de ingreso")
    report = pd.read_csv(REPORT_CSV)
    frames = []
    for split in SPLITS:
        df = pd.read_csv(TABULAR_DIR / f"df_{split}_acv.csv")
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:  # test no trae Diabetes: se recupera de dataset_report.csv
            lookup = report[["patient"] + [m.lower() for m in missing]]
            lookup = lookup.rename(columns={"patient": "Paciente_id", **{m.lower(): m for m in missing}})
            df = df.merge(lookup, on="Paciente_id", how="left")
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.dropna(subset=FEATURES + [TARGET])
        df["y"] = (df[TARGET] > 2).astype(int)
        df["split"] = split
        frames.append(df[["Paciente_id", "split", "y"] + FEATURES])
    return pd.concat(frames, ignore_index=True)


def with_images(df):
    """Mismos pacientes en todos los experimentos: los que tienen DWI y ADC originales."""
    ok = [all(nifti(SPLIT_ROOT / s / "rawdata" / p / "ses-01" / f"{m}.nii.gz").exists() for m in ("dwi", "adc"))
          for p, s in zip(df.Paciente_id, df.split)]
    return df[ok].reset_index(drop=True)


def load_images(pids, images, lesion_mask):
    chans = list(images) + ([f"{m}_lesion" for m in images] if lesion_mask else [])
    vols = []
    for pid in pids:
        d = np.load(CACHE_DIR / f"{pid}.npz")
        vols.append(np.stack([d[c].astype(np.float16) for c in chans]))
    return torch.from_numpy(np.stack(vols)), chans


def token_cache(cfg):
    if cfg.encoder == "dinov2":
        return DINO_ROOT / cfg.backbone.split(".")[0], f"python 04_imagen/extract_dinov2.py --backbone {cfg.backbone}"
    return LESION_ROOT / cfg.lesion_model, "python 04_imagen/lesion_encoder.py --etapa todo"


def load_tokens(pids, images, lesion_mask, cfg):
    """Tokens precalculados por paciente -> (N, T, C + n_aux), T = grid d*h*w.
    dinov2: un arreglo por modalidad (se concatenan). lesion: un arreglo por combinación
    de entrada ('adc', 'dwi', 'adc+dwi'), porque el encoder ve ADC y DWI juntos."""
    root, how = token_cache(cfg)
    if not (root / "meta.json").exists():
        raise FileNotFoundError(f"No existe {root}. Corré primero: {how}")
    meta = json.loads((root / "meta.json").read_text())
    joint = meta.get("joint_inputs", False)
    keys = ["+".join(images)] if joint else list(images)
    out = []
    for pid in pids:
        d = np.load(root / f"{pid}.npz")
        feats = [d[k].reshape(-1, d[k].shape[-1]) for k in keys]
        aux = [d[f"{m}_lesion"].reshape(-1, 1) for m in images] if lesion_mask else []
        out.append(np.concatenate(feats + aux, axis=1).astype(np.float16))
    chans = [f"{k}_{cfg.encoder}" for k in keys] + ([f"{m}_lesion_frac" for m in images] if lesion_mask else [])
    info = {"feat_dim": meta["feat_dim"] * len(keys), "n_aux": len(images) if lesion_mask else 0,
            "grid": tuple(meta["grid_dhw"])}
    return torch.from_numpy(np.stack(out)), chans, info


def augment_tokens(x, feat_dim, cfg, feat_std):
    B, T, _ = x.shape
    feat = x[..., :feat_dim] + cfg.feat_noise * feat_std * torch.randn_like(x[..., :feat_dim])
    keep = (torch.rand(B, T, 1, device=x.device) >= cfg.token_drop).to(x.dtype)
    return torch.cat([feat, x[..., feat_dim:]], dim=-1) * keep


def augment(x, n_img_ch, cfg):
    B, _, D, H, W = x.shape
    dev = x.device
    u = lambda *s: torch.rand(*s, device=dev) * 2 - 1
    ang = u(B) * math.radians(cfg.aug_rot)
    s = 1 + u(B) * cfg.aug_scale
    theta = torch.zeros(B, 3, 4, device=dev)
    theta[:, 0, 0], theta[:, 0, 1] = ang.cos() / s, -ang.sin() / s
    theta[:, 1, 0], theta[:, 1, 1] = ang.sin() / s, ang.cos() / s
    theta[:, 2, 2] = 1
    theta[:, 0, 3] = u(B) * cfg.aug_shift * 2 / W
    theta[:, 1, 3] = u(B) * cfg.aug_shift * 2 / H
    theta[:, 2, 3] = u(B) * 0.5 * 2 / D
    if cfg.lr_flip:
        theta[torch.rand(B, device=dev) < 0.5, 0] *= -1
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    img = x[:, :n_img_ch]
    inside = (img > 0).to(img.dtype)
    img = img * (1 + u(B, n_img_ch, 1, 1, 1) * cfg.aug_intensity) + cfg.aug_noise * torch.randn_like(img) * inside
    return torch.cat([img, x[:, n_img_ch:]], dim=1)


def batches(idx, size, rng=None):
    idx = np.asarray(idx)
    if rng is not None:
        idx = rng.permutation(idx)
    return [idx[i:i + size] for i in range(0, len(idx), size)]


# ---------------------------------------------------------------------------
# Métricas (idénticas en espíritu a Nuevaprueba.py)
# ---------------------------------------------------------------------------

def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def compute_metrics(y, p, thr):
    """Umbral fijo (Youden en OOF) para todo lo que depende de él; AUC / AUPRC / Brier no dependen."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    tn, fn = int(((pred == 0) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: float(a / b) if b else 0.0
    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    both = len(np.unique(y)) > 1
    return {
        "auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "auprc": float(average_precision_score(y, p)) if both else float("nan"),
        "acc": div(tp + tn, len(y)), "bal_acc": (sens + spec) / 2, "sens": sens, "spec": spec,
        "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn), "f1": div(2 * tp, 2 * tp + fp + fn),
        "mcc": div(tp * tn - fp * fn, math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
        "brier": float(np.mean((p - y) ** 2)), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


CI_METRICS = ("auc", "auprc", "acc", "bal_acc", "sens", "spec", "ppv", "npv", "f1", "mcc", "brier")


def bootstrap_ci(y, p, thr, n_boot, seed):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    stats = {k: [] for k in CI_METRICS}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        m = compute_metrics(y[i], p[i], thr)
        for k in stats:
            stats[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats.items()}


def evaluate_splits(ys, probs, thr, n_boot, seed):
    """ys / probs por split (train = OOF) -> métricas + IC95% de oof, val, test y ext (= val+test juntos)."""
    ys = {**ys, "ext": np.concatenate([ys["val"], ys["test"]])}
    probs = {**probs, "ext": np.concatenate([probs["val"], probs["test"]])}
    out = {}
    for s, key in (("train", "oof"), ("val", "val"), ("test", "test"), ("ext", "ext")):
        out[key] = compute_metrics(ys[s], probs[s], thr)
        out[key]["ci95"] = bootstrap_ci(ys[s], probs[s], thr, n_boot, seed)
    return out


def paired_delta_auc(y, p_a, p_b, n_boot=2000, seed=42):
    """AUC(a) - AUC(b) en los MISMOS pacientes con IC95% por bootstrap pareado.
    Si el IC incluye 0, no hay evidencia de que a sea mejor (o peor) que b."""
    y = np.asarray(y).astype(bool)
    pairs = lambda p: (lambda d: (d > 0) + 0.5 * (d == 0))(np.asarray(p)[y][:, None] - np.asarray(p)[~y][None, :])
    C = pairs(p_a) - pairs(p_b)  # (positivos x negativos): contribución de cada par a la diferencia
    w = np.random.default_rng(seed).multinomial(len(y), np.full(len(y), 1 / len(y)), size=n_boot)
    wp, wn = w[:, y], w[:, ~y]
    ok = (wp.sum(1) > 0) & (wn.sum(1) > 0)
    boot = np.einsum("bi,ij,bj->b", wp[ok], C, wn[ok]) / (wp[ok].sum(1) * wn[ok].sum(1))
    return float(C.mean()), [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def make_model(cfg, img_info):
    info = img_info or {}
    return MultimodalNet(
        in_ch=info.get("n_ch", 0), input_shape=info.get("input_shape"), n_tab=len(FEATURES) if cfg.tabular else 0,
        fusion=cfg.fusion, dim=cfg.dim, depth=cfg.depth, heads=cfg.heads, tab_depth=cfg.tab_depth,
        cross_depth=cfg.cross_depth, drop=cfg.drop, drop_path=cfg.drop_path, head_drop=cfg.head_drop,
        tokenizer=cfg.tokenizer, encoder=cfg.encoder, feat_dim=info.get("feat_dim", 0),
        n_aux=info.get("n_aux", 0), token_grid=info.get("grid"))


def loss_fn(out, yb, pos_weight, aux_weight):
    loss = F.binary_cross_entropy_with_logits(out["logit"].float(), yb, pos_weight=pos_weight)
    if "aux_logits" in out:
        loss = loss + aux_weight * sum(F.binary_cross_entropy_with_logits(l.float(), yb, pos_weight=pos_weight)
                                       for l in out["aux_logits"])
    return loss


@torch.no_grad()
def predict(model, imgs, tab, idx, dev, amp_dtype, batch_size=16):
    model.eval()
    keys = ("logit", "emb", "img_attn", "gates", "cross_attn")
    acc = {k: [] for k in keys}
    for b in batches(idx, batch_size):
        img_b = imgs[b].to(dev).float() if imgs is not None else None
        tab_b = tab[b].to(dev) if tab is not None else None
        with torch.autocast("cuda", dtype=amp_dtype, enabled=dev == "cuda"):
            out = model(img_b, tab_b)
        for k in keys:
            if k in out:
                acc[k].append(out[k].float().cpu())
    res = {k: torch.cat(v).numpy() for k, v in acc.items() if v}
    res["prob"] = 1 / (1 + np.exp(-res.pop("logit").ravel()))
    return res


def train_fold(fold, tr_idx, va_idx, data, cfg, dev, amp_dtype, img_info, history):
    torch.manual_seed(cfg.seed + fold)
    rng = np.random.default_rng(cfg.seed + fold)
    tr = data["train"]
    y_all = torch.tensor(tr["y"], dtype=torch.float32).view(-1, 1)

    scaler = None
    tab_all = None
    if cfg.tabular:
        scaler = StandardScaler().fit(tr["tab_raw"][tr_idx])
        tab_all = torch.tensor(scaler.transform(tr["tab_raw"]), dtype=torch.float32)
    imgs = tr["imgs"]
    model = make_model(cfg, img_info).to(dev)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and n not in ("tab.weight", "tab.bias")]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim > 1 and n not in ("tab.weight", "tab.bias"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / cfg.warmup if e < cfg.warmup else
                                              0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (e - cfg.warmup) / max(1, cfg.epochs - cfg.warmup))))
    grad_scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda" and amp_dtype == torch.float16))

    y_tr = tr["y"][tr_idx]
    pos_weight = torch.tensor([math.sqrt((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))], device=dev)

    best = {"loss": float("inf"), "state": None, "epoch": 0}
    bad = 0
    for epoch in range(cfg.epochs):
        model.train()
        t0, tr_loss, nb = time.time(), 0.0, 0
        for b in batches(tr_idx, cfg.batch_size, rng):
            img_b = img_info["augment"](imgs[b].to(dev, non_blocking=True).float()) if imgs is not None else None
            tab_b = tab_all[b].to(dev) if tab_all is not None else None
            yb = y_all[b].to(dev)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=dev == "cuda"):
                out = model(img_b, tab_b)
            loss = loss_fn(out, yb, pos_weight, cfg.aux_weight)
            opt.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(opt)
            grad_scaler.update()
            tr_loss += loss.item()
            nb += 1
        sched.step()

        pv = predict(model, imgs, tab_all, va_idx, dev, amp_dtype)
        y_va = tr["y"][va_idx]
        p = np.clip(pv["prob"], 1e-6, 1 - 1e-6)
        pw = pos_weight.item()
        va_loss = float(np.mean(-(pw * y_va * np.log(p) + (1 - y_va) * np.log(1 - p))))
        va_auc = roc_auc_score(y_va, p) if len(np.unique(y_va)) > 1 else float("nan")
        history.append({"fold": fold, "epoch": epoch + 1, "train_loss": tr_loss / nb, "val_loss": va_loss,
                        "val_auc": va_auc, "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0})

        if epoch + 1 >= cfg.warmup and va_loss < best["loss"]:
            best = {"loss": va_loss, "state": copy.deepcopy(model.state_dict()), "epoch": epoch + 1}
            bad = 0
        elif epoch + 1 >= cfg.warmup:
            bad += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"    ep {epoch + 1:3d}  train {tr_loss / nb:.4f}  val {va_loss:.4f}  "
                  f"AUC {va_auc:.3f}  ({time.time() - t0:.1f}s/ep)", flush=True)
        if epoch + 1 >= cfg.min_epochs and bad >= cfg.patience:
            print(f"    early stop ep {epoch + 1} (mejor ep {best['epoch']})", flush=True)
            break

    model.load_state_dict(best["state"] if best["state"] is not None else model.state_dict())
    return model, scaler, best["epoch"]


def run_experiment(cfg):
    cfg = finalize_config(cfg)
    out_dir = Path(cfg.out_dir)
    (out_dir / "modelos").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(cfg), indent=1))
    t_start = time.time()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = amp_dtype()
    torch.backends.cudnn.benchmark = True
    print(f"\n{'=' * 70}\n  {cfg.name}   (device={dev}, amp={amp})\n{'=' * 70}", flush=True)

    tab_df = with_images(load_tabular())
    shuffle_rng = np.random.default_rng(cfg.seed + 999)
    data, chans, img_info = {}, [], None
    for split in SPLITS:
        d = tab_df[tab_df.split == split].reset_index(drop=True)
        imgs = None
        if cfg.images:
            if cfg.encoder == "vit3d":
                imgs, chans = load_images(d.Paciente_id, cfg.images, not cfg.no_lesion_mask)
                img_info = {"n_ch": imgs.shape[1], "input_shape": tuple(imgs.shape[2:]),
                            "grid": tuple(int(n) // p for n, p in zip(imgs.shape[2:], (4, 8, 8)))}
            else:
                imgs, chans, info = load_tokens(d.Paciente_id, cfg.images, not cfg.no_lesion_mask, cfg)
                img_info = {"n_ch": len(cfg.images), **info}
            if cfg.control == "shuffle_image":
                imgs = imgs[torch.from_numpy(shuffle_rng.permutation(len(d)))]
            if dev == "cuda":
                imgs = imgs.pin_memory()
        data[split] = {"df": d, "imgs": imgs, "y": d.y.values,
                       "tab_raw": d[FEATURES].values.astype(np.float32) if cfg.tabular else None}
    if img_info and cfg.encoder == "vit3d":
        img_info["augment"] = lambda x: augment(x, len(cfg.images), cfg)
    elif img_info:
        fd = img_info["feat_dim"]
        feat_std = float(data["train"]["imgs"][:16, :, :fd].float().std())
        img_info["augment"] = lambda x: augment_tokens(x, fd, cfg, feat_std)
    print(f"  pacientes train/val/test: {[len(data[s]['y']) for s in SPLITS]}  "
          f"positivos: {[int(data[s]['y'].sum()) for s in SPLITS]}  canales: {chans or '-'}"
          + (f"  encoder: {cfg.encoder}" if cfg.images else ""), flush=True)

    y = data["train"]["y"]
    skf = StratifiedKFold(cfg.folds, shuffle=True, random_state=cfg.seed)
    oof, oof_fold = np.zeros(len(y)), np.zeros(len(y), dtype=int)
    per_fold = {s: [] for s in SPLITS}
    history, best_epochs = [], []

    for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        print(f"  -- fold {k + 1}/{cfg.folds}  (train {len(tr_idx)}, val {len(va_idx)})", flush=True)
        model, scaler, best_ep = train_fold(k, tr_idx, va_idx, data, cfg, dev, amp, img_info, history)
        best_epochs.append(best_ep)
        for s in SPLITS:
            tab = torch.tensor(scaler.transform(data[s]["tab_raw"]), dtype=torch.float32) if cfg.tabular else None
            per_fold[s].append(predict(model, data[s]["imgs"], tab, np.arange(len(data[s]["y"])), dev, amp))
        oof[va_idx] = per_fold["train"][k]["prob"][va_idx]
        oof_fold[va_idx] = k
        if not cfg.no_save_models:
            torch.save({"state_dict": model.state_dict(), "scaler_mean": getattr(scaler, "mean_", None),
                        "scaler_scale": getattr(scaler, "scale_", None)}, out_dir / "modelos" / f"fold{k}.pt")
        elapsed = (time.time() - t_start) / 60
        print(f"     AUC fold-val {roc_auc_score(y[va_idx], oof[va_idx]):.3f} | {elapsed:.1f} min "
              f"(ETA experimento ~{elapsed / (k + 1) * (cfg.folds - k - 1):.1f} min)", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---- métricas ----
    thr = youden_threshold(y, oof)
    probs = {"train": oof, **{s: np.mean([f["prob"] for f in per_fold[s]], axis=0) for s in ("val", "test")}}
    metrics = {"name": cfg.name, "threshold_youden_oof": thr, "best_epochs": best_epochs}
    y_by = {s: data[s]["y"] for s in SPLITS}
    metrics.update(evaluate_splits(y_by, probs, thr, cfg.n_boot, cfg.seed))
    if "gates" in per_fold["val"][0]:
        g = np.concatenate([np.mean([f["gates"] for f in per_fold[s]], axis=0) for s in ("val", "test")])
        metrics["gate_mean"] = {"imagen": float(g[:, 0].mean()), "tabular": float(g[:, 1].mean())}
    metrics["minutes"] = (time.time() - t_start) / 60
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))

    # ---- predicciones por paciente (para stacking / fusión tardía con texto) ----
    rows = []
    for s in SPLITS:
        d = data[s]["df"]
        part = pd.DataFrame({"Paciente_id": d.Paciente_id, "split": s, "y": d.y, "prob": probs[s],
                             "fold": oof_fold if s == "train" else -1})
        if "gates" in per_fold[s][0]:
            gates = (np.stack([f["gates"] for f in per_fold[s]])[oof_fold, np.arange(len(d))] if s == "train"
                     else np.mean([f["gates"] for f in per_fold[s]], axis=0))
            part["gate_imagen"], part["gate_tabular"] = gates[:, 0], gates[:, 1]
        rows.append(part)
    pd.concat(rows).to_csv(out_dir / "predictions.csv", index=False)

    # ---- embeddings por fold (mismo espacio solo dentro de un fold) ----
    np.savez_compressed(out_dir / "embeddings.npz",
                        **{f"emb_{s}": np.stack([f["emb"] for f in per_fold[s]]).astype(np.float16) for s in SPLITS},
                        **{f"pid_{s}": data[s]["df"].Paciente_id.values.astype(str) for s in SPLITS},
                        oof_fold=oof_fold)

    # ---- mapas espaciales: mismo grid en todos los folds => promediables ----
    if cfg.images:
        grid = img_info["grid"]
        att = {}
        for s in SPLITS:
            stack = np.stack([f["img_attn"] for f in per_fold[s]])
            a = stack[oof_fold, np.arange(stack.shape[1])] if s == "train" else stack.mean(0)
            att[f"img_attn_{s}"] = a.reshape(-1, *grid).astype(np.float16)
            if "cross_attn" in per_fold[s][0]:
                c = np.stack([f["cross_attn"] for f in per_fold[s]])
                c = c[oof_fold, np.arange(c.shape[1])] if s == "train" else c.mean(0)
                att[f"cross_attn_{s}"] = c.reshape(c.shape[0], c.shape[1], *grid).astype(np.float16)
            att[f"pid_{s}"] = data[s]["df"].Paciente_id.values.astype(str)
        np.savez_compressed(out_dir / "attention.npz", grid=np.array(grid), features=np.array(FEATURES), **att)

    pd.DataFrame(history).to_csv(out_dir / "log.csv", index=False)

    def fmt(m):
        lo, hi = m["ci95"]["auc"]
        return (f"AUC {m['auc']:.3f} [{lo:.2f}-{hi:.2f}]  AUPRC {m['auprc']:.2f}  F1 {m['f1']:.2f}  "
                f"acc {m['acc']:.2f}  sens {m['sens']:.2f}  spec {m['spec']:.2f}  FP {m['fp']}")
    print(f"\n  RESULTADO {cfg.name}  (umbral Youden OOF = {thr:.3f}, {metrics['minutes']:.1f} min)")
    print(f"    OOF  : {fmt(metrics['oof'])}\n    VAL  : {fmt(metrics['val'])}\n    TEST : {fmt(metrics['test'])}"
          f"\n    EXT  : {fmt(metrics['ext'])}   (val+test juntos, n={len(y_by['val']) + len(y_by['test'])})")
    if "gate_mean" in metrics:
        print(f"    confianza media -> imagen {metrics['gate_mean']['imagen']:.2f} | "
              f"tabular {metrics['gate_mean']['tabular']:.2f}")
    return metrics


if __name__ == "__main__":
    run_experiment(build_parser().parse_args())
