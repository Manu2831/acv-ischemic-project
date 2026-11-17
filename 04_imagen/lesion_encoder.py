"""
Encoder ESPECIALIZADO EN LESIÓN: SwinUNETR (transformer 3D) entrenado para SEGMENTAR la
lesión isquémica en ADC+DWI, y luego usado congelado como extractor de features.

Por qué así: con 109 pacientes, una sola etiqueta (mRS) por paciente no alcanza para que
un encoder aprenda a ver lesiones. Segmentar da una etiqueta por vóxel (millones por
volumen) y hay datos públicos: ISLES 2022 (250 casos DWI+ADC con máscara, CC BY 4.0).

Etapas (--etapa):
  preparar : remuestrea TODOS tus pacientes + ISLES 2022 (si está) a un grid común
  entrenar : SwinUNETR (Dice + BCE) con DOS salidas:
               canal 0 = lesión en DWI (ana_dwi_mask de tu cohorte + máscara de ISLES)
               canal 1 = lesión en ADC (ana_adc_mask, el núcleo; solo tu cohorte: ISLES no la
                         trae, así que en ISLES ese canal no suma pérdida)
             y "modality dropout": a veces apaga DWI (solo ADC, --p_solo_adc) o ADC (solo DWI,
             --p_solo_dwi), para que el mismo encoder sirva con ADC solo, DWI solo o ambos.
             Solo se entrena con máscaras de TRAIN de tu cohorte + ISLES; val/test nunca entran.
             El mejor checkpoint se elige por el Dice promedio de las tres entradas, cada una
             contra su máscara (ADC solo -> máscara ADC; DWI y ADC+DWI -> máscara DWI).
  extraer  : congela el modelo y guarda por paciente tokens 3D (grid 4x8x8) para las tres
             entradas (adc, dwi, adc+dwi): features del encoder a 2 escalas + probabilidad
             de lesión predicha por región (DWI y ADC)  -> cache_lesion/<modelo>/
  todo     : las tres (cada una se salta si ya está hecha)

Uso (desde text/):
  python 04_imagen/lesion_encoder.py --etapa todo --descargar_isles
  python 04_imagen/lesion_encoder.py --etapa todo --quick          (prueba de humo)
"""

import argparse
import json
import math
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from preprocess import SPLITS, crop_pad_center
from rutas import ISLES_DIR, LESION_ROOT, SPLIT_ROOT, amp_dtype, nifti

PREP_DIR = LESION_ROOT / "prep"
SPACING = (1.75, 1.75, 3.0)   # mm (x, y, z): 224 x 224 x 192 mm de FOV
SHAPE = (128, 128, 64)        # múltiplos de 32 (lo exige SwinUNETR)
ISLES_URL = "https://zenodo.org/records/7153326/files/ISLES-2022.zip?download=1"
CONDITIONS = {"adc": (1, 0), "dwi": (0, 1), "adc+dwi": (1, 1)}  # canales [ADC, DWI] encendidos
HEADS = ("dwi_lesion", "adc_lesion")                             # canales de salida de la segmentación
MATCHED = {"adc": 1, "dwi": 0, "adc+dwi": 0}                     # máscara que corresponde a cada entrada


# ---------------------------------------------------------------------------
# Preparación (mismo pipeline para tu cohorte e ISLES -> mismo dominio)
# ---------------------------------------------------------------------------

def _canon(path):
    img = nib.as_closest_canonical(nib.load(str(nifti(path))))
    return img.get_fdata(dtype=np.float32), np.array(img.header.get_zooms()[:3], dtype=np.float32)


def prepare_case(adc_path, dwi_path, masks):
    """Sin skull-stripping (ISLES no lo trae). z-score por canal sobre vóxeles > 0, como DeepISLES."""
    adc, sp = _canon(adc_path)
    dwi, sp_dwi = _canon(dwi_path)
    if adc.shape != dwi.shape:
        adc = ndimage.zoom(adc, np.array(dwi.shape) / np.array(adc.shape), order=1)
    factors = sp_dwi / np.asarray(SPACING, dtype=np.float32)
    vols = [ndimage.zoom(np.clip(v, 0, None), factors, order=1) for v in (adc, dwi)]
    head = vols[1] > 0.1 * np.percentile(vols[1], 99)
    center = np.array(ndimage.center_of_mass(head)) if head.any() else np.array(vols[1].shape) / 2
    out = {}
    for name, v in zip(("adc", "dwi"), vols):
        v = crop_pad_center(v, center, SHAPE)
        nz = v > 0
        v = np.where(nz, (v - v[nz].mean()) / (v[nz].std() + 1e-6), 0)
        out[name] = np.transpose(v, (2, 1, 0)).astype(np.float16)
    for name, path in masks.items():
        m = _canon(path)[0] > 0
        if m.shape != dwi.shape:
            m = ndimage.zoom(m.astype(np.uint8), np.array(dwi.shape) / np.array(m.shape), order=0) > 0
        m = ndimage.zoom(m.astype(np.uint8), factors, order=0)
        out[name] = np.transpose(crop_pad_center(m, center, SHAPE), (2, 1, 0)).astype(np.uint8)
    return out


def isles_cases():
    """Busca casos ISLES por nombre de archivo (BIDS: *_adc, *_dwi, *_msk), sin asumir carpetas exactas."""
    if not ISLES_DIR.exists():
        return []
    masks = {p.name.split("_")[0]: p for p in ISLES_DIR.rglob("*_msk.nii.gz")}
    cases = []
    for dwi in sorted(ISLES_DIR.rglob("*_dwi.nii.gz")):
        sub = dwi.name.split("_")[0]
        adc = dwi.with_name(dwi.name.replace("_dwi.nii.gz", "_adc.nii.gz"))
        if adc.exists() and sub in masks:
            cases.append((f"isles_{sub}", adc, dwi, masks[sub]))
    return cases


def download_isles():
    zpath = ISLES_DIR / "ISLES-2022.zip"
    ISLES_DIR.mkdir(parents=True, exist_ok=True)
    if not zpath.exists():
        print(f"Descargando ISLES 2022 (1.7 GB) -> {zpath}", flush=True)
        urllib.request.urlretrieve(ISLES_URL, zpath)
    print("Descomprimiendo ISLES 2022...", flush=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(ISLES_DIR)


def _prep_job(job):
    cid, adc, dwi, masks = job
    out = PREP_DIR / f"{cid}.npz"
    if not out.exists():
        np.savez_compressed(out, **prepare_case(adc, dwi, masks))
    return cid


def stage_prepare(args):
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split in SPLITS:
        for pdir in sorted((SPLIT_ROOT / split / "rawdata").iterdir()):
            raw, der = pdir / "ses-01", SPLIT_ROOT / split / "derivatives" / pdir.name / "ses-01"
            jobs.append((pdir.name, raw / "adc.nii.gz", raw / "dwi.nii.gz",
                         {"dwi_lesion": der / "ana_dwi_mask.nii.gz", "adc_lesion": der / "ana_adc_mask.nii.gz"}))
    isles = [] if args.sin_isles else isles_cases()
    jobs += [(cid, adc, dwi, {"dwi_lesion": msk}) for cid, adc, dwi, msk in isles]
    print(f"Preparando {len(jobs)} casos ({len(isles)} de ISLES 2022) -> {PREP_DIR}", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(_prep_job, jobs)):
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    (PREP_DIR / "meta.json").write_text(json.dumps({"spacing_xyz_mm": SPACING, "shape_xyz": SHAPE,
                                                    "channels": ["adc", "dwi"], "n_isles": len(isles)}))
    if len(isles) == 0 and not args.sin_isles:
        print("AVISO: no se encontró ISLES 2022; el encoder se entrenará solo con tu cohorte (más débil). "
              "Usá --descargar_isles o ACV_ISLES_DIR.", flush=True)


# ---------------------------------------------------------------------------
# Entrenamiento de la segmentación
# ---------------------------------------------------------------------------

def build_model(feature_size, out_channels=len(HEADS), checkpointing=False):
    from monai.networks.nets import SwinUNETR
    return SwinUNETR(in_channels=2, out_channels=out_channels, feature_size=feature_size, use_v2=True,
                     use_checkpoint=checkpointing)


def training_ids(args):
    ids = [p.name for p in (SPLIT_ROOT / "train" / "rawdata").iterdir()]
    if not args.sin_isles:
        ids += [p.stem for p in PREP_DIR.glob("isles_*.npz")]
    ids = [i for i in ids if (PREP_DIR / f"{i}.npz").exists()]
    return sorted(ids)


def dice_score(logits, target):
    pred = (logits > 0).float()
    inter = (pred * target).sum(dim=(1, 2, 3, 4))
    denom = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    return torch.where(denom > 0, 2 * inter / denom.clamp(min=1), torch.ones_like(denom))


def stage_train(args, ckpt):
    from monai.losses import DiceCELoss
    from train import augment

    ids = training_ids(args)
    if args.quick:
        ids = ids[:6] + [i for i in ids if i.startswith("isles_")][:6]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ids))
    n_val = max(2, len(ids) // 10)
    val_ids, tr_ids = [ids[i] for i in perm[:n_val]], [ids[i] for i in perm[n_val:]]
    n_isles = sum(i.startswith("isles_") for i in ids)
    print(f"Segmentación: {len(tr_ids)} train / {len(val_ids)} val (ISLES: {n_isles}, tu cohorte: {len(ids) - n_isles})",
          flush=True)

    X, has_adc = {}, {}   # canales: [ADC, DWI, máscara DWI, máscara ADC]; ISLES no trae máscara ADC
    for i in ids:
        d = np.load(PREP_DIR / f"{i}.npz")
        has_adc[i] = "adc_lesion" in d.files
        adc_mask = d["adc_lesion"] if has_adc[i] else np.zeros_like(d["dwi_lesion"])
        X[i] = torch.from_numpy(np.stack([d["adc"], d["dwi"], d["dwi_lesion"].astype(np.float16),
                                          adc_mask.astype(np.float16)]))
    print(f"Máscara ADC (núcleo) disponible en {sum(has_adc.values())} casos de tu cohorte; "
          f"entradas: ADC solo {args.p_solo_adc:.0%}, DWI solo {args.p_solo_dwi:.0%}, "
          f"ambas {1 - args.p_solo_adc - args.p_solo_dwi:.0%}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = amp_dtype()
    low_mem = dev == "cuda" and torch.cuda.get_device_properties(0).total_memory < 8e9
    model = build_model(args.feature_size, checkpointing=args.checkpointing or low_mem).to(dev)
    print(f"SwinUNETR fs={args.feature_size}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parámetros, "
          f"amp={amp}, checkpointing={args.checkpointing or low_mem}", flush=True)
    loss_fn = DiceCELoss(sigmoid=True, batch=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = args.epochs * math.ceil(len(tr_ids) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda" and amp == torch.float16)
    aug_cfg = SimpleNamespace(aug_rot=15.0, aug_scale=0.15, aug_shift=8.0, aug_intensity=0.15, aug_noise=0.05,
                              lr_flip=True)

    best, log = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        order = rng.permutation(len(tr_ids))
        for s in range(0, len(order), args.batch_size):
            idx = [tr_ids[j] for j in order[s:s + args.batch_size]]
            b = torch.stack([X[i] for i in idx]).to(dev).float()
            b = augment(b, 2, aug_cfg)
            x, y = b[:, :2], (b[:, 2:] > 0.5).float()
            r = torch.rand(x.shape[0], device=dev)
            x[r < args.p_solo_adc, 1] = 0                                                   # solo ADC
            x[(r >= args.p_solo_adc) & (r < args.p_solo_adc + args.p_solo_dwi), 0] = 0     # solo DWI
            with torch.autocast("cuda", dtype=amp, enabled=dev == "cuda"):
                logits = model(x).float()
            loss = loss_fn(logits[:, :1], y[:, :1])
            sup = torch.tensor([has_adc[i] for i in idx], device=dev)
            if sup.any():
                loss = loss + args.peso_adc * loss_fn(logits[sup, 1:], y[sup, 1:])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        row = {"epoch": epoch + 1, "loss": tot / math.ceil(len(order) / args.batch_size), "sec": time.time() - t0}
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            model.eval()
            dices = {}   # dice_<entrada>: vs máscara DWI (como antes) | dice_adcmask_<entrada>: vs máscara ADC
            with torch.no_grad():
                for i in val_ids:
                    v = X[i][None].to(dev).float()
                    tgt = (v[:, 2:] > 0.5).float()
                    for c, on in CONDITIONS.items():
                        x = v[:, :2] * torch.tensor(on, device=dev, dtype=v.dtype).view(1, 2, 1, 1, 1)
                        with torch.autocast("cuda", dtype=amp, enabled=dev == "cuda"):
                            out = model(x).float()
                        dices.setdefault(f"dice_{c}", []).append(dice_score(out[:, :1], tgt[:, :1]).item())
                        if has_adc[i]:
                            dices.setdefault(f"dice_adcmask_{c}", []).append(dice_score(out[:, 1:], tgt[:, 1:]).item())
            row.update({k: float(np.mean(d)) for k, d in dices.items()})
            matched = [row.get(f"dice_adcmask_{c}" if MATCHED[c] else f"dice_{c}", row[f"dice_{c}"]) for c in CONDITIONS]
            row["dice_seleccion"] = float(np.mean(matched))
            if row["dice_seleccion"] > best:
                best = row["dice_seleccion"]
                torch.save({"state_dict": model.state_dict(), "feature_size": args.feature_size,
                            "out_channels": len(HEADS), "heads": HEADS, "epoch": epoch + 1, "dice_val": row}, ckpt)
            adc_core = row.get("dice_adcmask_adc", float("nan"))
            print(f"  ep {epoch + 1:4d}  loss {row['loss']:.4f}  Dice val vs máscara DWI: ADC+DWI {row['dice_adc+dwi']:.3f}"
                  f" | ADC {row['dice_adc']:.3f} | DWI {row['dice_dwi']:.3f}  ·  ADC solo vs máscara ADC {adc_core:.3f}"
                  f"  ·  selección {row['dice_seleccion']:.3f}  ({row['sec']:.0f}s/ep)", flush=True)
        log.append(row)
        (ckpt.parent / "entrenamiento_log.json").write_text(json.dumps(log, indent=1))
    print(f"Mejor Dice de selección (promedio de las 3 entradas, cada una vs su máscara): {best:.3f} -> {ckpt}",
          flush=True)


# ---------------------------------------------------------------------------
# Extracción de tokens (encoder congelado)
# ---------------------------------------------------------------------------

@torch.no_grad()
def stage_extract(args, ckpt, out):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = amp_dtype()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    n_out = state.get("out_channels", 1)   # checkpoints viejos: una sola salida (máscara DWI)
    model = build_model(state["feature_size"], n_out).to(dev).eval()
    model.load_state_dict(state["state_dict"])
    hidden = {}
    model.swinViT.register_forward_hook(lambda m, i, o: hidden.__setitem__("h", o))

    pids = [p.name for s in SPLITS for p in sorted((SPLIT_ROOT / s / "rawdata").iterdir())]
    pids = [p for p in pids if (PREP_DIR / f"{p}.npz").exists()]
    grid = None
    for k, pid in enumerate(pids):
        d = np.load(PREP_DIR / f"{pid}.npz")
        x_full = torch.from_numpy(np.stack([d["adc"], d["dwi"]]))[None].to(dev).float()
        save = {}
        for c, on in CONDITIONS.items():
            x = x_full * torch.tensor(on, device=dev, dtype=x_full.dtype).view(1, 2, 1, 1, 1)
            with torch.autocast("cuda", dtype=amp, enabled=dev == "cuda"):
                logits = model(x)
            h3 = hidden["h"][3].float()                                   # 1/16: (1, 8fs, 4, 8, 8)
            h2 = F.adaptive_avg_pool3d(hidden["h"][2].float(), h3.shape[2:])  # 1/8 -> mismo grid
            pred = F.adaptive_avg_pool3d(torch.sigmoid(logits.float()), h3.shape[2:])
            tok = torch.cat([h3, h2, pred], dim=1)[0].permute(1, 2, 3, 0)  # (d, h, w, C)
            save[c] = tok.cpu().numpy().astype(np.float16)
            grid = list(tok.shape[:3])
        for m in ("dwi_lesion", "adc_lesion"):
            mk = torch.from_numpy(d[m].astype(np.float32))[None, None]
            save[m] = F.adaptive_avg_pool3d(mk, grid)[0, 0].numpy().astype(np.float16)
        np.savez_compressed(out / f"{pid}.npz", **save)
        if (k + 1) % 40 == 0:
            print(f"  {k + 1}/{len(pids)}", flush=True)
    feat_dim = int(save["adc+dwi"].shape[-1])
    (out / "meta.json").write_text(json.dumps({
        "joint_inputs": True, "feat_dim": feat_dim, "grid_dhw": grid, "conditions": list(CONDITIONS),
        "features": "swin 1/16 + swin 1/8 (promediado) + prob. de lesión predicha",
        "pred_channels": list(HEADS[:n_out]),
        "spacing_xyz_mm": SPACING, "shape_xyz": SHAPE, "checkpoint_dice_val": state["dice_val"]}, indent=1))
    print(f"Tokens: grid {grid}, dim {feat_dim} -> {out}", flush=True)
    representation_report(out)


def representation_report(out):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    import train as T

    df = T.load_tabular()
    df = df[(df.split == "train") & np.array([(out / f"{p}.npz").exists() for p in df.Paciente_id])]
    if df.y.nunique() < 2 or len(df) < 20:
        return
    y = df.y.values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    files = [np.load(out / f"{p}.npz") for p in df.Paciente_id]
    print(f"\n--- Chequeo de la representación (train, n={len(y)}): AUC lineal sobre PCA(8), CV 5x5 ---")
    print("  ~0.50 = el encoder congelado no trae señal del desenlace por sí solo")
    for c in CONDITIONS:
        V = np.stack([f[c].astype(np.float32).reshape(-1, f[c].shape[-1]).mean(0) for f in files])
        auc = cross_val_score(make_pipeline(StandardScaler(), PCA(8), LogisticRegression(max_iter=2000)),
                              V, y, cv=cv, scoring="roc_auc")
        print(f"  {c:8s} AUC lineal {auc.mean():.3f} ± {auc.std():.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--etapa", default="todo", choices=["preparar", "entrenar", "extraer", "todo"])
    p.add_argument("--modelo", default="swinunetr_fs24", help="nombre de la carpeta en cache_lesion/")
    p.add_argument("--feature_size", type=int, default=24, help="SwinUNETR: 24 (rápido) o 48 (más grande)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--p_solo_adc", type=float, default=0.30, help="fracción de lotes con DWI apagado (antes 0.15)")
    p.add_argument("--p_solo_dwi", type=float, default=0.15, help="fracción de lotes con ADC apagado")
    p.add_argument("--peso_adc", type=float, default=1.0, help="peso de la pérdida de la máscara ADC")
    p.add_argument("--checkpointing", action="store_true", help="menos memoria GPU, más lento")
    p.add_argument("--sin_isles", action="store_true", help="no usar ISLES 2022 aunque esté disponible")
    p.add_argument("--descargar_isles", action="store_true", help="descargar ISLES 2022 si no está (necesita internet)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--quick", action="store_true", help="prueba de humo: pocos casos, 2 épocas")
    p.add_argument("--forzar", action="store_true", help="rehacer la etapa aunque ya exista")
    args = p.parse_args()
    if args.quick:
        args.epochs, args.val_every, args.modelo = 2, 1, args.modelo + "_quick"

    out = LESION_ROOT / args.modelo
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "swinunetr_lesion.pt"
    todo = args.etapa == "todo"

    if args.descargar_isles and not args.sin_isles and not isles_cases():
        download_isles()
    if (todo or args.etapa == "preparar") and (args.forzar or not (PREP_DIR / "meta.json").exists()):
        stage_prepare(args)
    if (todo or args.etapa == "entrenar") and (args.forzar or not ckpt.exists()):
        stage_train(args, ckpt)
    if (todo or args.etapa == "extraer") and (args.forzar or not (out / "meta.json").exists()):
        stage_extract(args, ckpt, out)


if __name__ == "__main__":
    main()
