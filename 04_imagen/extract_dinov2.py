"""
Extrae tokens de DINOv2 (ViT 2D preentrenado, CONGELADO) corte por corte, una sola vez.

La adquisición es 2D (cortes axiales de ~6 mm con 0.94 mm en el plano), así que cada
corte es una imagen 2D real: se remuestrea a 1 mm en el plano (224 x 224 = 224 mm de
FOV, el mismo que el ViT 3D) sin interpolar entre cortes.

Por corte: DINOv2 -> 16x16 tokens -> promedio 2x2 -> 8x8 (cada token ~28 x 28 mm).
Por paciente queda un grid (28 cortes, 8, 8) que conserva la posición 3D de cada token,
más la fracción de lesión y de cerebro de cada token.

Al final repite el chequeo que se le hizo a BERT (similitud coseno entre pacientes) y
una prueba lineal (regresión logística sobre PCA) para ver si la representación
congelada trae señal del desenlace ANTES de entrenar nada.

Uso (desde text/):  python 04_imagen/extract_dinov2.py
                    python 04_imagen/extract_dinov2.py --backbone vit_base_patch14_reg4_dinov2.lvd142m
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import timm
import torch
import torch.nn.functional as F

from preprocess import SPLIT_ROOT, SPLITS, load_patient
from rutas import DINO_ROOT, amp_dtype

DEFAULT_BACKBONE = "vit_small_patch14_reg4_dinov2.lvd142m"
SPACING = (1.0, 1.0, 6.0)
SHAPE = (224, 224, 28)
INTENSITY_SCALE = {"dwi": 4.0, "adc": 3.0}  # rDWI y ADC (1e-3 mm²/s) -> [0, 1]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def cache_dir(backbone):
    return DINO_ROOT / backbone.split(".")[0]


def _load(job):
    split, pid = job
    vols, _ = load_patient(split, pid, SPACING, SHAPE)
    return pid, vols


@torch.no_grad()
def encode_volume(model, vol, scale, pool, dev):
    x = torch.from_numpy(vol.astype(np.float32) / scale).clamp(0, 1).unsqueeze(1)
    x = ((x.expand(-1, 3, -1, -1) - MEAN) / STD).to(dev)
    with torch.autocast("cuda", dtype=amp_dtype(), enabled=dev == "cuda"):
        tok = model.forward_features(x)[:, model.num_prefix_tokens:]
    side = int(round(tok.shape[1] ** 0.5))
    tok = tok.float().transpose(1, 2).reshape(x.shape[0], -1, side, side)
    tok = F.avg_pool2d(tok, pool)
    return tok.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16)  # (D, h, w, C)


def token_fraction(mask, hw):
    m = torch.from_numpy(mask.astype(np.float32)).unsqueeze(1)
    return F.adaptive_avg_pool2d(m, hw).squeeze(1).numpy().astype(np.float16)


def representation_report(out):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    import train as T

    df = T.load_tabular()
    df = df[df.split == "train"]
    df = df[[(out / f"{p}.npz").exists() for p in df.Paciente_id]]
    y = df.y.values
    print(f"\n--- Chequeo de la representación congelada (train, n={len(y)}, positivos={y.sum()}) ---")
    print("  coseno crudo ~0.99 = colapso (lo que pasó con BERT); 'centrado' resta la media entre pacientes.")
    print("  AUC lineal = regresión logística sobre PCA(8), CV 5x5. ~0.50 = sin señal del desenlace.\n")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    for m in ("adc", "dwi"):
        vec = {"cerebro": [], "lesion": []}
        for pid in df.Paciente_id:
            d = np.load(out / f"{pid}.npz")
            f = d[m].astype(np.float32).reshape(-1, d[m].shape[-1])
            wb = d["brain"].astype(np.float32).reshape(-1)
            wl = d[f"{m}_lesion"].astype(np.float32).reshape(-1)
            vec["cerebro"].append(f.T @ wb / max(wb.sum(), 1e-6))
            vec["lesion"].append(f.T @ wl / wl.sum() if wl.sum() > 0 else f.T @ wb / max(wb.sum(), 1e-6))
        for pooling, v in vec.items():
            V = np.stack(v)
            iu = np.triu_indices(len(V), k=1)
            raw = cosine_similarity(V)[iu]
            cen = cosine_similarity(V - V.mean(0))[iu]
            auc = cross_val_score(make_pipeline(StandardScaler(), PCA(8), LogisticRegression(max_iter=2000)),
                                  V, y, cv=cv, scoring="roc_auc")
            print(f"  {m.upper()} promedio-{pooling:8s} coseno crudo {raw.mean():.3f} (min {raw.min():.3f}) | "
                  f"centrado {cen.mean():+.3f} | AUC lineal {auc.mean():.3f} ± {auc.std():.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default=DEFAULT_BACKBONE,
                   help="cualquier DINOv2 de timm, ej. vit_base_patch14_reg4_dinov2.lvd142m")
    p.add_argument("--pool", type=int, default=2, help="promedio de tokens en el plano (2 -> grid 8x8)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    out = cache_dir(args.backbone)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(args.backbone, pretrained=True, img_size=SHAPE[0], num_classes=0).to(dev).eval()

    jobs = [(s, p.name) for s in SPLITS for p in sorted((SPLIT_ROOT / s / "rawdata").iterdir())]
    todo = [j for j in jobs if not (out / f"{j[1]}.npz").exists()]
    print(f"{args.backbone}: {len(jobs)} pacientes, {len(todo)} por extraer -> {out}")
    grid = None
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (pid, vols) in enumerate(ex.map(_load, todo)):
            feats = {m: encode_volume(model, vols[m], INTENSITY_SCALE[m], args.pool, dev) for m in ("adc", "dwi")}
            hw = feats["adc"].shape[1:3]
            grid = [feats["adc"].shape[0], *hw]
            np.savez_compressed(out / f"{pid}.npz", **feats,
                                adc_lesion=token_fraction(vols["adc_lesion"], hw),
                                dwi_lesion=token_fraction(vols["dwi_lesion"], hw),
                                brain=token_fraction(vols["brain"], hw))
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)}", flush=True)

    if grid is None:
        d = np.load(out / f"{jobs[0][1]}.npz")
        grid = list(d["adc"].shape[:3])
    feat_dim = int(np.load(out / f"{jobs[0][1]}.npz")["adc"].shape[-1])
    (out / "meta.json").write_text(json.dumps({"backbone": args.backbone, "grid_dhw": grid, "feat_dim": feat_dim,
                                               "spacing_xyz_mm": SPACING, "shape_xyz": SHAPE, "pool": args.pool},
                                              indent=1))
    print(f"grid de tokens por paciente {grid}, dim {feat_dim}")
    representation_report(out)


if __name__ == "__main__":
    main()
