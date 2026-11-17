"""
Preprocesa DWI / ADC / máscaras (.nii.gz) a un grid común y los guarda en cache/*.npz.
Se corre UNA vez; el entrenamiento lee solo la caché.

  1. Reorienta a RAS canónico (todos los pacientes con la misma orientación).
  2. Remuestrea a espaciado físico fijo (por defecto 2 x 2 x 6 mm): mismo tamaño
     físico por vóxel => la posición de un token significa lo mismo entre pacientes.
  3. Recorta / rellena alrededor del centroide de la máscara de cerebro.
  4. Intensidades: DWI / mediana del cerebro (rDWI, tejido normal ~1, lesión >1);
     ADC / 1000 (unidades 1e-3 mm²/s, se conserva el valor físico, core < ~0.62).
  5. Guarda en layout (Z, Y, X) = (D, H, W) para torch.

Uso:  python 04_imagen/preprocess.py        (desde la carpeta text/)
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from scipy import ndimage

from rutas import CACHE_DIR, SPLIT_ROOT, nifti

SPLITS = ("train", "val", "test")


def load_canonical(path):
    img = nib.as_closest_canonical(nib.load(str(nifti(path))))
    return img.get_fdata(dtype=np.float32), np.array(img.header.get_zooms()[:3], dtype=np.float32)


def crop_pad_center(vol, center, shape):
    out = np.zeros(shape, dtype=vol.dtype)
    src, dst = [], []
    for c, s, n in zip(center, shape, vol.shape):
        lo = int(round(c)) - s // 2
        s0, s1 = max(lo, 0), min(lo + s, n)
        src.append(slice(s0, s1))
        dst.append(slice(s0 - lo, s1 - lo))
    out[tuple(dst)] = vol[tuple(src)]
    return out


def load_patient(split, pid, target_spacing, target_shape):
    """Devuelve volúmenes normalizados en layout (D, H, W) + metadatos."""
    raw = SPLIT_ROOT / split / "rawdata" / pid / "ses-01"
    der = SPLIT_ROOT / split / "derivatives" / pid / "ses-01"

    dwi, spacing = load_canonical(raw / "dwi.nii.gz")
    adc, _ = load_canonical(raw / "adc.nii.gz")
    brain = (load_canonical(raw / "dwi_brain_mask.nii.gz")[0] > 0) | (load_canonical(raw / "adc_brain_mask.nii.gz")[0] > 0)
    les_dwi = load_canonical(der / "ana_dwi_mask.nii.gz")[0] > 0
    les_adc = load_canonical(der / "ana_adc_mask.nii.gz")[0] > 0
    lesion_ml = float(les_dwi.sum()) * float(np.prod(spacing)) / 1000.0

    factors = spacing / np.asarray(target_spacing, dtype=np.float32)
    dwi = ndimage.zoom(dwi, factors, order=1)
    adc = ndimage.zoom(adc, factors, order=1)
    brain = ndimage.zoom(brain.astype(np.uint8), factors, order=0) > 0
    les_dwi = ndimage.zoom(les_dwi.astype(np.uint8), factors, order=0) > 0
    les_adc = ndimage.zoom(les_adc.astype(np.uint8), factors, order=0) > 0

    center = np.array(ndimage.center_of_mass(brain))
    dwi, adc = (crop_pad_center(v, center, target_shape) for v in (dwi, adc))
    brain, les_dwi, les_adc = (crop_pad_center(v, center, target_shape) for v in (brain, les_dwi, les_adc))

    dwi = np.clip(dwi / max(np.median(dwi[brain]), 1e-6), 0, 6) * brain
    adc = np.clip(adc / 1000.0, 0, 4) * brain

    to_dhw = lambda v: np.ascontiguousarray(np.transpose(v, (2, 1, 0)))
    vols = {"dwi": to_dhw(dwi).astype(np.float16), "adc": to_dhw(adc).astype(np.float16),
            "dwi_lesion": to_dhw(les_dwi).astype(np.uint8), "adc_lesion": to_dhw(les_adc).astype(np.uint8),
            "brain": to_dhw(brain).astype(np.uint8)}
    return vols, {
        "Paciente_id": pid, "split": split,
        "orig_spacing": spacing.round(3).tolist(),
        "lesion_dwi_ml_orig": lesion_ml,
        "lesion_dwi_vox_cache": int(les_dwi.sum()),
        "lesion_adc_vox_cache": int(les_adc.sum()),
        "brain_vox_cache": int(brain.sum()),
    }


def process_patient(args):
    split, pid, target_spacing, target_shape = args
    vols, info = load_patient(split, pid, target_spacing, target_shape)
    np.savez_compressed(CACHE_DIR / f"{pid}.npz", **vols)
    return info


def save_qc_png(records, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sorted(records, key=lambda r: -r["lesion_dwi_vox_cache"])
    picks = [recs[i] for i in np.linspace(0, len(recs) - 1, n).astype(int)]
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for row, r in zip(axes, picks):
        d = np.load(CACHE_DIR / f"{r['Paciente_id']}.npz")
        z = int(np.argmax(d["dwi_lesion"].sum(axis=(1, 2)))) if d["dwi_lesion"].any() else d["dwi"].shape[0] // 2
        row[0].imshow(d["dwi"][z].astype(np.float32), cmap="gray", origin="lower", vmin=0, vmax=3)
        row[1].imshow(d["adc"][z].astype(np.float32), cmap="gray", origin="lower", vmin=0, vmax=2)
        row[2].imshow(d["dwi"][z].astype(np.float32), cmap="gray", origin="lower", vmin=0, vmax=3)
        row[2].contour(d["dwi_lesion"][z], colors="r", linewidths=0.8)
        row[0].set_ylabel(f"{r['Paciente_id']}  z={z}")
        for ax, t in zip(row, ["rDWI", "ADC (1e-3)", "DWI + lesión"]):
            ax.set_title(t, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(CACHE_DIR / "qc_preprocess.png", dpi=90)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spacing", type=float, nargs=3, default=[2.0, 2.0, 6.0], help="mm (x y z)")
    p.add_argument("--shape", type=int, nargs=3, default=[112, 112, 28], help="vóxeles (x y z); múltiplos de 8,8,4")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(s, pid, tuple(args.spacing), tuple(args.shape))
            for s in SPLITS for pid in sorted(p.name for p in (SPLIT_ROOT / s / "rawdata").iterdir())]
    print(f"Procesando {len(jobs)} pacientes -> {CACHE_DIR}")
    with ProcessPoolExecutor(args.workers) as ex:
        records = list(ex.map(process_patient, jobs))

    meta = {"spacing_xyz_mm": args.spacing, "shape_xyz": args.shape, "layout": "DHW = (Z, Y, X), RAS",
            "patients": records}
    (CACHE_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    lost = [r["Paciente_id"] for r in records if r["lesion_dwi_ml_orig"] > 0 and r["lesion_dwi_vox_cache"] == 0]
    print(f"Listo. Pacientes con lesión que quedó fuera del recorte: {lost or 'ninguno'}")
    save_qc_png(records)
    print(f"QC visual: {CACHE_DIR / 'qc_preprocess.png'}")


if __name__ == "__main__":
    main()
