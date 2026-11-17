"""Prueba corta: ¿las lesiones de buen (mRS <= 2) y mal desenlace (mRS > 2) caen en zonas distintas?

1. Lleva cada máscara de lesión DWI a una caja común (normalización aproximada con la máscara de cerebro:
   endereza la rotación axial y escala ancho/largo/alto; NO es un registro a un atlas).
2. Promedia las máscaras por grupo (mapa de frecuencia) y calcula la diferencia.
3. Mide si la ubicación predice el desenlace con validación cruzada (el mapa se arma solo con los folds de
   entrenamiento) y lo compara con el volumen y con etiquetas permutadas.

Solo usa TRAIN (val y test quedan sin mirar). Uso: python mapa_lesiones.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "acv_split" / "train"
OUT = Path(__file__).resolve().parent / "resultados"
OUT.mkdir(exist_ok=True)
G = (72, 88, 24)  # caja común: x (izq-der), y (post-ant), z (inf-sup)
# 011/030: misma RM con mRS en conflicto; 031: misma RM que 014.
EXCLUIR = {"ACV-011", "ACV-030", "ACV-031"}


def load(p):
    if not p.exists():
        p = p.with_suffix("")
    img = nib.as_closest_canonical(nib.load(str(p)))
    return np.asarray(img.get_fdata(), dtype=np.float32), np.array(img.header.get_zooms()[:3], dtype=float)


def normalizar(pid):
    """Devuelve (lesion, dwi) en la caja común, o None si falta algo."""
    raw, der = SPLIT / "rawdata" / pid / "ses-01", SPLIT / "derivatives" / pid / "ses-01"
    try:
        dwi, zm = load(raw / "dwi.nii.gz")
        brain = load(raw / "dwi_brain_mask.nii.gz")[0] > 0
        les = load(der / "ana_dwi_mask.nii.gz")[0] > 0
    except FileNotFoundError:
        return None
    if brain.sum() < 1000 or les.shape != brain.shape:
        return None
    ijk = np.argwhere(brain)
    xy = ijk[:, :2] * zm[:2]
    c = xy.mean(0)
    # Rotación axial: el eje largo del cerebro (PCA) debe quedar antero-posterior.
    w, v = np.linalg.eigh(np.cov((xy - c).T))
    ax = v[:, np.argmax(w)]
    if ax[1] < 0:
        ax = -ax
    th = np.arctan2(ax[0], ax[1])  # ángulo respecto del eje y
    if abs(th) > np.deg2rad(30):   # PCA poco fiable (cerebro casi redondo): no rotar
        th = 0.0
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])  # normalizado -> paciente
    rot = (xy - c) @ R  # coordenadas del paciente en el marco enderezado
    lo, hi = np.percentile(rot, [0.5, 99.5], axis=0)
    z0, z1 = ijk[:, 2].min(), ijk[:, 2].max()

    gx, gy, gz = np.meshgrid(*(np.arange(n) for n in G), indexing="ij")
    u = lo[0] + (gx + 0.5) / G[0] * (hi[0] - lo[0])
    w_ = lo[1] + (gy + 0.5) / G[1] * (hi[1] - lo[1])
    px = (u * R[0, 0] + w_ * R[0, 1] + c[0]) / zm[0]
    py = (u * R[1, 0] + w_ * R[1, 1] + c[1]) / zm[1]
    pz = z0 + (gz + 0.5) / G[2] * (z1 - z0 + 1) - 0.5
    coords = np.stack([px, py, pz])
    les_n = ndi.map_coordinates(les.astype(np.float32), coords, order=1, mode="constant") > 0.5
    b = dwi[brain]
    dwi_n = ndi.map_coordinates(np.clip((dwi - b.mean()) / (b.std() + 1e-6), -3, 5) * brain, coords, order=1)
    return les_n, dwi_n


mrs = pd.read_csv(ROOT / "text/datos/df_train_acv.csv", encoding="utf-8-sig").set_index("Paciente_id")["mRS egreso"]
ids, L, D = [], [], []
for pid in mrs.index:
    if pid in EXCLUIR:
        continue
    r = normalizar(pid)
    if r is None:
        print("sin datos:", pid)
        continue
    ids.append(pid); L.append(r[0]); D.append(r[1])
L = np.array(L, dtype=np.float32)
fondo = np.mean(D, axis=0)
y = (mrs.loc[ids].values > 2).astype(int)
vol = L.reshape(len(L), -1).sum(1)
ok = vol > 0
print(f"pacientes: {len(ids)} | mal desenlace: {y.sum()} | máscara vacía: {(~ok).sum()} (mal desenlace entre ellas: {y[~ok].sum()})")

# ---------- Mapas de frecuencia ----------
f0, f1 = L[y == 0].mean(0), L[y == 1].mean(0)
dif = ndi.gaussian_filter(f1 - f0, (1.5, 1.5, 0.5))
np.savez_compressed(OUT / "mapas.npz", frec_buen=f0, frec_mal=f1, diferencia=dif, fondo=fondo)
np.savez_compressed(OUT / "mascaras_normalizadas.npz", ids=np.array(ids), y=y, L=L.astype(bool))  # para ubicacion_mas_clinico.py

cortes = [5, 8, 11, 13, 15, 17, 19, 21]
fig, axes = plt.subplots(3, len(cortes), figsize=(2.0 * len(cortes), 7.6), facecolor="black")
vmax = max(np.percentile(f0[f0 > 0], 99), np.percentile(f1[f1 > 0], 99))
dmax = np.abs(dif).max()
filas = [(f0, "hot", 0, vmax, f"Buen desenlace\n(n={int((y == 0).sum())})"),
         (f1, "hot", 0, vmax, f"Mal desenlace\n(n={int(y.sum())})"),
         (dif, "RdBu_r", -dmax, dmax, "Diferencia\nmal − buen")]
sl = lambda a, z: np.fliplr(np.rot90(a[:, :, z]))  # convención radiológica
for r, (m, cmap, lo_, hi_, lab) in enumerate(filas):
    for c, z in enumerate(cortes):
        ax = axes[r, c]
        ax.imshow(sl(fondo, z), cmap="gray", vmin=-1, vmax=3, aspect=1.15)
        a = sl(m, z)
        alpha = np.clip(np.abs(a) / (hi_ * 0.35), 0, 1) * 0.85
        im = ax.imshow(a, cmap=cmap, vmin=lo_, vmax=hi_, alpha=alpha, aspect=1.15)
        ax.set_axis_off()
    axes[r, 0].text(-0.08, 0.5, lab, transform=axes[r, 0].transAxes, color="white", ha="right", va="center", fontsize=11)
    cb = fig.colorbar(im, ax=axes[r, :], fraction=0.012, pad=0.01)
    cb.ax.tick_params(colors="white", labelsize=8)
    cb.set_label("fracción de pacientes" if r < 2 else "diferencia", color="white", fontsize=8)
fig.suptitle("Frecuencia de lesión DWI por grupo (train, espacio normalizado aproximado; inferior → superior)",
             color="white", fontsize=12)
fig.savefig(OUT / "mapa_frecuencia.png", dpi=170, facecolor="black", bbox_inches="tight")

# ---------- Descriptores simples de ubicación ----------
gx, gy, gz = np.meshgrid(*(np.linspace(0, 1, n) for n in G), indexing="ij")
filas = []
for i in np.where(ok)[0]:
    m = L[i] > 0
    der = float((gx[m] > 0.5).mean())  # x > 0.5 = derecha del paciente (RAS)
    filas.append({"id": ids[i], "y": y[i], "vol": vol[i], "x": gx[m].mean(), "y_ap": gy[m].mean(), "z": gz[m].mean(),
                  "frac_derecha": der, "lateralidad_abs": abs(gx[m].mean() - 0.5), "bilateral": float(0.1 < der < 0.9)})
d = pd.DataFrame(filas)
d.to_csv(OUT / "descriptores_ubicacion.csv", index=False)
print("\nDescriptor (mediana buen | mal | p Mann-Whitney | AUC univariado)")
for col, nom in [("vol", "volumen (vóxeles norm.)"), ("x", "centro izq→der"), ("y_ap", "centro post→ant"), ("z", "centro inf→sup"),
                 ("lateralidad_abs", "distancia a línea media"), ("frac_derecha", "fracción en hemisferio der."),
                 ("bilateral", "bilateral (sí=1)")]:
    a, b = d.loc[d.y == 0, col], d.loc[d.y == 1, col]
    p = mannwhitneyu(a, b).pvalue
    auc = roc_auc_score(d.y, d[col])
    print(f"  {nom:30s} {a.median():8.3f} | {b.median():8.3f} | p={p:.3f} | AUC={max(auc, 1 - auc):.3f}")

# ---------- ¿La ubicación predice? Validación cruzada, mapa armado solo con los folds de train ----------
Lk, yk, vk = L[ok], y[ok], vol[ok]
flat = Lk.reshape(len(Lk), -1)


def cv_auc(labels, seed=0, reps=10):
    aucs = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=seed).split(flat, labels):
        m = Lk[tr][labels[tr] == 1].mean(0) - Lk[tr][labels[tr] == 0].mean(0)
        m = ndi.gaussian_filter(m, (2, 2, 0.7)).ravel()
        s = flat[te] @ m / flat[te].sum(1)  # promedio del mapa dentro de la lesión: no depende del volumen
        aucs.append(roc_auc_score(labels[te], s))
    return float(np.mean(aucs))


auc_loc = cv_auc(yk)
rng = np.random.default_rng(0)
nulos = np.array([cv_auc(rng.permutation(yk), seed=i, reps=2) for i in range(200)])
p_perm = (1 + (nulos >= auc_loc).sum()) / (1 + len(nulos))
auc_vol = roc_auc_score(yk, vk)
print(f"\nAUC de la UBICACIÓN (CV 5x10, n={len(yk)}, {yk.sum()} eventos): {auc_loc:.3f}")
print(f"  etiquetas permutadas: media {nulos.mean():.3f}, percentil 95 {np.percentile(nulos, 95):.3f} -> p = {p_perm:.3f}")
print(f"AUC del VOLUMEN en los mismos pacientes: {max(auc_vol, 1 - auc_vol):.3f}")
print("\nFigura y tablas en:", OUT)
