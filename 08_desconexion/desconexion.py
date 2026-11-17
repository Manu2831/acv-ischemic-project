"""Desconexión de tractos por la lesión: ¿agrega algo al modelo clínico (edad + NIHSS)?

1. Registra cada paciente al espacio MNI152 (afín sobre las máscaras de cerebro; sin T1 y con cortes de 6 mm no
   se intenta registro no lineal). Guarda las transformaciones en cache/ y un control de calidad (Dice + figura).
2. Lleva la máscara de lesión DWI a MNI y la cruza con el atlas probabilístico de tractos JHU (20 tractos).
3. Medida PRINCIPAL, fijada antes de mirar resultados: carga de lesión sobre el tracto corticoespinal (wCST-LL,
   mL ponderados por la probabilidad del tracto, el lado más afectado).
   Secundario/exploratorio: fracción desconectada de cada uno de los 11 tractos (lados agrupados).
4. Compara clínico vs clínico + desconexión con validación cruzada repetida y test de razón de verosimilitud.

Atlas (en acv-ischemic/atlas/, descargados de los repositorios públicos de FSL; uso académico):
  MNI152_T1_1mm_brain(.nii.gz, _mask), JHU-ICBM-tracts-prob-1mm.nii.gz
Uso: python desconexion.py            (la primera vez registra los 160 pacientes; después usa el caché)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.stats import chi2, mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
ATLAS = ROOT / "atlas"
CACHE = AQUI / "cache"
OUT = AQUI / "resultados"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

TRACTOS = {"Radiación talámica anterior": (0, 1), "Corticoespinal": (2, 3), "Cíngulo (giro cingulado)": (4, 5),
           "Cíngulo (hipocampo)": (6, 7), "Fórceps mayor": (8,), "Fórceps menor": (9,),
           "Fascículo fronto-occipital inferior": (10, 11), "Fascículo longitudinal inferior": (12, 13),
           "Fascículo longitudinal superior": (14, 15), "Fascículo uncinado": (16, 17),
           "Fascículo longitudinal superior (temporal)": (18, 19)}


def leer(p):
    p = Path(p)
    if not p.exists():
        p = p.with_suffix("")
    return sitk.ReadImage(str(p), sitk.sitkFloat32)


# ---------- Plantilla y atlas a 2 mm (misma grilla) ----------
fixed = sitk.BinShrink(leer(ATLAS / "MNI152_T1_1mm_brain.nii.gz"), [2, 2, 2])
fixed_mask = sitk.BinShrink(leer(ATLAS / "MNI152_T1_1mm_brain_mask.nii.gz"), [2, 2, 2]) > 0.5
jhu = nib.load(str(ATLAS / "JHU-ICBM-tracts-prob-1mm.nii.gz")).get_fdata(dtype=np.float32)
jhu = jhu.reshape(91, 2, 109, 2, 91, 2, 20).mean((1, 3, 5)) / 100.0  # bloques 2x2x2 = BinShrink
jhu = jhu.transpose(2, 1, 0, 3)  # (x,y,z,t) de nibabel -> (z,y,x,t) de SimpleITK
VOX_ML = 8 / 1000.0
fixed_np = sitk.GetArrayFromImage(fixed)
fmask_np = sitk.GetArrayFromImage(fixed_mask).astype(bool)


def registrar(pid, split):
    """Afín (12 parámetros) alineando la máscara de cerebro del paciente con la de MNI152.
    Se probó refinar con información mutua ADC <-> T1 y empeoraba (ventrículos desalineados, Dice 0,70-0,79)."""
    raw = ROOT / "acv_split" / split / "rawdata" / pid / "ses-01"
    adc = leer(raw / "adc.nii.gz")
    bm = sitk.Resample(leer(raw / "dwi_brain_mask.nii.gz"), adc, sitk.Transform(), sitk.sitkNearestNeighbor) > 0.5
    f_s = sitk.SmoothingRecursiveGaussian(sitk.Cast(fixed_mask, sitk.sitkFloat32), 4.0)
    m_s = sitk.SmoothingRecursiveGaussian(sitk.Cast(bm, sitk.sitkFloat32), 4.0)
    init = sitk.CenteredTransformInitializer(f_s, m_s, sitk.AffineTransform(3),
                                             sitk.CenteredTransformInitializerFilter.MOMENTS)
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMeanSquares()
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-5, 400)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([4, 2, 1])
    r.SetSmoothingSigmasPerLevel([2, 1, 0])
    r.SetInitialTransform(init, inPlace=False)
    t = r.Execute(f_s, m_s)
    if isinstance(t, sitk.CompositeTransform):
        t = t.GetNthTransform(0)
    return t, adc, bm


tab = pd.read_csv(ROOT / "text" / "datos" / "tabla_limpia.csv", encoding="utf-8-sig")
filas, mni_les, qc_img = [], {}, {}
for _, row in tab.iterrows():
    pid, split = row.Paciente_id, row.split
    der = ROOT / "acv_split" / split / "derivatives" / pid / "ses-01" / "ana_dwi_mask.nii.gz"
    if not (der.exists() or der.with_suffix("").exists()):
        continue
    tf = CACHE / f"{pid}.tfm"
    if tf.exists():
        t = sitk.ReadTransform(str(tf))
        raw = ROOT / "acv_split" / split / "rawdata" / pid / "ses-01"
        adc = leer(raw / "adc.nii.gz")
        bm = sitk.Resample(leer(raw / "dwi_brain_mask.nii.gz"), adc, sitk.Transform(), sitk.sitkNearestNeighbor) > 0.5
    else:
        print(".", end="", flush=True)  # registrando (sin IDs en la salida)
        t, adc, bm = registrar(pid, split)
        sitk.WriteTransform(t, str(tf))
    wm = sitk.GetArrayFromImage(sitk.Resample(sitk.Cast(bm, sitk.sitkFloat32), fixed, t, sitk.sitkLinear)) > 0.5
    dice = 2 * (wm & fmask_np).sum() / (wm.sum() + fmask_np.sum())
    les = leer(der)
    les_mni = sitk.GetArrayFromImage(sitk.Resample(les, fixed, t, sitk.sitkLinear, 0.0))
    les_mni = np.clip(les_mni, 0, 1) * fmask_np
    mni_les[pid] = les_mni.astype(np.float16)
    if len(qc_img) < 12:
        qc_img[pid] = sitk.GetArrayFromImage(sitk.Resample(adc, fixed, t, sitk.sitkLinear, 0.0))
    f = {"Paciente_id": pid, "split": split, "dice_registro": dice, "vol_lesion_ml": les_mni.sum() * VOX_ML}
    for nom, idx in TRACTOS.items():
        fr, ll = [], []
        for k in idx:
            p = jhu[..., k]
            ll.append((les_mni * p).sum() * VOX_ML)
            core = p >= 0.25
            fr.append((les_mni * core).sum() / max(core.sum(), 1))
        f[f"frac_{nom}"] = max(fr)
        f[f"carga_{nom}"] = max(ll)
    filas.append(f)

d = pd.DataFrame(filas).merge(tab, on=["Paciente_id", "split"])
d.to_csv(OUT / "desconexion_por_paciente.csv", index=False, encoding="utf-8-sig")
print(f"\nregistrados: {len(d)} | Dice máscara de cerebro vs MNI: mediana {d.dice_registro.median():.3f}, "
      f"mínimo {d.dice_registro.min():.3f} | "
      f"< 0,85: {(d.dice_registro < 0.85).sum()}")

# ---------- Figura de control de calidad del registro ----------
fig, axes = plt.subplots(3, 4, figsize=(12, 9), facecolor="black")
zc = 40
for ax, (pid, img) in zip(axes.ravel(), qc_img.items()):
    ax.imshow(np.flipud(fixed_np[zc]), cmap="gray")
    v = img[zc]
    ax.imshow(np.flipud(np.ma.masked_where(v <= 0, v)), cmap="magma", alpha=0.45,
              vmin=np.percentile(v[v > 0], 2) if (v > 0).any() else 0, vmax=np.percentile(v[v > 0], 98) if (v > 0).any() else 1)
    ax.contour(np.flipud(fmask_np[zc]), levels=[0.5], colors="cyan", linewidths=0.6)
    ax.set_title(f"Dice {d.set_index('Paciente_id').dice_registro.get(pid, np.nan):.2f}", color="white", fontsize=9)
    ax.set_axis_off()
fig.suptitle("Control de registro: ADC del paciente (color) sobre MNI152 (gris); cian = borde del cerebro MNI",
             color="white")
fig.savefig(OUT / "qc_registro.png", dpi=110, facecolor="black", bbox_inches="tight")

# ---------- Análisis ----------
d = d[d.excluido.isna()].dropna(subset=["Edad", "NIHSS ingreso", "mRS egreso"]).reset_index(drop=True)
d["y"] = (d["mRS egreso"] > 2).astype(int)
d["wcst"] = np.log1p(d["carga_Corticoespinal"] * 10)  # log: distribución muy asimétrica
CLIN = ["Edad", "NIHSS ingreso"]


def cv(df, cols, reps=50):
    X, y = df[cols].values, df.y.values
    out = []
    for r in range(reps):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            oof[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        out.append(roc_auc_score(y, oof))
    return np.array(out)


def lrt(df, extra):
    """Razón de verosimilitud: ¿la variable extra aporta sobre edad + NIHSS? (modelos sin penalización)"""
    y = df.y.values
    ll = []
    for cols in (CLIN, CLIN + extra):
        X = StandardScaler().fit_transform(df[cols].values)
        m = LogisticRegression(penalty=None, max_iter=5000).fit(X, y)
        p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        ll.append(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return chi2.sf(2 * (ll[1] - ll[0]), len(extra))


print("\n=== MEDIDA PRINCIPAL: carga de lesión sobre el corticoespinal (wCST-LL) ===")
for nom, sub in (("todos los pacientes", d), ("solo train", d[d.split == "train"])):
    a0, a1 = cv(sub, CLIN), cv(sub, CLIN + ["wcst"])
    g0, g1 = sub.loc[sub.y == 0, "carga_Corticoespinal"], sub.loc[sub.y == 1, "carga_Corticoespinal"]
    print(f"[{nom}] n={len(sub)}, {sub.y.sum()} eventos")
    print(f"  wCST-LL mediana (mL): buen {g0.median():.3f} | mal {g1.median():.3f} | p Mann-Whitney "
          f"{mannwhitneyu(g0, g1).pvalue:.3f} | AUC solo {roc_auc_score(sub.y, sub.wcst):.3f}")
    print(f"  edad + NIHSS            AUC {a0.mean():.3f} ± {a0.std():.3f}")
    print(f"  edad + NIHSS + wCST-LL  AUC {a1.mean():.3f} ± {a1.std():.3f} | diferencia {np.mean(a1 - a0):+.3f} "
          f"(mejor en {(a1 > a0).mean():.0%} de las repeticiones) | razón de verosimilitud p = {lrt(sub, ['wcst']):.3f}")
rho = spearmanr(d.wcst, d["NIHSS ingreso"])
print(f"  correlación wCST-LL vs NIHSS: rho = {rho.statistic:.2f} (p = {rho.pvalue:.1e})")
print(f"  pacientes con algo de corticoespinal tocado (> 0,01 mL): {(d.carga_Corticoespinal > 0.01).mean():.0%}")

print("\n=== EXPLORATORIO: fracción desconectada de cada tracto (todos; 11 pruebas, Bonferroni p < 0,0045) ===")
print(f"  {'tracto':44s} {'tocado buen|mal':>16s} {'AUC solo':>9s} {'p (+clínico)':>13s}")
res = []
for nom in TRACTOS:
    c = f"frac_{nom}"
    d["_x"] = np.log1p(d[c] * 100)
    toca = d[c] > 0.01
    auc = roc_auc_score(d.y, d._x)
    res.append((nom, toca[d.y == 0].mean(), toca[d.y == 1].mean(), auc, lrt(d, ["_x"])))
for nom, a, b, auc, p in sorted(res, key=lambda x: x[-1]):
    print(f"  {nom:44s} {a:6.0%} | {b:5.0%} {auc:9.3f} {p:13.3f}")
d["_x"] = 0
X11 = [f"frac_{n}" for n in TRACTOS]
for c in X11:
    d["l_" + c] = np.log1p(d[c] * 100)
a0, a11 = cv(d, CLIN), cv(d, CLIN + ["l_" + c for c in X11])
print(f"\n  edad + NIHSS + perfil de 11 tractos: AUC {a11.mean():.3f} (clínico {a0.mean():.3f}, diferencia {np.mean(a11 - a0):+.3f})")

# ---------- Figuras ----------
L = {k: v.astype(np.float32) for k, v in mni_les.items()}
ids_b, ids_m = d.Paciente_id[d.y == 0], d.Paciente_id[d.y == 1]
f0, f1 = np.mean([L[i] for i in ids_b], 0), np.mean([L[i] for i in ids_m], 0)
cst = np.maximum(jhu[..., 2], jhu[..., 3])
cortes = [22, 28, 34, 40, 46, 52, 58, 64]
fig, axes = plt.subplots(3, len(cortes), figsize=(2.1 * len(cortes), 8), facecolor="black")
vmax = max(f0.max(), f1.max()) * 0.8
dif = f1 - f0
dm = np.abs(dif).max() * 0.8
for r, (m, cmap, lo, hi, lab) in enumerate([(f0, "hot", 0, vmax, f"Buen desenlace\n(n={len(ids_b)})"),
                                           (f1, "hot", 0, vmax, f"Mal desenlace\n(n={len(ids_m)})"),
                                           (dif, "RdBu_r", -dm, dm, "Diferencia\nmal − buen")]):
    for c, z in enumerate(cortes):
        ax = axes[r, c]
        ax.imshow(np.flipud(fixed_np[z]), cmap="gray")
        a = np.flipud(m[z])
        ax.imshow(a, cmap=cmap, vmin=lo, vmax=hi, alpha=np.clip(np.abs(a) / (hi * 0.3), 0, 1) * 0.85)
        ax.contour(np.flipud(cst[z]), levels=[0.25], colors="#4dd2ff", linewidths=0.8)
        ax.set_axis_off()
    axes[r, 0].text(-0.08, 0.5, lab, transform=axes[r, 0].transAxes, color="white", ha="right", va="center")
fig.suptitle("Frecuencia de lesión en MNI152 (todos los pacientes; celeste = corticoespinal JHU p ≥ 0,25; "
             "orientación radiológica)", color="white")
fig.savefig(OUT / "mapa_frecuencia_mni.png", dpi=150, facecolor="black", bbox_inches="tight")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].boxplot([d.loc[d.y == 0, "wcst"], d.loc[d.y == 1, "wcst"]], tick_labels=["mRS ≤ 2", "mRS > 2"])
axes[0].set_ylabel("log(1 + 10 × wCST-LL [mL])")
axes[0].set_title("Carga sobre el corticoespinal")
axes[1].scatter(d["NIHSS ingreso"], d.wcst, c=d.y, cmap="coolwarm", s=18)
axes[1].set_xlabel("NIHSS al ingreso")
axes[1].set_ylabel("log(1 + 10 × wCST-LL)")
axes[1].set_title(f"wCST-LL vs NIHSS (rho = {rho.statistic:.2f}); rojo = mal desenlace")
fig.tight_layout()
fig.savefig(OUT / "carga_cst.png", dpi=130)
print("\nresultados en", OUT)
