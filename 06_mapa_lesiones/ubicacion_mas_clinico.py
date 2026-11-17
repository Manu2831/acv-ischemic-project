"""¿La zona de la lesión mejora al modelo clínico (edad + NIHSS)?

Usa las máscaras normalizadas de mapa_lesiones.py (solo TRAIN) y la tabla limpia. En cada fold el mapa
"mal − buen desenlace" se arma solo con los pacientes de entrenamiento del fold.
Uso: python mapa_lesiones.py && python ubicacion_mas_clinico.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).resolve().parent
z = np.load(AQUI / "resultados" / "mascaras_normalizadas.npz")
tab = pd.read_csv(AQUI.parent / "datos" / "tabla_limpia.csv", encoding="utf-8-sig").set_index("Paciente_id")
tab = tab[tab.excluido.isna()].dropna(subset=["Edad", "NIHSS ingreso", "mRS egreso"])
keep = np.array([i in tab.index for i in z["ids"]])
ids, L = z["ids"][keep], z["L"][keep].astype(np.float32)
t = tab.loc[ids]
y = (t["mRS egreso"].values > 2).astype(int)
clin = t[["Edad", "NIHSS ingreso"]].values
vol = np.log1p(L.reshape(len(L), -1).sum(1))
Lf = np.maximum(L, L[:, ::-1])  # hemisferios plegados: la lesión cuenta igual a izquierda o derecha
print(f"pacientes (train, tabla limpia): {len(y)} | mal desenlace: {y.sum()}")


def puntaje(M, tr, te):
    m = M[tr][y[tr] == 1].mean(0) - M[tr][y[tr] == 0].mean(0)
    m = ndi.gaussian_filter(m, (2, 2, 0.7)).ravel()
    f = M.reshape(len(M), -1)
    n = np.maximum(f.sum(1), 1)
    return (f[tr] @ m / n[tr])[:, None], (f[te] @ m / n[te])[:, None]


MODELOS = ["clínico (edad + NIHSS)", "clínico + volumen", "clínico + zona", "clínico + zona (hemisferios plegados)",
           "clínico + volumen + zona", "solo zona"]
res = {m: [] for m in MODELOS}
for rep in range(30):
    oof = {m: np.zeros(len(y)) for m in MODELOS}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(clin, y):
        ztr, zte = puntaje(L, tr, te)
        ftr, fte = puntaje(Lf, tr, te)
        X = {"clínico (edad + NIHSS)": (clin[tr], clin[te]),
             "clínico + volumen": (np.c_[clin[tr], vol[tr]], np.c_[clin[te], vol[te]]),
             "clínico + zona": (np.c_[clin[tr], ztr], np.c_[clin[te], zte]),
             "clínico + zona (hemisferios plegados)": (np.c_[clin[tr], ftr], np.c_[clin[te], fte]),
             "clínico + volumen + zona": (np.c_[clin[tr], vol[tr], ztr], np.c_[clin[te], vol[te], zte]),
             "solo zona": (ztr, zte)}
        for m, (a, b) in X.items():
            oof[m][te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(a, y[tr]).predict_proba(b)[:, 1]
    for m in MODELOS:
        res[m].append(roc_auc_score(y, oof[m]))
base = np.array(res[MODELOS[0]])
print("\nAUC en validación cruzada repetida (5 folds x 30):")
for m in MODELOS:
    a = np.array(res[m])
    extra = "" if m == MODELOS[0] else f" | diferencia vs clínico {np.mean(a - base):+.3f} (mejor en {(a > base).mean():.0%} de las repeticiones)"
    print(f"  {m:40s} {a.mean():.3f} ± {a.std():.3f}{extra}")

# ---- Regiones gruesas: ¿qué fracción de cada grupo tiene lesión ahí? ----
G = L.shape[1:]
gx, gy, gz = np.meshgrid(*(np.linspace(0, 1, n) for n in G), indexing="ij")
REG = {"hemisferio izquierdo": gx < 0.5, "hemisferio derecho": gx > 0.5,
       "profunda (ganglios basales, cápsula, tálamo)": (abs(gx - 0.5) < 0.22) & (abs(gy - 0.5) < 0.17) & (gz > 0.35) & (gz < 0.7),
       "fosa posterior (tronco, cerebelo)": (gz < 0.3) & (gy < 0.55),
       "frontal": (gy > 0.62) & (gz > 0.35), "parieto-occipital": (gy < 0.38) & (gz > 0.35)}
print("\nPacientes con lesión en cada región (>= 10 vóxeles): buen | mal desenlace | p de Fisher")
for nom, r in REG.items():
    toca = (L * r).reshape(len(L), -1).sum(1) >= 10
    a, b = toca[y == 0], toca[y == 1]
    p = fisher_exact([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])[1]
    print(f"  {nom:46s} {a.mean():5.0%} | {b.mean():5.0%} | p={p:.2f}")

# ---- Regiones gruesas como variables (no se ajusta ningún mapa: sin riesgo de sobreajuste por el mapa) ----
R = np.stack([((L * r).reshape(len(L), -1).sum(1) >= 10).astype(float) for r in REG.values()], axis=1)
nombres = list(REG)
variantes = {"clínico + región profunda": np.c_[clin, R[:, [2]]], "clínico + frontal": np.c_[clin, R[:, [4]]],
             "clínico + las 6 regiones": np.c_[clin, R]}
print("\nRegiones gruesas agregadas al clínico (CV 5 folds x 30):")
for nom, X in variantes.items():
    aucs = []
    for rep in range(30):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            oof[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, oof))
    a = np.array(aucs)
    print(f"  {nom:40s} {a.mean():.3f} ± {a.std():.3f} | diferencia vs clínico {np.mean(a - base):+.3f}")
