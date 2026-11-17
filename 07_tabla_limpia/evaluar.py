"""Compara la regresión logística clínica con los datos del CSV original y con la tabla limpia.

Protocolo principal: validación cruzada repetida (5 folds x 50) sobre todos los pacientes usables.
También reporta el protocolo anterior (entrenar en train, medir en val+test) para comparar con el 0,855.
Uso: python evaluar.py   (antes: python construir_tabla.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
t = pd.read_csv(Path(__file__).resolve().parent / "tabla_limpia.csv", encoding="utf-8-sig")
MODELOS = {"edad + NIHSS": ["edad", "nihss"], "5 variables (edad, NIHSS, HTA, DM, FA)": ["edad", "nihss", "hta", "dm", "fa"]}


def datos(version):
    d = t.copy()
    if version != "limpia":
        for c in ("edad", "nihss", "hta", "dm", "fa"):
            d[c] = d[c + "_csv"]
        if version == "solo_120":
            d.loc[d.id == "ACV-040", "nihss"] = 10
    else:
        d = d[d.excluido.isna()]
    return d


def cv(d, cols, reps=50):
    d = d.dropna(subset=cols + ["mal_desenlace"])
    X, y = d[cols].values, d.mal_desenlace.values.astype(int)
    aucs = []
    for r in range(reps):
        oof = np.zeros(len(y))
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=r).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return np.mean(aucs), np.std(aucs), len(y), int(y.sum())


def ext(d, cols):
    d = d.dropna(subset=cols + ["mal_desenlace"])
    tr, te = d[d.split == "train"], d[d.split != "train"]
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[cols].values, tr.mal_desenlace.astype(int))
    return roc_auc_score(te.mal_desenlace.astype(int), m.predict_proba(te[cols].values)[:, 1]), len(te), int(te.mal_desenlace.sum())


VERS = {"original": "CSV original", "solo_120": "CSV + solo NIHSS 120→10", "limpia": "Tabla limpia"}
for nom, cols in MODELOS.items():
    print(f"\n== Modelo: {nom}")
    for v, lab in VERS.items():
        d = datos(v)
        a, s, n, e = cv(d, cols)
        ae, ne, ee = ext(d, cols)
        print(f"  {lab:26s} CV repetida AUC {a:.3f} ± {s:.3f} (n={n}, {e} eventos) | train→ext AUC {ae:.3f} (n={ne}, {ee} eventos)")
