"""Cruce de la desconexión de Tractotron (68 tractos del atlas BCB) con los datos clínicos, de tratamiento y territorio.

Entradas:
  tractotron_probabilidad.csv / tractotron_proporcion.csv  (de parsear_log.py)
  ../datos/tabla_limpia.csv                                (edad, NIHSS, mRS corregidos)
  dataset_report.csv (ruta en ACV_REPORTE o argumento 1)   (territorio, tratamiento, TICI, volumen, mRS 90 d)
  ../08_desconexion/resultados/desconexion_por_paciente.csv (nuestro registro + JHU, para control de calidad)
Uso: python analizar.py <dataset_report.csv>
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, fisher_exact, mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).resolve().parent
TXT = AQUI.parent
OUT = AQUI / "resultados"
OUT.mkdir(exist_ok=True)

prob = pd.read_csv(AQUI / "tractotron_probabilidad.csv", index_col=0)
prop = pd.read_csv(AQUI / "tractotron_proporcion.csv", index_col=0)
TR = list(prob.columns)
clin = pd.read_csv(TXT / "datos/tabla_limpia.csv", encoding="utf-8-sig").set_index("Paciente_id")
rep = pd.read_csv(sys.argv[1]).set_index("patient")
nos = pd.read_csv(TXT / "08_desconexion/resultados/desconexion_por_paciente.csv", encoding="utf-8-sig").set_index("Paciente_id")

d = clin[clin.excluido.isna()].dropna(subset=["Edad", "NIHSS ingreso", "mRS egreso"]).copy()
d["y"] = (d["mRS egreso"] > 2).astype(int)
r = rep.reindex(d.index)
d["vol_ml"] = r.lesion_volume_ml
d["log_vol"] = np.log1p(d.vol_ml)
d["gran_vaso"] = ((r.at_internal_carotid + r.at_mca_m1 + r.at_basilar) > 0).astype(int)
d["peq_vaso"] = r.at_small_vessel
d["post"] = ((r.at_pca + r.at_vertebral + r.at_basilar) > 0).astype(int)
d["trombectomia"] = r.treatment_mt
d["trombolisis_iv"] = r.treatment_pt_iv
d["conservador"] = r.treatment_conservative
d["izq"] = r.laterality.astype(str).str.contains("2").astype(int)  # 2 = izquierdo (ver QC abajo)
d["mrs90"] = r.mrs_90_days
P = prob.reindex(d.index)
Q = prop.reindex(d.index)
falta = P.isna().all(1)
print(f"Pacientes analizables: {len(d)} ({d.y.sum()} mal desenlace); sin Tractotron: {falta.sum()}")
d, P, Q = d[~falta], P[~falta], Q[~falta]
DESC = (P > 0.5).astype(int)  # convención Tractotron: tracto desconectado si probabilidad > 0.5
d["n_desc"] = DESC.sum(1)
d["log_n_desc"] = np.log1p(d.n_desc)
cst = [c for c in TR if c.startswith("Cortico_Spinal")]
d["cst_desc"] = DESC[cst].max(1)
d["cst_prop"] = Q[cst].max(1)
d["carga_total"] = np.log1p(Q.sum(1) * 100)

# ---------- 1. Control de calidad ----------
print("\n=== 1. CONTROL DE CALIDAD (Tractotron vs nuestro registro / reporte) ===")
cero = (P.max(1) == 0)
print(f"  pacientes con lesión que no toca ningún tracto (prob máx = 0): {cero.sum()} | volumen mediano de ellos {d.vol_ml[cero].median():.2f} mL")
n = nos.reindex(d.index)
ok = n.frac_Corticoespinal.notna()
rho = spearmanr(d.cst_prop[ok], n.frac_Corticoespinal[ok])
print(f"  CST: proporción Tractotron (BCB) vs fracción JHU de nuestro registro: rho = {rho.statistic:.2f} (n={ok.sum()})")
rho = spearmanr(d.vol_ml[ok], n.vol_lesion_ml[ok])
print(f"  volumen: reporte vs nuestra máscara en MNI: rho = {rho.statistic:.2f}")
# lateralidad: el reporte dice izq/der; ¿coincide con el hemisferio de los tractos desconectados?
izqT = DESC[[c for c in TR if c.lower().endswith("left")]].sum(1)
derT = DESC[[c for c in TR if c.lower().endswith("right")]].sum(1)
lado = np.sign(izqT - derT)
m = lado != 0
lat = r.laterality.reindex(d.index).astype(str)
print("  lateralidad del reporte vs hemisferio con más tractos desconectados (Tractotron):")
print(pd.crosstab(lat[m], lado[m].map({1: "izq", -1: "der"})).to_string().replace("\n", "\n    "))

# ---------- 2. Validez: ¿la desconexión se relaciona con el déficit? ----------
print("\n=== 2. VALIDEZ CLÍNICA ===")
g0, g1 = d.loc[d.cst_desc == 0, "NIHSS ingreso"], d.loc[d.cst_desc == 1, "NIHSS ingreso"]
print(f"  NIHSS mediano: CST intacto {g0.median():.0f} (n={len(g0)}) | CST desconectado {g1.median():.0f} (n={len(g1)}) "
      f"| p = {mannwhitneyu(g0, g1).pvalue:.1e}")
rho = spearmanr(d.n_desc, d["NIHSS ingreso"])
print(f"  nº de tractos desconectados vs NIHSS: rho = {rho.statistic:.2f} (p = {rho.pvalue:.1e})")
rho = spearmanr(d.n_desc, d.vol_ml)
print(f"  nº de tractos desconectados vs volumen: rho = {rho.statistic:.2f}")
print(f"  mediana de tractos desconectados: buen {d.n_desc[d.y == 0].median():.0f} | mal {d.n_desc[d.y == 1].median():.0f} "
      f"| p = {mannwhitneyu(d.n_desc[d.y == 0], d.n_desc[d.y == 1]).pvalue:.1e} | AUC solo {roc_auc_score(d.y, d.n_desc):.3f}")

CLIN = ["Edad", "NIHSS ingreso"]


def lrt(df, extra, base=CLIN):
    y = df.y.values
    ll = []
    for cols in (base, base + extra):
        X = StandardScaler().fit_transform(df[cols].values)
        mdl = LogisticRegression(penalty=None, max_iter=5000).fit(X, y)
        p = np.clip(mdl.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        ll.append(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return chi2.sf(2 * (ll[1] - ll[0]), len(extra))


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


# ---------- 3. Tracto por tracto ----------
print("\n=== 3. TRACTO POR TRACTO (desconectado = prob > 0.5; solo tractos desconectados en ≥ 8 pacientes) ===")
filas = []
for t in TR:
    k = DESC[t]
    if k.sum() < 8:
        continue
    a, b = k[d.y == 0], k[d.y == 1]
    odds, pf = fisher_exact([[b.sum(), len(b) - b.sum()], [a.sum(), len(a) - a.sum()]])
    d["_x"] = k
    filas.append({"tracto": t, "n_desc": int(k.sum()), "desc_buen": a.mean(), "desc_mal": b.mean(), "OR": odds,
                  "p_fisher": pf, "p_ajust_edad_nihss": lrt(d, ["_x"])})
T = pd.DataFrame(filas)
T["q_fisher"], T["q_ajust"] = bh(T.p_fisher), bh(T.p_ajust_edad_nihss)
T = T.sort_values("p_ajust_edad_nihss")
T.to_csv(OUT / "tractos_vs_desenlace.csv", index=False)
print(f"  tractos evaluados: {len(T)}")
print(f"  {'tracto':44s} {'n':>3s} {'buen':>5s} {'mal':>5s} {'OR':>5s} {'p crudo':>8s} {'q':>6s} {'p ajust':>8s} {'q ajust':>8s}")
for _, x in T.head(12).iterrows():
    print(f"  {x.tracto:44s} {x.n_desc:3d} {x.desc_buen:5.0%} {x.desc_mal:5.0%} {x.OR:5.1f} {x.p_fisher:8.3f} {x.q_fisher:6.3f} "
          f"{x.p_ajust_edad_nihss:8.3f} {x.q_ajust:8.3f}")
print(f"  tractos con q crudo < 0.05: {(T.q_fisher < 0.05).sum()} | con q ajustado < 0.05: {(T.q_ajust < 0.05).sum()}")

# ---------- 4. Modelos predictivos ----------
print("\n=== 4. ¿MEJORA AL CLÍNICO? (validación cruzada 5x50, mismas particiones para todos) ===")
d["_x"] = 0
Xtr = np.log1p(Q[TR].fillna(0).values * 100)


def cv(df, cols, reps=50, tractos=False, C=1.0):
    y = df.y.values
    X = df[cols].values if cols else np.zeros((len(y), 0))
    if tractos:
        X = np.hstack([X, Xtr])
    out = []
    for rr in range(reps):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rr).split(X, y):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
            oof[te] = mdl.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        out.append(roc_auc_score(y, oof))
    return np.array(out)


base = cv(d, CLIN)
print(f"  {'modelo':52s} {'AUC':>6s} {'Δ':>7s} {'mejor %':>8s} {'p LRT':>7s}")
print(f"  {'edad + NIHSS':52s} {base.mean():6.3f}")
modelos = [
    ("+ nº tractos desconectados", ["log_n_desc"]),
    ("+ corticoespinal desconectado", ["cst_desc"]),
    ("+ carga total de desconexión", ["carga_total"]),
    ("+ volumen (reporte)", ["log_vol"]),
    ("+ gran vaso (ACI / M1 / basilar)", ["gran_vaso"]),
    ("+ pequeño vaso", ["peq_vaso"]),
    ("+ territorio posterior", ["post"]),
    ("+ hemisferio izquierdo", ["izq"]),
    ("+ trombectomía", ["trombectomia"]),
    ("+ trombólisis IV", ["trombolisis_iv"]),
    ("+ gran vaso + trombectomía + nº desconectados", ["gran_vaso", "trombectomia", "log_n_desc"]),
]
res = []
for nom, extra in modelos:
    a = cv(d, CLIN + extra)
    res.append((nom, a.mean(), np.mean(a - base), (a > base).mean(), lrt(d, extra)))
    print(f"  {nom:52s} {a.mean():6.3f} {np.mean(a - base):+7.3f} {(a > base).mean():8.0%} {res[-1][-1]:7.3f}")
for C in (0.05, 0.2):
    a = cv(d, CLIN, tractos=True, C=C)
    print(f"  {'+ perfil 68 tractos (L2, C=' + str(C) + ')':52s} {a.mean():6.3f} {np.mean(a - base):+7.3f} {(a > base).mean():8.0%}")
a = cv(d, [], tractos=True, C=0.05)
print(f"  {'solo 68 tractos, sin clínica (L2, C=0.05)':52s} {a.mean():6.3f}")
a = cv(d, ["log_n_desc"])
print(f"  {'solo nº tractos desconectados':52s} {a.mean():6.3f}")

# ---------- 5. Pacientes donde el NIHSS se queda corto ----------
print("\n=== 5. NIHSS BAJO (≤ 5): ¿la desconexión explica a los que salen mal? ===")
s = d[d["NIHSS ingreso"] <= 5]
print(f"  n = {len(s)}, mal desenlace = {s.y.sum()} ({s.y.mean():.0%})")
if s.y.sum() >= 3:
    for col, nom in [("n_desc", "nº tractos desconectados"), ("vol_ml", "volumen mL"), ("Edad", "edad")]:
        a, b = s.loc[s.y == 0, col], s.loc[s.y == 1, col]
        print(f"    {nom:26s} buen {a.median():6.1f} | mal {b.median():6.1f} | p = {mannwhitneyu(a, b).pvalue:.3f}")
    for col, nom in [("cst_desc", "CST desconectado"), ("gran_vaso", "gran vaso"), ("trombectomia", "trombectomía")]:
        a, b = s.loc[s.y == 0, col], s.loc[s.y == 1, col]
        print(f"    {nom:26s} buen {a.mean():5.0%} | mal {b.mean():5.0%} | p Fisher = "
              f"{fisher_exact([[b.sum(), len(b) - b.sum()], [a.sum(), len(a) - a.sum()]])[1]:.3f}")
    # (sin tabla por paciente: la salida no debe mostrar IDs)

# ---------- 6. mRS a 90 días ----------
print("\n=== 6. mRS A 90 DÍAS ===")
s = d[d.mrs90.notna() & (d.mrs90 >= 0)].copy()
s["y"] = (s.mrs90 > 2).astype(int)
print(f"  con mRS 90 d: {len(s)}, mal desenlace a 90 d: {s.y.sum()}")
if s.y.sum() >= 10:
    b90 = cv(s, CLIN)
    for nom, extra in [("+ nº tractos desconectados", ["log_n_desc"]), ("+ corticoespinal desconectado", ["cst_desc"])]:
        a = cv(s, CLIN + extra)
        print(f"  edad+NIHSS {b90.mean():.3f} -> {nom} {a.mean():.3f} (Δ {np.mean(a - b90):+.3f}, p LRT {lrt(s, extra):.3f})")

d.drop(columns="_x").join(DESC.add_prefix("desc_")).to_csv(OUT / "tabla_cruzada.csv", encoding="utf-8-sig")
print("\nTablas en:", OUT)
