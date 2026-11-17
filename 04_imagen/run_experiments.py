"""
Corre la grilla de experimentos en orden de prioridad. Es reanudable: si un
experimento ya tiene metrics.json se salta, así que se puede cortar y relanzar.

Normalmente no se llama directo: ejecutar_todo.py prepara los encoders y después lo llama.

Uso (desde text/):
  python 04_imagen/run_experiments.py --suite pedido                          # encoder de lesión
  python 04_imagen/run_experiments.py --suite pedido --encoders dinov2        # DINOv2 preentrenado
  python 04_imagen/run_experiments.py --suite todo --encoders lesion dinov2   # todo, ambos
  python 04_imagen/run_experiments.py --suite fusiones --quick
  python 04_imagen/run_experiments.py --suite todo --dim 64 --results_dir 04_imagen/results_dim64
      (cualquier argumento de train.py se reenvía a todos los experimentos)
  python 04_imagen/run_experiments.py --resumen                               # solo imprime la tabla
  python 04_imagen/run_experiments.py --resumen --results_dir C:/ruta/resultados_kaggle
      (no entrena: rearma las tablas y, si los resultados son de una versión vieja, recalcula todas
       las métricas desde predictions.csv)

Estructura de 04_imagen/results/: ver train.finalize_config. Tablas:
  resumen.csv             una fila por experimento: AUC de cada split + todas las métricas en ext
                          + ΔAUC pareado contra la regresión logística tabular
  metricas_por_split.csv  una fila por experimento x split (oof/val/test/ext), cada métrica con IC95%
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import train as T

IMG = {"adc": ["--images", "adc"], "dwi": ["--images", "dwi"], "adc_dwi": ["--images", "adc", "dwi"]}
TAB = ["--tabular"]

SUITES = {
    # lo pedido: ADC solo, DWI solo, ADC+tab, DWI+tab, ADC+DWI+tab (fusión concat) + referencias
    "pedido": [
        TAB,
        IMG["adc"],
        IMG["dwi"],
        IMG["adc"] + TAB,
        IMG["dwi"] + TAB,
        IMG["adc_dwi"] + TAB,
        IMG["adc_dwi"],
    ],
    # las otras tres fusiones sobre la combinación completa
    "fusiones": [IMG["adc_dwi"] + TAB + ["--fusion", f] for f in ("film", "cross_attn", "gated")],
    # las otras fusiones por modalidad individual
    "fusiones_modalidad": [IMG[m] + TAB + ["--fusion", f] for m in ("adc", "dwi")
                           for f in ("film", "cross_attn", "gated")],
    # controles: ¿la imagen aporta info del paciente correcto? ¿o todo lo hace la máscara?
    "controles": [
        IMG["adc_dwi"] + TAB + ["--control", "shuffle_image"],
        IMG["adc_dwi"] + ["--control", "shuffle_image"],
        IMG["adc_dwi"] + ["--no_lesion_mask"],
    ],
}
SUITES["todo"] = SUITES["pedido"] + SUITES["fusiones"] + SUITES["fusiones_modalidad"] + SUITES["controles"]


def logreg_dir(results_dir):
    return Path(results_dir) / "00_referencia_tabular" / "regresion_logistica"


def logreg_baseline(results_dir, seed=42, folds=5, n_boot=2000):
    """Referencia barata: regresión logística tabular con el mismo protocolo."""
    out_dir = logreg_dir(results_dir)
    if (out_dir / "predictions.csv").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    df = T.with_images(T.load_tabular())
    d = {s: df[df.split == s] for s in T.SPLITS}
    X, y = d["train"][T.FEATURES].values, d["train"].y.values
    oof, models = np.zeros(len(y)), []
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        ratio = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        clf = LogisticRegression(class_weight={0: 1.0, 1: ratio ** 0.5}, max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        oof[va] = clf.predict_proba(sc.transform(X[va]))[:, 1]
        models.append((sc, clf))
    probs = {"train": oof}
    for s in ("val", "test"):
        probs[s] = np.mean([c.predict_proba(sc.transform(d[s][T.FEATURES].values))[:, 1] for sc, c in models], axis=0)
    ys = {s: d[s].y.values for s in T.SPLITS}
    thr = T.youden_threshold(y, oof)
    metrics = {"name": "TAB_regresion_logistica", "threshold_youden_oof": thr, "minutes": 0.0}
    metrics.update(T.evaluate_splits(ys, probs, thr, n_boot, seed))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    pd.concat([pd.DataFrame({"Paciente_id": d[s].Paciente_id, "split": s, "y": ys[s], "prob": probs[s]})
               for s in T.SPLITS]).to_csv(out_dir / "predictions.csv", index=False)


METRICS = {"auc": "AUC", "auprc": "AUPRC", "acc": "Acc", "bal_acc": "BalAcc", "sens": "Sens", "spec": "Spec",
           "ppv": "PPV", "npv": "NPV", "f1": "F1", "mcc": "MCC", "brier": "Brier"}
SHOW = ["experimento", "AUC_oof", "AUC_ext", "IC_AUC_ext", "AUPRC_ext", "F1_ext", "Acc_ext", "BalAcc_ext",
        "Sens_ext", "Spec_ext", "PPV_ext", "NPV_ext", "MCC_ext", "dAUC_vs_logreg", "IC_dAUC", "conf_imagen", "min"]


def rescore(d):
    """Resultados de una versión vieja (menos métricas): las recalcula desde predictions.csv, sin reentrenar."""
    m = json.loads((d / "metrics.json").read_text())
    complete = all(k in m[key]["ci95"] for key in ("oof", "val", "test", "ext") for k in T.CI_METRICS)
    if complete or not (d / "predictions.csv").exists():
        return m
    print(f"  recalculando métricas: {m['name']}", flush=True)
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    pr = pd.read_csv(d / "predictions.csv")
    ys = {s: pr.loc[pr.split == s, "y"].values for s in T.SPLITS}
    probs = {s: pr.loc[pr.split == s, "prob"].values for s in T.SPLITS}
    m.update(T.evaluate_splits(ys, probs, m["threshold_youden_oof"], cfg.get("n_boot", 2000), cfg.get("seed", 42)))
    (d / "metrics.json").write_text(json.dumps(m, indent=1))
    return m


def delta_vs_logreg(d, ref):
    """ΔAUC en ext (val+test) contra la regresión logística tabular, pareado por paciente."""
    if ref is None or not (d / "predictions.csv").exists() or (d / "predictions.csv").samefile(ref):
        return {}
    pr = pd.read_csv(d / "predictions.csv").merge(pd.read_csv(ref), on=["Paciente_id", "split"], suffixes=("", "_ref"))
    pr = pr[pr.split.isin(["val", "test"])]
    delta, (lo, hi) = T.paired_delta_auc(pr.y.values, pr.prob.values, pr.prob_ref.values)
    return {"dAUC_vs_logreg": round(delta, 3), "IC_dAUC": f"[{lo:+.2f}, {hi:+.2f}]"}


def summarize(results_dir, exp_dirs):
    """Tablas con TODO lo que haya en results_dir (no solo la suite actual), agrupadas por carpeta."""
    results_dir = Path(results_dir)
    found = sorted({p.parent for p in results_dir.rglob("metrics.json")})
    order = {Path(d): i for i, d in enumerate(exp_dirs)}
    found.sort(key=lambda d: (str(d.relative_to(results_dir).parent), order.get(d, len(order)), str(d)))
    ref = logreg_dir(results_dir) / "predictions.csv"
    ref = ref if ref.exists() else None
    rows, per_split = [], []
    for d in found:
        m = rescore(d)
        grupo = str(d.relative_to(results_dir).parent).replace("\\", "/")
        thr = round(m["threshold_youden_oof"], 3)
        row = {"grupo": grupo, "experimento": m["name"], **{f"AUC_{k}": round(m[k]["auc"], 3) for k in ("oof", "val", "test")}}
        for key in ("oof", "val", "test", "ext"):
            mk = m[key]
            split_row = {"grupo": grupo, "experimento": m["name"], "split": key,
                         "n": mk["tp"] + mk["fp"] + mk["tn"] + mk["fn"], "positivos": mk["tp"] + mk["fn"], "umbral": thr}
            for k, col in METRICS.items():
                lo, hi = mk["ci95"].get(k, [np.nan, np.nan])
                split_row.update({col: round(mk.get(k, np.nan), 3), f"{col}_lo": round(lo, 3), f"{col}_hi": round(hi, 3)})
                if key == "ext":
                    row.update({f"{col}_ext": round(mk.get(k, np.nan), 3), f"IC_{col}_ext": f"[{lo:.2f}-{hi:.2f}]"})
            split_row.update({c.upper(): mk[c] for c in ("tp", "fp", "tn", "fn")})
            per_split.append(split_row)
        row.update({f"{c.upper()}_ext": m["ext"][c] for c in ("tp", "fp", "tn", "fn")})
        row.update(delta_vs_logreg(d, ref))
        row.update({"conf_imagen": round(m.get("gate_mean", {}).get("imagen", np.nan), 2), "umbral": thr,
                    "min": round(m.get("minutes", 0), 1)})
        rows.append(row)
    if not rows:
        print("Todavía no hay resultados.")
        return
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "resumen.csv", index=False)
    pd.DataFrame(per_split).to_csv(results_dir / "metricas_por_split.csv", index=False)
    parts = table["grupo"].str.split("/")
    label = [" / ".join(p[:2]) if p[0] in T.ENCODER_FOLDER.values() else p[0] for p in parts]
    table["_label"] = label
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for g in dict.fromkeys(label):
            print(f"\n=== {g} ===")
            print(table.loc[table["_label"] == g, [c for c in SHOW if c in table]].to_string(index=False))
    print("\n(_ext = val+test juntos, n=48; IC = IC95% bootstrap; umbral = Youden en OOF de train, fijo para val/test;"
          "\n dAUC_vs_logreg = AUC del experimento - AUC de la regresión logística tabular en los mismos pacientes:"
          "\n si su IC incluye 0, la diferencia no es concluyente; conf_imagen = compuerta media del modo gated)")
    print(f"Tablas: {results_dir / 'resumen.csv'}\n        {results_dir / 'metricas_por_split.csv'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="pedido", choices=list(SUITES))
    p.add_argument("--encoders", nargs="+", default=["lesion"], choices=list(T.ENCODER_FOLDER),
                   help="codificador(es) de imagen; se corren en este orden")
    p.add_argument("--resumen", action="store_true", help="no entrena, solo imprime la tabla")
    args, passthrough = p.parse_known_args()

    cfgs, seen = [], set()
    for enc in args.encoders:
        for spec in SUITES[args.suite]:
            c = T.finalize_config(T.build_parser().parse_args(spec + ["--encoder", enc] + passthrough))
            if c.out_dir not in seen:  # la referencia tabular es la misma para ambos codificadores
                seen.add(c.out_dir)
                cfgs.append(c)
    results_dir = cfgs[0].results_dir
    exp_dirs = [logreg_dir(results_dir)] + [Path(c.out_dir) for c in cfgs]

    logreg_baseline(results_dir, seed=cfgs[0].seed, folds=cfgs[0].folds, n_boot=cfgs[0].n_boot)
    if not args.resumen:
        pending = [c for c in cfgs if not (Path(c.out_dir) / "metrics.json").exists()]
        print(f"Suite '{args.suite}': {len(cfgs)} experimentos, {len(cfgs) - len(pending)} ya hechos, "
              f"{len(pending)} pendientes -> {results_dir}")
        t0 = time.time()
        for i, cfg in enumerate(pending):
            print(f"\n[{i + 1}/{len(pending)}] {cfg.name}   (transcurrido {(time.time() - t0) / 60:.1f} min)")
            try:
                T.run_experiment(cfg)
            except Exception:
                traceback.print_exc()
                print(f"!! {cfg.name} falló, sigo con el siguiente")
            torch.cuda.empty_cache()
    summarize(results_dir, exp_dirs)


if __name__ == "__main__":
    main()
