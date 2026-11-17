"""
Sube a Weights & Biases (wandb.ai) los resultados YA generados, sin reentrenar: sirve al final
del notebook de Kaggle o con un resultados_acv.zip descargado.

Qué sube (un run por experimento, agrupados por carpeta):
  - curvas por época de cada fold: pérdida train / val y AUC de val          (log.csv)
  - métricas finales de oof / val / test / ext con IC95%, comparables en la
    tabla de runs de W&B (ext/auc, ext/f1, ext/dAUC_vs_logreg, ...)          (metrics.json)
  - curvas ROC y PR + matriz de confusión en ext y oof                       (predictions.csv, sin IDs)
  - entrenamiento de la segmentación del encoder: pérdida y Dice por época   (entrenamiento_log.json)
  - resumen.csv como tabla + barras de AUC ext
NO sube imágenes, mapas de atención, modelos ni IDs de pacientes.

Requisitos: cuenta en wandb.ai y su API key (wandb.ai/authorize)
  Kaggle: Add-ons -> Secrets -> WANDB_API_KEY  (el notebook la usa sola si existe)
  PC:     pip install wandb  y  wandb login

Uso (desde text/):
  python 04_imagen/subir_wandb.py --results_dir C:/ruta/resultados_kaggle --grupo corrida_completa
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

try:  # resultados de una versión vieja: completa las métricas desde predictions.csv
    from run_experiments import rescore
except ImportError:
    rescore = lambda d: json.loads((d / "metrics.json").read_text())

SPLITS = ("oof", "val", "test", "ext")
CLASSES = ["mRS 0-2", "mRS 3-6"]
CURVES = {"train_loss": "pérdida train", "val_loss": "pérdida val del fold", "val_auc": "AUC val del fold"}


def flat_metrics(m):
    out = {"umbral_youden_oof": m["threshold_youden_oof"], "minutos": m.get("minutes", 0)}
    for s in SPLITS:
        for k, v in m[s].items():
            if k == "ci95":
                out.update({f"{s}/{kk}_lo": lo for kk, (lo, hi) in v.items()})
                out.update({f"{s}/{kk}_hi": hi for kk, (lo, hi) in v.items()})
            else:
                out[f"{s}/{k}"] = v
    out.update({f"compuerta/{k}": v for k, v in m.get("gate_mean", {}).items()})
    return out


def log_curves(run, log):
    folds = sorted(log.fold.unique())
    for col, title in CURVES.items():
        run.log({f"curvas/{col}": wandb.plot.line_series(
            xs=[log.loc[log.fold == f, "epoch"].tolist() for f in folds],
            ys=[log.loc[log.fold == f, col].fillna(0).tolist() for f in folds],
            keys=[f"fold {f}" for f in folds], title=title, xname="época")})


def log_predictions(run, pred, thr):
    for name, d in (("ext", pred[pred.split.isin(["val", "test"])]), ("oof", pred[pred.split == "train"])):
        y, p = d.y.values.astype(int), d.prob.values
        probas = np.stack([1 - p, p], axis=1)
        run.log({f"{name}/roc": wandb.plot.roc_curve(y, probas, labels=CLASSES, classes_to_plot=[1],
                                                     title=f"ROC {name}"),
                 f"{name}/pr": wandb.plot.pr_curve(y, probas, labels=CLASSES, classes_to_plot=[1],
                                                   title=f"Precision-Recall {name}"),
                 f"{name}/confusion": wandb.plot.confusion_matrix(y_true=y, preds=(p >= thr).astype(int).tolist(),
                                                                  class_names=CLASSES,
                                                                  title=f"Confusión {name} (umbral {thr:.2f})")})


def upload_experiment(d, results_dir, resumen, args):
    m = rescore(d)
    folder = d.relative_to(results_dir).parent.as_posix()
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    cfg = {k: v for k, v in cfg.items() if k not in ("out_dir", "results_dir")}
    with wandb.init(project=args.proyecto, group=args.grupo, name=f"{m['name']} ({folder.split('/')[0]})",
                    job_type=folder.split("/")[0], tags=folder.split("/"), config={**cfg, "carpeta": folder},
                    dir=args.wandb_dir, mode=args.modo) as run:
        run.summary.update(flat_metrics(m))
        row = resumen[(resumen.grupo == folder) & (resumen.experimento == m["name"])] if resumen is not None else []
        if len(row) and "dAUC_vs_logreg" in row and pd.notna(row.dAUC_vs_logreg.iloc[0]):
            run.summary["ext/dAUC_vs_logreg"] = float(row.dAUC_vs_logreg.iloc[0])
        if (d / "log.csv").exists():
            log_curves(run, pd.read_csv(d / "log.csv"))
        if (d / "predictions.csv").exists():
            log_predictions(run, pd.read_csv(d / "predictions.csv"), m["threshold_youden_oof"])


def upload_segmentation(f, args):
    log = json.loads(f.read_text())
    meta = f.with_name("meta.json")
    with wandb.init(project=args.proyecto, group=args.grupo, name=f"segmentación {f.parent.name}",
                    job_type="segmentacion_lesion", dir=args.wandb_dir, mode=args.modo,
                    config=json.loads(meta.read_text()) if meta.exists() else {}) as run:
        for row in log:
            run.log({k: v for k, v in row.items() if k != "epoch"}, step=row["epoch"])
        for k in ("dice_adc+dwi", "dice_dwi", "dice_adc", "dice_adcmask_adc", "dice_seleccion"):
            dice = [r[k] for r in log if k in r]
            if dice:
                run.summary[f"mejor_{k}"] = max(dice)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True, help="carpeta con resumen.csv (results/ o el zip descomprimido)")
    p.add_argument("--proyecto", default="acv-mrs")
    p.add_argument("--grupo", default="corrida", help="agrupa los runs de una misma corrida en W&B")
    p.add_argument("--modo", default=None, choices=["online", "offline"],
                   help="offline: guarda local y se sube después con 'wandb sync' (por defecto: WANDB_MODE u online)")
    args = p.parse_args()
    results_dir = Path(args.results_dir)
    args.wandb_dir = str(results_dir)
    resumen = pd.read_csv(results_dir / "resumen.csv") if (results_dir / "resumen.csv").exists() else None

    for f in sorted(results_dir.rglob("entrenamiento_log.json")):
        print("segmentación:", f.parent.name, flush=True)
        upload_segmentation(f, args)
    dirs = sorted({q.parent for q in results_dir.rglob("metrics.json")})
    for i, d in enumerate(dirs):
        print(f"[{i + 1}/{len(dirs)}] {d.relative_to(results_dir).as_posix()}", flush=True)
        upload_experiment(d, results_dir, resumen, args)
    if resumen is not None:
        with wandb.init(project=args.proyecto, group=args.grupo, name="resumen", job_type="resumen",
                        dir=args.wandb_dir, mode=args.modo) as run:
            resumen.insert(0, "nombre", resumen.experimento + " (" + resumen.grupo.str.split("/").str[0] + ")")
            table = wandb.Table(dataframe=resumen)
            run.log({"resumen": table,
                     "AUC_ext": wandb.plot.bar(table, "nombre", "AUC_ext", title="AUC ext (val+test) por experimento")})
    print(f"Listo: {len(dirs)} experimentos en el proyecto W&B '{args.proyecto}', grupo '{args.grupo}'")


if __name__ == "__main__":
    main()
