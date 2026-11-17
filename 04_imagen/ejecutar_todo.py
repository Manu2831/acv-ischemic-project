"""
EL ARCHIVO PARA CORRER TODO (imagen + tabular), en orden y saltando lo ya hecho:

  1. lesion : prepara datos -> entrena SwinUNETR segmentando la lesión (ISLES 2022 + tu
              train) -> extrae tokens congelados                       (lesion_encoder.py)
  2. dinov2 : extrae tokens DINOv2 corte por corte (opcional)          (extract_dinov2.py)
  3. vit3d  : caché de volúmenes para el ViT 3D desde cero (opcional)  (preprocess.py)
  4. experimentos de mRS para cada encoder                             (run_experiments.py)

Uso (desde text/, o en Kaggle con las variables de rutas.py):
  python 04_imagen/ejecutar_todo.py                              # encoder de lesión, suite 'todo'
  python 04_imagen/ejecutar_todo.py --encoders lesion dinov2     # + comparación con DINOv2
  python 04_imagen/ejecutar_todo.py --seg_epochs 60              # menos tiempo de segmentación
  python 04_imagen/ejecutar_todo.py --quick                      # prueba de humo de todo el flujo
Cualquier otro argumento se reenvía a los experimentos (ej. --dim 64).

Necesita GPU de >= 12 GB para entrenar la segmentación en tiempo razonable (Kaggle T4/P100).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from rutas import CACHE_DIR, DINO_ROOT, LESION_ROOT, RESULTS_DIR

HERE = Path(__file__).resolve().parent


def run(script, *args):
    cmd = [sys.executable, str(HERE / script), *map(str, args)]
    print(f"\n>>> {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"<<< {script} listo en {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", default=["lesion"], choices=["lesion", "dinov2", "vit3d"])
    p.add_argument("--suite", default="todo", choices=["pedido", "fusiones", "fusiones_modalidad", "controles", "todo"])
    p.add_argument("--seg_epochs", type=int, default=65, help="épocas de la segmentación de lesión")
    p.add_argument("--feature_size", type=int, default=24, help="SwinUNETR: 24 o 48")
    p.add_argument("--sin_isles", action="store_true", help="no descargar/usar ISLES 2022")
    p.add_argument("--backbone", default="vit_small_patch14_reg4_dinov2.lvd142m")
    p.add_argument("--quick", action="store_true")
    args, passthrough = p.parse_known_args()

    t0 = time.time()
    lesion_model = f"swinunetr_fs{args.feature_size}" + ("_quick" if args.quick else "")
    if "lesion" in args.encoders:
        seg = ["--etapa", "todo", "--epochs", args.seg_epochs, "--feature_size", args.feature_size,
               "--modelo", f"swinunetr_fs{args.feature_size}"]
        seg += ["--sin_isles"] if args.sin_isles else ["--descargar_isles"]
        if args.quick:
            seg.append("--quick")
        if not (LESION_ROOT / lesion_model / "meta.json").exists():
            run("lesion_encoder.py", *seg)
    if "dinov2" in args.encoders and not (DINO_ROOT / args.backbone.split(".")[0] / "meta.json").exists():
        run("extract_dinov2.py", "--backbone", args.backbone)
    if "vit3d" in args.encoders and not (CACHE_DIR / "meta.json").exists():
        run("preprocess.py")

    exp = ["--suite", args.suite, "--encoders", *args.encoders, "--lesion_model", lesion_model,
           "--backbone", args.backbone, *passthrough]
    if args.quick:
        exp.append("--quick")
    run("run_experiments.py", *exp)
    print(f"\nTODO LISTO en {(time.time() - t0) / 60:.1f} min. Resultados: "
          f"{RESULTS_DIR.with_name('results_quick') if args.quick else RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
