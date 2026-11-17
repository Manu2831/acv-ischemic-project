"""
Rutas del proyecto. Por defecto usan el layout local; en Kaggle (u otra máquina) se
cambian con variables de entorno, sin tocar el código:

  ACV_DATA_ROOT   carpeta con acv_split/ y dataset_report.csv     (local: acv-ischemic/)
  ACV_TABULAR_DIR carpeta con df_train/val/test_acv.csv            (local: text/datos/)
  ACV_TABLA_LIMPIA tabla corregida con las notas de ingreso        (local: text/datos/tabla_limpia.csv)
  ACV_TABLA=original  ignora la tabla limpia y usa los df_*_acv.csv de antes
  ACV_WORK_DIR    donde se escriben cachés, modelos y resultados   (local: text/04_imagen/)
  ACV_ISLES_DIR   ISLES 2022 descomprimido (opcional)              (local: acv-ischemic/isles2022/)
  ACV_AMP=fp16    fuerza fp16 como en la T4 de Kaggle               (local: bf16)
"""

import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent

DATA_ROOT = Path(os.environ.get("ACV_DATA_ROOT", HERE.parents[1]))
TABULAR_DIR = Path(os.environ.get("ACV_TABULAR_DIR", HERE.parent / "datos"))
TABLA_LIMPIA = Path(os.environ.get("ACV_TABLA_LIMPIA", TABULAR_DIR / "tabla_limpia.csv"))
WORK_DIR = Path(os.environ.get("ACV_WORK_DIR", HERE))
ISLES_DIR = Path(os.environ.get("ACV_ISLES_DIR", DATA_ROOT / "isles2022"))

SPLIT_ROOT = DATA_ROOT / "acv_split"
REPORT_CSV = DATA_ROOT / "dataset_report.csv"
CACHE_DIR = WORK_DIR / "cache"
DINO_ROOT = WORK_DIR / "cache_dinov2"
LESION_ROOT = WORK_DIR / "cache_lesion"
RESULTS_DIR = WORK_DIR / "results"


def nifti(path):
    """Kaggle puede descomprimir los .nii.gz al procesar un dataset subido: si falta X.nii.gz, usa X.nii."""
    path = Path(path)
    if not path.exists() and path.suffix == ".gz" and path.with_suffix("").exists():
        return path.with_suffix("")
    return path


def amp_dtype():
    """bf16 solo en GPUs Ampere+ (tu RTX 2050). En T4/P100 de Kaggle bf16 se emula y es lento -> fp16.
    ACV_AMP=fp16 lo fuerza (para probar localmente el mismo camino que corre en la T4)."""
    if os.environ.get("ACV_AMP") == "fp16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16
