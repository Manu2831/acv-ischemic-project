"""
Construye la carpeta para subir a Kaggle:

  acv-ischemic/kaggle_upload/
    acv-ischemic-mri.zip     <- subir este ÚNICO archivo como Dataset PRIVADO (Kaggle lo descomprime)
    acv-ischemic-mri/        contenido del zip (solo datos):
      acv_split/             DWI, ADC, máscaras de cerebro y de lesión (encabezados sin hora de adquisición)
      datos/                 df_train / df_val / df_test
      dataset_report.csv     SOLO patient + diabetes (lo único que usa el código)
    acv_mrs_kaggle.ipynb     <- importar como Notebook; lleva el código de 04_imagen/ en celdas %%writefile

Datos y código van separados: si cambiás el código, volvé a correr esto y re-importá SOLO el
notebook (el dataset no se vuelve a subir).

Minimización de datos: NO se copian los .txt (historia clínica) ni los .json (institución,
serial del equipo, horas).

Uso (desde text/):  python 05_kaggle/construir_paquete.py
"""

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import pandas as pd

TEXT_DIR = Path(__file__).resolve().parents[1]
PROJECT = TEXT_DIR.parent
CODE_FILES = ["rutas.py", "preprocess.py", "models.py", "train.py", "run_experiments.py",
              "lesion_encoder.py", "extract_dinov2.py", "ejecutar_todo.py", "subir_wandb.py"]
RAW_FILES = ["dwi.nii.gz", "adc.nii.gz", "dwi_brain_mask.nii.gz", "adc_brain_mask.nii.gz"]
DERIV_FILES = ["ana_dwi_mask.nii.gz", "ana_adc_mask.nii.gz"]


def copy_nifti_clean(src, dst):
    img = nib.load(str(src))
    hdr = img.header.copy()
    for field in ("descrip", "aux_file", "db_name"):
        if field in hdr:
            hdr[field] = b""
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(img.dataobj, img.affine, hdr), str(dst))


def notebook():
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(keepends=True)}
    code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": s.strip().splitlines(keepends=True)}
    writefile = lambda name: code(f"%%writefile codigo/{name}\n"
                                  + (TEXT_DIR / "04_imagen" / name).read_text(encoding="utf-8"))
    cells = [
        md("""
# mRS dicotomizado en ACV isquémico — imagen (ADC/DWI) + tabular

**Cómo usarlo**
1. Subí el archivo `acv-ischemic-mri.zip` como **Dataset privado** (*Datasets → New Dataset → Browse Files*,
   o arrastrarlo). Es un solo archivo: Kaggle lo descomprime solo (si no lo hace, este notebook lo descomprime).
2. Importá este notebook (*Code → New Notebook → File → Import Notebook*).
3. Panel derecho: *Add Input* → tu dataset · *Accelerator* → **GPU T4 x1** (o P100) · *Internet* → **On**
   (necesita teléfono verificado; se usa para instalar MONAI y descargar ISLES 2022).
4. Ajustá la celda de configuración y usá **Save Version → Save & Run All (Commit)**: corre en segundo plano
   (máx. 12 h) y los resultados quedan en la pestaña *Output* como `resultados_acv.zip`.

El código va **dentro de este notebook** (celdas `%%writefile`). Si hay una versión nueva del código:
*File → Import Notebook* otra vez; el dataset NO se vuelve a subir.

**Tabla clínica corregida**: subí `tabla_limpia.csv` como un segundo **Dataset privado** (pesa pocos KB) y agregalo
también con *Add Input*. Si no está, el notebook usa la tabla original y lo avisa en la celda de abajo.

**Curvas en Weights & Biases (opcional)**: *Add-ons → Secrets → Add Secret*, nombre `WANDB_API_KEY`, valor = tu
key de [wandb.ai/authorize](https://wandb.ai/authorize), y marcá *Attach*. Al final se suben curvas por época, ROC/PR,
matrices de confusión y métricas al proyecto `acv-mrs`. No se suben imágenes, modelos ni IDs de pacientes.

**Qué hace**: entrena SwinUNETR para **segmentar la lesión** (ISLES 2022 + tus pacientes de train), lo congela
como encoder especializado en lesión, y lo fusiona con las variables tabulares (concat / FiLM / cross-attention /
compuerta de confianza) para predecir mRS > 2. Todo con CV 5-fold, ensemble y validación en val/test.

⚠️ Son datos de pacientes reales: mantené el dataset **privado** y verificá que tu aval de ética /
acuerdo de uso de datos permita procesarlos en la nube.
"""),
        code("""
# ===================== CONFIGURACIÓN =====================
ENCODERS   = ["lesion"]   # agregá "dinov2" para comparar con un ViT preentrenado en imágenes naturales
SUITE      = "todo"       # "pedido" = ADC, DWI, ADC+TAB, DWI+TAB, ADC+DWI+TAB (+ referencias), más rápido
SEG_EPOCHS = 65           # épocas de segmentación de la lesión (lo más lento: ~1-2.5 h en T4)
QUICK      = False        # True = prueba de humo (~15 min) para verificar que todo corre
WANDB_PROYECTO = "acv-mrs"  # se sube a W&B solo si agregaste el secreto WANDB_API_KEY (ver arriba)
"""),
        code("""
import glob, os, shutil, subprocess, sys, tempfile
from pathlib import Path

KAGGLE_INPUT = Path(os.environ.get("ACV_KAGGLE_INPUT", "/kaggle/input"))
KAGGLE_WORKING = Path(os.environ.get("ACV_KAGGLE_WORKING", "/kaggle/working"))
buscar = lambda raiz: sorted(Path(p).parent for p in glob.glob(str(Path(raiz) / "**" / "acv_split"), recursive=True))
found = buscar(KAGGLE_INPUT)
if not found:  # Kaggle no descomprimió el .zip al subirlo: lo descomprimimos acá
    zips = glob.glob(str(KAGGLE_INPUT / "**" / "*.zip"), recursive=True)
    if zips:
        destino = Path(tempfile.gettempdir()) / "acv_data"
        print("descomprimiendo", zips[0], "->", destino, flush=True)
        shutil.unpack_archive(zips[0], destino)
        found = buscar(destino)
assert found, "No encuentro acv_split/ en /kaggle/input: ¿agregaste el dataset como Input?"
DATA = found[0]
print("datos :", DATA)

# diagnóstico: 160 pacientes x 6 volúmenes (Kaggle puede dejarlos como .nii en vez de .nii.gz)
vols = glob.glob(str(DATA / "acv_split" / "*" / "*" / "*" / "ses-01" / "*.nii*"))
print(f"volúmenes: {len(vols)} de 960 esperados", {e: sum(v.endswith(e) for v in vols) for e in (".nii.gz", ".nii")})
ejemplo = sorted(DATA.glob("acv_split/train/rawdata/*/ses-01"))[:1]
if ejemplo:
    print("ejemplo:", ejemplo[0], sorted(p.name for p in ejemplo[0].iterdir()))
assert len(vols) >= 960, "Faltan volúmenes: ¿el dataset terminó de procesarse? Pegá esta salida"

WORK = KAGGLE_WORKING / "acv"
CODE = KAGGLE_WORKING / "codigo"
WORK.mkdir(parents=True, exist_ok=True)
CODE.mkdir(parents=True, exist_ok=True)
os.chdir(KAGGLE_WORKING)  # las celdas %%writefile de abajo escriben en codigo/
os.environ.update({"ACV_DATA_ROOT": str(DATA), "ACV_TABULAR_DIR": str(DATA / "datos"), "ACV_WORK_DIR": str(WORK)})
os.environ.setdefault("ACV_ISLES_DIR", "/tmp/isles2022")
# tabla clínica corregida con las notas de ingreso: en el zip de datos o en un dataset aparte
tablas = sorted(glob.glob(str(KAGGLE_INPUT / "**" / "tabla_limpia.csv"), recursive=True)) +     sorted(glob.glob(str(DATA / "datos" / "tabla_limpia.csv")))
tablas.sort(key=lambda t: Path(t).parent.name == "datos")  # si hay dos, gana la subida aparte (la más nueva)
if tablas:
    os.environ["ACV_TABLA_LIMPIA"] = tablas[0]
    print("tabla clínica: LIMPIA ->", tablas[0])
else:
    print("tabla clínica: ORIGINAL (no encontré tabla_limpia.csv: subila como dataset privado y agregala como Input)")
print("salida:", WORK)
"""),
        md("## Código (04_imagen/) — cada celda escribe un archivo en `/kaggle/working/codigo/`"),
        *[writefile(f) for f in CODE_FILES],
        code("""
# Dependencias: --no-deps para NO reemplazar el PyTorch con CUDA que ya trae Kaggle
import importlib.util
for mod, spec in {"nibabel": "nibabel", "monai": "monai==1.5.1", "timm": "timm"}.items():
    if importlib.util.find_spec(mod) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", spec], check=True)
import torch, monai
print("torch", torch.__version__, "| monai", monai.__version__)
assert torch.cuda.is_available(), "Activá la GPU: Settings -> Accelerator -> GPU T4 x1"
print("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
"""),
        code("""
cmd = [sys.executable, str(CODE / "ejecutar_todo.py"), "--encoders", *ENCODERS, "--suite", SUITE,
       "--seg_epochs", str(SEG_EPOCHS)] + (["--quick"] if QUICK else [])
print(" ".join(cmd[1:]), flush=True)
proc = subprocess.Popen(cmd, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
assert proc.wait() == 0, "Falló: revisá el log de arriba"
"""),
        code("""
import pandas as pd
RES = WORK / ("results_quick" if QUICK else "results")
for f in (WORK / "cache_lesion").glob("swinunetr_*/*.json"):
    dst = RES / "encoder_lesion" / f.parent.name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(f, dst / f.name)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
tabla = pd.read_csv(RES / "resumen.csv")
cols = ["grupo", "experimento", "AUC_ext", "IC_AUC_ext", "AUPRC_ext", "F1_ext", "Acc_ext", "BalAcc_ext",
        "Sens_ext", "Spec_ext", "PPV_ext", "NPV_ext", "MCC_ext", "dAUC_vs_logreg", "IC_dAUC"]
print(tabla[[c for c in cols if c in tabla]].to_string(index=False))
print("\\nTodas las métricas por split (oof/val/test/ext) con IC95%: metricas_por_split.csv")
"""),
        code("""
shutil.make_archive(str(KAGGLE_WORKING / "resultados_acv"), "zip", RES)
print("Listo: descargá resultados_acv.zip desde la pestaña Output")
"""),
        code("""
# (opcional) curvas y métricas a Weights & Biases, si agregaste el secreto WANDB_API_KEY
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
except Exception:
    print("Sin secreto WANDB_API_KEY: no se sube a W&B (es opcional)")
else:
    if importlib.util.find_spec("wandb") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wandb"], check=True)
    subprocess.run([sys.executable, str(CODE / "subir_wandb.py"), "--results_dir", str(RES), "--proyecto",
                    WANDB_PROYECTO, "--grupo", "quick" if QUICK else "completa"], cwd=CODE, check=True)
"""),
    ]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--salida", default=str(PROJECT / "kaggle_upload"))
    p.add_argument("--forzar", action="store_true", help="recopiar las imágenes aunque ya existan")
    args = p.parse_args()
    out = Path(args.salida)
    ds = out / "acv-ischemic-mri"

    n = 0
    for split_dir in sorted((PROJECT / "acv_split").iterdir()):
        for pdir in sorted((split_dir / "rawdata").iterdir()):
            pairs = [(pdir / "ses-01" / f, ds / "acv_split" / split_dir.name / "rawdata" / pdir.name / "ses-01" / f)
                     for f in RAW_FILES]
            der = split_dir / "derivatives" / pdir.name / "ses-01"
            pairs += [(der / f, ds / "acv_split" / split_dir.name / "derivatives" / pdir.name / "ses-01" / f)
                      for f in DERIV_FILES]
            for src, dst in pairs:
                if args.forzar or not dst.exists():
                    copy_nifti_clean(src, dst)
            n += 1
    print(f"imágenes: {n} pacientes")

    (ds / "datos").mkdir(parents=True, exist_ok=True)
    for f in sorted((TEXT_DIR / "datos").glob("df_*_acv.csv")):
        shutil.copy(f, ds / "datos" / f.name)
    pd.read_csv(PROJECT / "dataset_report.csv", usecols=["patient", "diabetes"]).to_csv(
        ds / "dataset_report.csv", index=False)
    limpia = TEXT_DIR / "datos" / "tabla_limpia.csv"  # la genera 07_tabla_limpia/construir_tabla.py
    if limpia.exists():
        shutil.copy(limpia, ds / "datos" / limpia.name)  # dentro del zip, por si se vuelve a subir
        shutil.copy(limpia, out / limpia.name)            # suelta, para subirla como dataset chico aparte

    shutil.rmtree(ds / "codigo", ignore_errors=True)  # el código ahora va en el notebook

    (out / "acv_mrs_kaggle.ipynb").write_text(json.dumps(notebook(), indent=1, ensure_ascii=False), encoding="utf-8")
    # el navegador no deja elegir carpetas en "Browse Files": se sube un solo .zip
    zip_path = shutil.make_archive(str(ds), "zip", root_dir=out, base_dir=ds.name)
    size = Path(zip_path).stat().st_size / 1e6
    print(f"Paquete listo en {out}")
    print(f"  1) subir como Dataset privado: {zip_path}  ({size:.0f} MB)")
    print(f"  2) importar como Notebook:     {out / 'acv_mrs_kaggle.ipynb'}")
    if limpia.exists():
        print(f"  3) tabla corregida, como segundo Dataset privado (si no volvés a subir el zip): {out / limpia.name}")


if __name__ == "__main__":
    main()
