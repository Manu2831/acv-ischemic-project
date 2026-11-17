"""
COMPARACIÓN A — Tabular (TabPFN) vs Tabular + features de texto extraídas por reglas
======================================================================================
Este script NO usa ningún encoder de lenguaje. Todo lo que viene del texto clínico
(pretreatment_clinical_imaging_summary) se extrae con expresiones regulares a
variables numéricas/categóricas explícitas, y se le pasa a TabPFN como columnas más.

Qué hace:
  1. Carga el CSV (ajusta la sección CONFIG con tus columnas reales).
  2. Extrae features estructuradas del texto e imprime un reporte de cobertura
     (% de pacientes donde cada patrón hizo match), para que valides/ajustes las
     expresiones regulares contra tu corpus completo de 162 pacientes.
  3. Evalúa con validación cruzada repetida y estratificada (no un split fijo de
     109/24/24) dos configuraciones:
       - "SOLO TABULAR"                       : tus 8 columnas tabulares
       - "TABULAR + TEXTO (reglas, sin BERT)"  : tabular + features extraídas del texto
  4. El umbral de decisión se selecciona DENTRO de cada fold de entrenamiento
     (índice de Youden), nunca mirando el fold de evaluación.
  5. Reporta AUC / especificidad / sensibilidad / FP con intervalos de confianza
     bootstrap sobre las predicciones agregadas de cada repetición de la CV, en vez
     de un único número de una sola corrida (que con n=24 de test es puro ruido).

Requisitos: pandas, numpy, scikit-learn, scipy, tabpfn   ->  pip install tabpfn
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # rutas relativas a la carpeta de este script


import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from tabpfn import TabPFNClassifier

warnings.filterwarnings("ignore")

# ============================== CONFIG — AJUSTAR ==============================
CSV_PATH = "../datos/df_train_acv.csv"
# El texto clínico no está en el CSV: vive en un .txt por paciente dentro de
# RAWDATA_PATH/<Paciente_id>/ses-01/, igual que en cuda.py / hola.py.
RAWDATA_PATH = "../../acv_split/train/rawdata"
TEXT_FILENAME = "pretreatment_clinical_summary.txt"
ID_COL = "Paciente_id"
RAW_LABEL_COL = "mRS egreso"  # mRS crudo 0-6, hay que dicotomizarlo
TEXT_COL = "text"
LABEL_COL = "mrs_dicotomizado"  # 0 = buen outcome (mRS<=2), 1 = mal outcome (mRS>2)
TABULAR_COLS = [
    "Edad", "NIHSS ingreso", "Hipertensión", "Diabetes", "Fibrilación Auricular",
]

N_SPLITS = 5
N_REPEATS = 20  # 5x20 = 100 folds -> estimación estable con n~162
RANDOM_STATE = 42
N_BOOTSTRAP = 2000
# ===============================================================================


# ---------------------------------------------------------------------------
# 1. Extracción de features del texto por reglas (sin encoders de lenguaje)
# ---------------------------------------------------------------------------
def _first_float(pattern, text, flags=re.IGNORECASE):
    m = re.search(pattern, text, flags)
    if not m:
        return np.nan
    for g in m.groups():
        if g is not None:
            try:
                return float(g)
            except ValueError:
                continue
    return np.nan


def _flag(pattern, text, flags=re.IGNORECASE):
    return int(bool(re.search(pattern, text, flags)))


TEXT_FEATURE_NAMES = [
    "txt_nihss", "txt_lesion_volume_ml", "txt_time_to_er_hours", "txt_glucose_mgdl",
    "txt_word_count", "txt_afib", "txt_hypertension", "txt_diabetes",
    "txt_previous_stroke", "txt_coronary_disease", "txt_hemiparesis",
    "txt_facial_deviation", "txt_aphasia_dysarthria", "txt_loss_of_consciousness",
    "txt_disorientation", "txt_left_sided", "txt_right_sided",
    "txt_dwi_adc_concordant", "txt_dwi_adc_discordant", "txt_territory_mca",
    "txt_territory_basilar", "txt_territory_small_vessel",
]


def extract_text_features(text):
    """Convierte un resumen clínico en texto plano a un dict de variables explícitas.
    Los patrones están sembrados con los pocos ejemplos que vimos en el reporte de
    interpretabilidad -- corre coverage_report() contra tu corpus completo y agrega
    variantes de frase que falten antes de confiar en estos números."""
    if not isinstance(text, str) or not text.strip():
        return {k: np.nan for k in TEXT_FEATURE_NAMES}

    t = text.lower()
    feats = {}

    feats["txt_nihss"] = _first_float(
        r"nihss(?:\s+score)?(?:\s+(?:of|was|recorded[^0-9]*?))?\s*[:=]?\s*(\d+(?:\.\d+)?)", t
    )
    feats["txt_lesion_volume_ml"] = _first_float(r"lesion volume of (\d+(?:\.\d+)?)\s*ml", t)
    feats["txt_time_to_er_hours"] = _first_float(
        r"time to emergency room arrival was (\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*hours?\b[^.]{0,30}emergency room",
        t,
    )
    feats["txt_glucose_mgdl"] = _first_float(r"(\d+(?:\.\d+)?)\s*mg\s*/\s*dl", t)
    feats["txt_word_count"] = float(len(t.split()))  # feature de control, no clínica

    feats["txt_afib"] = _flag(r"atrial fibrillation", t)
    feats["txt_hypertension"] = _flag(r"hypertension", t)
    feats["txt_diabetes"] = _flag(r"diabetes", t)
    feats["txt_previous_stroke"] = _flag(r"previous stroke|prior stroke", t)
    feats["txt_coronary_disease"] = _flag(r"coronary (artery|syndrome)|ischemic (heart|cardiomyopathy)", t)

    feats["txt_hemiparesis"] = _flag(r"hemiparesis|hemiplegia", t)
    feats["txt_facial_deviation"] = _flag(r"facial (deviation|palsy)", t)
    feats["txt_aphasia_dysarthria"] = _flag(r"aphasia|dysarthria|disartria", t)
    feats["txt_loss_of_consciousness"] = _flag(r"loss of consciousness", t)
    feats["txt_disorientation"] = _flag(r"disoriented|disorientation", t)
    feats["txt_left_sided"] = _flag(r"\bleft\b[^.]{0,15}(hemiparesis|facial|sided)", t)
    feats["txt_right_sided"] = _flag(r"\bright\b[^.]{0,15}(hemiparesis|facial|sided)", t)

    feats["txt_dwi_adc_concordant"] = _flag(
        r"matched adc low signal|restricted diffusion[^.]{0,40}indicating acute ischemia", t
    )
    feats["txt_dwi_adc_discordant"] = _flag(r"no corresponding low signal|adc[^.]{0,20}(no|without)[^.]{0,20}low signal", t)
    feats["txt_territory_mca"] = _flag(r"middle cerebral artery|\bmca\b|m1 segment", t)
    feats["txt_territory_basilar"] = _flag(r"basilar artery", t)
    feats["txt_territory_small_vessel"] = _flag(r"small vessel", t)

    return feats


def build_text_feature_frame(text_series):
    rows = [extract_text_features(t) for t in text_series]
    return pd.DataFrame(rows, index=text_series.index)


def load_clinical_texts(base_path, filename=TEXT_FILENAME):
    """Carga el texto clínico desde base_path/<Paciente_id>/ses-01/<filename>,
    ya que no viene como columna del CSV."""
    records = []
    for patient in os.listdir(base_path):
        fp = os.path.join(base_path, patient, "ses-01", filename)
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                records.append({ID_COL: patient, TEXT_COL: f.read().strip()})
    return pd.DataFrame(records)


NUMERIC_TEXT_FEATURES = {"txt_nihss", "txt_lesion_volume_ml", "txt_time_to_er_hours", "txt_glucose_mgdl"}


def coverage_report(text_feat_df):
    print("\n--- Cobertura de extracción de texto ---")
    print("  (numéricas: % de pacientes con valor extraído | binarias: % con match=1)")
    for col in text_feat_df.columns:
        if col == "txt_word_count":
            continue
        series = text_feat_df[col]
        if col in NUMERIC_TEXT_FEATURES:
            pct = 100 * series.notna().mean()
        else:
            pct = 100 * (series == 1).mean()
        print(f"  {col:32s} {pct:5.1f}%")
    print(
        "Si algún porcentaje te parece bajo comparado con lo que sabes que hay en el "
        "texto, revisa/agrega variantes de frase en extract_text_features().\n"
    )


# ---------------------------------------------------------------------------
# 2. Evaluación por CV repetida, con selección de threshold dentro del fold
# ---------------------------------------------------------------------------
def pick_threshold(y_true, y_proba):
    """Umbral que maximiza el índice de Youden (sensibilidad + especificidad - 1)
    sobre el propio fold de ENTRENAMIENTO -- nunca se mira el fold de evaluación."""
    best_t, best_j = 0.5, -1.0
    for t in np.unique(y_proba):
        pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return values.mean(), lo, hi


def summarize(label, per_repeat_df):
    print(f"\n=== {label} ===  ({len(per_repeat_df)} repeticiones de {N_SPLITS}-fold CV)")
    for metric in ["auc", "specificity", "sensitivity", "fp"]:
        mean, lo, hi = bootstrap_ci(per_repeat_df[metric].dropna().values)
        print(f"  {metric:12s} media={mean:.3f}   IC95%=[{lo:.3f}, {hi:.3f}]")


def evaluate_config(X, y, label, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=RANDOM_STATE):
    """RepeatedStratifiedKFold: cada repetición cubre TODOS los pacientes exactamente
    una vez (en folds distintos), así que agregamos las predicciones por repetición
    para obtener una métrica por repetición, y de ahí el IC bootstrap."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    X = X.reset_index(drop=True)
    y = np.asarray(y)

    oof_records = []
    for fold_id, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        repeat_id = fold_id // n_splits
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = TabPFNClassifier(device="cpu", random_state=seed)
        clf.fit(X_train, y_train)

        proba_train = clf.predict_proba(X_train)[:, 1]
        threshold = pick_threshold(y_train, proba_train)

        proba_test = clf.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= threshold).astype(int)

        for yt, yp, pr in zip(y_test, pred_test, proba_test):
            oof_records.append((repeat_id, yt, pr, yp))

    oof = pd.DataFrame(oof_records, columns=["repeat", "y_true", "y_proba", "y_pred"])

    per_repeat = []
    for r, g in oof.groupby("repeat"):
        auc = roc_auc_score(g.y_true, g.y_proba)
        tn, fp, fn, tp = confusion_matrix(g.y_true, g.y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        per_repeat.append({"repeat": r, "auc": auc, "specificity": spec, "sensitivity": sens, "fp": fp})
    per_repeat = pd.DataFrame(per_repeat)

    summarize(label, per_repeat)
    return oof, per_repeat


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in TABULAR_COLS + [ID_COL, RAW_LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Estas columnas no están en el CSV, ajusta CONFIG: {missing}")

    df[RAW_LABEL_COL] = pd.to_numeric(df[RAW_LABEL_COL], errors="coerce")
    df = df.dropna(subset=[RAW_LABEL_COL])
    df[LABEL_COL] = (df[RAW_LABEL_COL] > 2).astype(int)  # mRS 0-2 = buen outcome, 3-6 = mal outcome

    text_df = load_clinical_texts(RAWDATA_PATH)
    df = df.merge(text_df, on=ID_COL, how="inner")
    print(f"Pacientes con tabular + texto disponibles: {len(df)}")

    y = df[LABEL_COL].astype(int)
    X_tab = df[TABULAR_COLS].copy()

    text_feats = build_text_feature_frame(df[TEXT_COL])
    coverage_report(text_feats)

    X_tab_texto = pd.concat([X_tab.reset_index(drop=True), text_feats.reset_index(drop=True)], axis=1)

    print("############################################")
    print("# Comparación: tabular vs tabular+texto (reglas, sin BERT)")
    print("############################################")

    evaluate_config(X_tab, y, "SOLO TABULAR")
    evaluate_config(X_tab_texto, y, "TABULAR + TEXTO (reglas, sin BERT)")

    print(
        "\nSi 'TABULAR + TEXTO' no mejora el AUC/especificidad de forma consistente "
        "(el IC95% se solapa mucho con el de 'SOLO TABULAR'), la conclusión honesta es "
        "que el texto por reglas no aporta y puedes quedarte solo con TabPFN sobre tabular. "
        "Compara estos números contra modelo_con_bert.py antes de decidir."
    )


if __name__ == "__main__":
    main()
