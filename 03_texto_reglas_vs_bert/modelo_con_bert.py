"""
COMPARACIÓN B — Tabular + texto (reglas) vs + embeddings de PubMedBERT congelado
======================================================================================
Complementa a modelo_sin_bert.py. Aquí se agrega una tercera fuente de features:
un embedding de PubMedBERT congelado (sin fine-tuning) sobre pretreatment_clinical_
imaging_summary, reducido con PCA, y comparado bajo el mismo esquema de CV repetida
+ bootstrap que el archivo sin BERT, para que los números sean comparables entre sí.

Qué hace:
  1. Carga el CSV y extrae las mismas features por reglas que modelo_sin_bert.py.
  2. Calcula UNA sola vez el embedding de PubMedBERT (mean pooling con máscara de
     atención, encoder congelado -- torch.no_grad()) para cada paciente. Como el
     encoder no se re-entrena, no depende del fold, así que calcularlo una vez es
     correcto y evita repetir cientos de forward passes.
  3. Corre validación cruzada repetida evaluando tres configuraciones:
       - "TABULAR + TEXTO (reglas)"        : misma que en modelo_sin_bert.py (referencia)
       - "TABULAR + TEXTO + BERT"          : agrega PCA(embeddings) como columnas
       - "TABULAR + TEXTO + BERT BARAJADO" : control -- el embedding se asigna a un
         paciente aleatorio distinto en cada repetición. Si esta config rinde igual
         o mejor que la real, BERT no está aportando información específica del
         paciente correcto (mismo diagnóstico que ya hiciste a mano, pero ahora con
         intervalos de confianza en vez de una sola corrida).
     El PCA se ajusta SOLO con el fold de entrenamiento en cada iteración (nunca
     con el fold de evaluación), para no filtrar información de test hacia la
     reducción de dimensionalidad.
  4. Imprime además la similitud coseno media entre embeddings dentro/entre clases,
     como chequeo rápido de si el encoder está generando representaciones que
     distinguen pacientes (si esto sigue ~0.99, es una señal de alerta
     independiente del AUC).

Requisitos: pandas, numpy, scikit-learn, scipy, torch, transformers, tabpfn
   pip install tabpfn transformers torch

NOTA: la descarga de los pesos de PubMedBERT requiere acceso a huggingface.co la
primera vez que corras esto (luego queda en caché local).
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # rutas relativas a la carpeta de este script


import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import RepeatedStratifiedKFold
from tabpfn import TabPFNClassifier
from transformers import AutoModel, AutoTokenizer

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
LABEL_COL = "mrs_dicotomizado"
TABULAR_COLS = [
    "Edad", "NIHSS ingreso", "Hipertensión", "Diabetes", "Fibrilación Auricular",
]

BERT_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
BERT_MAX_LENGTH = 512
BERT_BATCH_SIZE = 8
N_PCA_COMPONENTS = 8  # columnas finales que aporta la rama BERT; con n~162 no subas mucho de esto

N_SPLITS = 5
N_REPEATS = 20
RANDOM_STATE = 42
N_BOOTSTRAP = 2000
# ===============================================================================


# ---------------------------------------------------------------------------
# 1. Features de texto por reglas (idéntico a modelo_sin_bert.py, para que la
#    columna base "TABULAR + TEXTO" sea exactamente comparable entre los dos archivos)
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
    feats["txt_word_count"] = float(len(t.split()))

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


# ---------------------------------------------------------------------------
# 1b. Carga del texto clínico (un .txt por paciente, no está en el CSV)
# ---------------------------------------------------------------------------
def load_clinical_texts(base_path, filename=TEXT_FILENAME):
    records = []
    for patient in os.listdir(base_path):
        fp = os.path.join(base_path, patient, "ses-01", filename)
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                records.append({ID_COL: patient, TEXT_COL: f.read().strip()})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Embeddings de PubMedBERT congelado (mean pooling, sin fine-tuning)
# ---------------------------------------------------------------------------
def compute_bert_embeddings(texts, model_name=BERT_MODEL_NAME, batch_size=BERT_BATCH_SIZE,
                             max_length=BERT_MAX_LENGTH, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    texts = [t if isinstance(t, str) else "" for t in texts]
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            token_embeds = out.last_hidden_state  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
            summed = (token_embeds * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = (summed / counts).cpu().numpy()
            all_embeds.append(mean_pooled)
    return np.vstack(all_embeds)


def cosine_similarity_report(bert_embeds, y):
    y = np.asarray(y)
    sims = cosine_similarity(bert_embeds)
    iu = np.triu_indices_from(sims, k=1)
    all_pairs = sims[iu]

    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    within0 = cosine_similarity(bert_embeds[idx0])[np.triu_indices(len(idx0), k=1)] if len(idx0) > 1 else np.array([])
    within1 = cosine_similarity(bert_embeds[idx1])[np.triu_indices(len(idx1), k=1)] if len(idx1) > 1 else np.array([])
    between = cosine_similarity(bert_embeds[idx0], bert_embeds[idx1]).ravel() if len(idx0) and len(idx1) else np.array([])

    print("\n--- Similitud coseno de los embeddings BERT (chequeo de colapso de representación) ---")
    print(f"  todos los pares       media={all_pairs.mean():.4f}  min={all_pairs.min():.4f}  max={all_pairs.max():.4f}")
    if len(within0):
        print(f"  dentro de clase 0     media={within0.mean():.4f}")
    if len(within1):
        print(f"  dentro de clase 1     media={within1.mean():.4f}")
    if len(between):
        print(f"  entre clases          media={between.mean():.4f}")
    print(
        "  Si estos tres números son casi idénticos y muy cercanos a 1.0, el encoder no está "
        "generando representaciones que separen pacientes por outcome, independientemente de "
        "lo que diga el AUC más abajo.\n"
    )


# ---------------------------------------------------------------------------
# 3. Evaluación por CV repetida (idéntica lógica de umbral/bootstrap que en
#    modelo_sin_bert.py), con opción de agregar columnas BERT (reales o barajadas)
# ---------------------------------------------------------------------------
def pick_threshold(y_true, y_proba):
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


def evaluate_config(X_base, y, label, bert_embeds=None, shuffle_bert=False, n_pca=N_PCA_COMPONENTS,
                     n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=RANDOM_STATE):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    X_base = X_base.reset_index(drop=True)
    y = np.asarray(y)
    n = len(X_base)
    rng = np.random.default_rng(seed)

    oof_records = []
    current_repeat = -1
    perm = np.arange(n)

    for fold_id, (train_idx, test_idx) in enumerate(rskf.split(X_base, y)):
        repeat_id = fold_id // n_splits

        if bert_embeds is not None and shuffle_bert and repeat_id != current_repeat:
            perm = rng.permutation(n)  # una permutación nueva por repetición, no por fold
            current_repeat = repeat_id

        if bert_embeds is not None:
            be = bert_embeds[perm] if shuffle_bert else bert_embeds
            pca = PCA(n_components=n_pca, random_state=seed)
            pca.fit(be[train_idx])  # PCA ajustado SOLO con el fold de train
            reduced = pca.transform(be)
            bert_cols = [f"bert_pc{i + 1}" for i in range(n_pca)]
            bert_df = pd.DataFrame(reduced, columns=bert_cols)
            X_full = pd.concat([X_base, bert_df], axis=1)
        else:
            X_full = X_base

        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
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
# 4. Main
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
    X_tab_texto = pd.concat([X_tab.reset_index(drop=True), text_feats.reset_index(drop=True)], axis=1)

    print("Calculando embeddings de PubMedBERT (una sola vez, encoder congelado)...")
    bert_embeds = compute_bert_embeddings(df[TEXT_COL])
    print(f"  embeddings shape: {bert_embeds.shape}")

    cosine_similarity_report(bert_embeds, y)

    print("############################################")
    print("# Comparación: tabular+texto(reglas)  vs  + BERT  vs  + BERT barajado (control)")
    print("############################################")

    evaluate_config(X_tab_texto, y, "TABULAR + TEXTO (reglas)  [referencia, ver modelo_sin_bert.py]")
    evaluate_config(X_tab_texto, y, "TABULAR + TEXTO + BERT", bert_embeds=bert_embeds, shuffle_bert=False)
    evaluate_config(
        X_tab_texto, y, "TABULAR + TEXTO + BERT BARAJADO (control)", bert_embeds=bert_embeds, shuffle_bert=True
    )

    print(
        "\nLectura de los tres bloques de arriba:\n"
        "  - Si 'BERT' no le gana de forma consistente a 'BERT BARAJADO' (IC95% solapados), "
        "BERT no está usando información del paciente correcto: es ruido con forma de señal.\n"
        "  - Si 'BERT' no le gana a la referencia sin BERT, la rama no aporta nada que las "
        "features por reglas no den ya, y el veredicto es dejar el texto solo con TabPFN "
        "(modelo_sin_bert.py) y descartar esta rama.\n"
        "  - Mira también la similitud coseno de arriba: si sigue ~0.99 entre clases, cualquier "
        "mejora de AUC que veas aquí es frágil y no vas a poder confiar en que se sostenga fuera "
        "de esta muestra."
    )


if __name__ == "__main__":
    main()
