# ---------------------------------------------------------------------------
# ¿La rama de texto aporta señal o ruido? Diagnóstico en 4 partes.
#
# 1. CONTROL EXPERIMENTAL: entrena el mismo modelo tres veces, con
#    (a) textos reales, (b) textos barajados entre pacientes, (c) sin texto.
#    Si (a) no le gana claramente a (b), la rama de texto no está usando
#    información real del paciente: está ajustando ruido.
# 2. CUÁNTO MUEVE EL TEXTO la predicción (|Δ prob| al anular el texto).
# 3. SIMILITUD COSENO entre embeddings: si todos los informes se parecen
#    entre sí, BERT no puede distinguir un paciente de otro.
# 4. INTEGRATED GRADIENTS por token -> HTML con el texto resaltado, para
#    ver en qué palabras se apoya realmente el modelo.
#
# Reutiliza datos/modelo/métricas de Nuevaprueba.py (importarlo carga los
# datos y precalcula los embeddings de BERT, pero NO corre su entrenamiento).
# ---------------------------------------------------------------------------


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # rutas relativas a la carpeta de este script

import copy
import html
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from Nuevaprueba import (
    df, df_val, df_test,
    tokenizer, bert_model, mean_pool, MultimodalModel,
    compute_metrics, find_best_threshold_youden, make_batches,
    emb_train_fixed, emb_val_fixed, emb_test_fixed,
    device, FEATURES, MAX_LEN,
    N_SPLITS, RANDOM_SEED, EPOCHS, BATCH_SIZE, PATIENCE, LR_MODEL, WEIGHT_DECAY,
)

IG_STEPS  = 32
HTML_OUT  = "resultados/interpretabilidad_bert.html"
TOP_WORDS = 25

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

X_tab      = df[FEATURES].values.astype(np.float32)
y          = df["mRS_bin"].values
X_val_tab  = df_val[FEATURES].values.astype(np.float32)
y_val      = df_val["mRS_bin"].values
X_test_tab = df_test[FEATURES].values.astype(np.float32)
y_test     = df_test["mRS_bin"].values

# ---------------------------------------------------------------------------
# 1. CONTROL EXPERIMENTAL — textos reales vs barajados vs sin texto
# ---------------------------------------------------------------------------

def build_text_embeddings(mode, seed=RANDOM_SEED):
    """real = como vienen | shuffled = texto de otro paciente | none = vector cero."""
    if mode == "real":
        return emb_train_fixed.clone(), emb_val_fixed.clone(), emb_test_fixed.clone()
    if mode == "shuffled":
        rng = np.random.default_rng(seed)
        return (emb_train_fixed[rng.permutation(len(emb_train_fixed))].clone(),
                emb_val_fixed[rng.permutation(len(emb_val_fixed))].clone(),
                emb_test_fixed[rng.permutation(len(emb_test_fixed))].clone())
    if mode == "none":
        return (torch.zeros_like(emb_train_fixed),
                torch.zeros_like(emb_val_fixed),
                torch.zeros_like(emb_test_fixed))
    raise ValueError(mode)


def run_experiment(text_mode):
    emb_tr_all, emb_vl_all, emb_te_all = build_text_embeddings(text_mode)
    emb_vl_all = emb_vl_all.to(device)
    emb_te_all = emb_te_all.to(device)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_probs = np.zeros(len(y))
    models, scalers = [], []

    for train_idx, val_idx in skf.split(X_tab, y):
        X_train, X_val_fold = X_tab[train_idx], X_tab[val_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        tab_tr_t = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32).to(device)
        tab_vl_t = torch.tensor(scaler.transform(X_val_fold), dtype=torch.float32).to(device)
        y_tr_t   = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        y_vl_t   = torch.tensor(y[val_idx], dtype=torch.float32).view(-1, 1).to(device)

        emb_tr = emb_tr_all[train_idx].to(device)
        emb_vl = emb_tr_all[val_idx].to(device)

        ratio_fold = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        pos_weight = torch.tensor([ratio_fold ** 0.5], dtype=torch.float32).to(device)

        torch.manual_seed(RANDOM_SEED)
        model = MultimodalModel(n_tab_features=len(FEATURES)).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MODEL, weight_decay=WEIGHT_DECAY)

        best_val_loss, patience_counter, best_state = float("inf"), 0, None
        for epoch in range(EPOCHS):
            model.train()
            for batch_idx in make_batches(len(train_idx), BATCH_SIZE, seed=epoch):
                optimizer.zero_grad()
                loss = criterion(model(tab_tr_t[batch_idx], emb_tr[batch_idx]), y_tr_t[batch_idx])
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(tab_vl_t, emb_vl), y_vl_t).item()
            if val_loss < best_val_loss:
                best_val_loss, patience_counter = val_loss, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof_probs[val_idx] = torch.sigmoid(model(tab_vl_t, emb_vl)).cpu().numpy().flatten()
        models.append(copy.deepcopy(best_state))
        scalers.append(copy.deepcopy(scaler))

    def ensemble(X_ext, emb_ext):
        probs = []
        for state, scaler in zip(models, scalers):
            m = MultimodalModel(n_tab_features=len(FEATURES)).to(device)
            m.load_state_dict(state)
            m.eval()
            tab_t = torch.tensor(scaler.transform(X_ext), dtype=torch.float32).to(device)
            with torch.no_grad():
                probs.append(torch.sigmoid(m(tab_t, emb_ext)).cpu().numpy().flatten())
        return np.mean(probs, axis=0)

    return {
        "mode": text_mode,
        "oof_probs": oof_probs,
        "val_probs": ensemble(X_val_tab, emb_vl_all),
        "test_probs": ensemble(X_test_tab, emb_te_all),
        "models": models,
        "scalers": scalers,
    }


print("\n" + "=" * 72)
print("  1. CONTROL EXPERIMENTAL — ¿el modelo usa información real del texto?")
print("=" * 72)

experiments = {mode: run_experiment(mode) for mode in ["real", "shuffled", "none"]}

etiquetas = {
    "real":     "Texto real",
    "shuffled": "Texto BARAJADO (de otro paciente)",
    "none":     "SIN texto (solo tabular)",
}

print(f"\n  {'Configuración':<38} {'AUC OOF':>8} {'AUC val':>8} {'AUC test':>9} "
      f"{'Spec test':>10} {'FP test':>8}")
print("  " + "-" * 84)

resumen_control = []
for mode, exp in experiments.items():
    thr = find_best_threshold_youden(y, exp["oof_probs"])
    auc_oof  = roc_auc_score(y, exp["oof_probs"])
    auc_val  = roc_auc_score(y_val, exp["val_probs"])
    m_test   = compute_metrics(y_test, exp["test_probs"], threshold=thr)
    print(f"  {etiquetas[mode]:<38} {auc_oof:>8.3f} {auc_val:>8.3f} "
          f"{m_test['auc']:>9.3f} {m_test['spec']:>10.3f} {m_test['fp']:>8d}")
    resumen_control.append({
        "etiqueta": etiquetas[mode], "auc_oof": auc_oof, "auc_val": auc_val,
        "auc_test": m_test["auc"], "spec_test": m_test["spec"], "fp_test": m_test["fp"],
    })

print("\n  Lectura: si 'texto real' no le gana claramente a 'texto barajado',")
print("  la rama de texto no está extrayendo información real del paciente.")

# ---------------------------------------------------------------------------
# 2. ¿CUÁNTO MUEVE EL TEXTO LA PREDICCIÓN?
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("  2. ¿CUÁNTO MUEVE EL TEXTO LA PREDICCIÓN? (anular el texto vs. dejarlo)")
print("=" * 72)

real_exp = experiments["real"]


def probs_con_y_sin_texto(X_ext, emb_ext):
    con, sin = [], []
    for state, scaler in zip(real_exp["models"], real_exp["scalers"]):
        m = MultimodalModel(n_tab_features=len(FEATURES)).to(device)
        m.load_state_dict(state)
        m.eval()
        tab_t = torch.tensor(scaler.transform(X_ext), dtype=torch.float32).to(device)
        with torch.no_grad():
            con.append(torch.sigmoid(m(tab_t, emb_ext)).cpu().numpy().flatten())
            sin.append(torch.sigmoid(m(tab_t, torch.zeros_like(emb_ext))).cpu().numpy().flatten())
    return np.mean(con, axis=0), np.mean(sin, axis=0)


p_con, p_sin = probs_con_y_sin_texto(X_test_tab, emb_test_fixed)
delta = np.abs(p_con - p_sin)
print(f"\n  |Δ probabilidad| al anular el texto (test, n={len(y_test)}):")
print(f"    media = {delta.mean():.4f} | máx = {delta.max():.4f} | "
      f"rango de las probs = [{p_con.min():.3f}, {p_con.max():.3f}]")
corr_con_outcome = np.corrcoef(p_con - p_sin, y_test)[0, 1]
print(f"    correlación entre el empujón del texto y el outcome real: {corr_con_outcome:+.3f}")
print("    (cerca de 0 = el texto mueve la predicción, pero en la dirección equivocada)")

# ---------------------------------------------------------------------------
# 3. ¿QUÉ TAN DISTINGUIBLES SON LOS TEXTOS PARA BERT?
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("  3. SIMILITUD ENTRE EMBEDDINGS — ¿BERT distingue un paciente de otro?")
print("=" * 72)

emb_np = emb_train_fixed.numpy()
emb_norm = emb_np / np.linalg.norm(emb_np, axis=1, keepdims=True)
cos_sim = emb_norm @ emb_norm.T
triu = cos_sim[np.triu_indices_from(cos_sim, k=1)]

print(f"\n  Similitud coseno entre los {len(emb_np)} informes de train:")
print(f"    media = {triu.mean():.4f} | mín = {triu.min():.4f} | máx = {triu.max():.4f}")
print("    (valores ~0.95+ = los informes son casi idénticos para BERT,")
print("     así que el embedding casi no distingue un paciente de otro)")

sim_intra_0 = cos_sim[np.ix_(y == 0, y == 0)][np.triu_indices((y == 0).sum(), k=1)].mean()
sim_intra_1 = cos_sim[np.ix_(y == 1, y == 1)][np.triu_indices((y == 1).sum(), k=1)].mean()
sim_inter   = cos_sim[np.ix_(y == 0, y == 1)].mean()
print(f"\n  Similitud media dentro de 'buen outcome' : {sim_intra_0:.4f}")
print(f"  Similitud media dentro de 'mal outcome'  : {sim_intra_1:.4f}")
print(f"  Similitud media ENTRE las dos clases     : {sim_inter:.4f}")
print("  (si los tres números son casi iguales, no hay separación entre clases)")

# ---------------------------------------------------------------------------
# 4. INTEGRATED GRADIENTS — ¿en qué palabras se apoya el modelo?
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("  4. INTEGRATED GRADIENTS por token")
print("=" * 72)

embed_layer = bert_model.get_input_embeddings()


def integrated_gradients(text, tab_row, n_steps=IG_STEPS):
    """Atribución por token, promediada sobre los N modelos del ensemble."""
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        inputs_embeds = embed_layer(input_ids)
    baseline = torch.zeros_like(inputs_embeds)

    attr_total = torch.zeros(input_ids.shape[1], device=device)

    for state, scaler in zip(real_exp["models"], real_exp["scalers"]):
        m = MultimodalModel(n_tab_features=len(FEATURES)).to(device)
        m.load_state_dict(state)
        m.eval()
        tab_t = torch.tensor(scaler.transform(tab_row.reshape(1, -1)),
                             dtype=torch.float32).to(device)

        grads = torch.zeros_like(inputs_embeds)
        for alpha in np.linspace(1.0 / n_steps, 1.0, n_steps):
            interp = (baseline + alpha * (inputs_embeds - baseline)).detach().requires_grad_(True)
            out    = bert_model(inputs_embeds=interp, attention_mask=attention_mask)
            pooled = mean_pool(out.last_hidden_state, attention_mask)
            logit  = m(tab_t, pooled)
            grads += torch.autograd.grad(logit.sum(), interp)[0]

        avg_grads = grads / n_steps
        attr_total += ((inputs_embeds - baseline) * avg_grads).sum(dim=-1).squeeze(0)

    attr = (attr_total / len(real_exp["models"])).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return tokens, attr


def merge_wordpieces(tokens, scores):
    words, wscores = [], []
    for tok, sc in zip(tokens, scores):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if tok.startswith("##") and words:
            words[-1] += tok[2:]
            wscores[-1] += sc
        else:
            words.append(tok)
            wscores.append(float(sc))
    return words, np.array(wscores)


def highlight_html(words, scores):
    max_abs = np.abs(scores).max() or 1.0
    spans = []
    for w, s in zip(words, scores):
        alpha = abs(s) / max_abs
        rgb = "220,53,40" if s > 0 else "40,90,220"
        spans.append(
            f'<span class="tok" style="background-color: rgba({rgb},{alpha:.2f})" '
            f'title="{s:+.4f}">{html.escape(w)}</span>'
        )
    return " ".join(spans)


# Elegimos casos ilustrativos del test: falso positivo, verdadero positivo, verdadero negativo
thr_real = find_best_threshold_youden(y, real_exp["oof_probs"])
probs_test = real_exp["test_probs"]
preds_test = (probs_test > thr_real).astype(int)

casos = []
for etiqueta, cond in [
    ("FALSO POSITIVO (el problema que querés corregir)", (preds_test == 1) & (y_test == 0)),
    ("Verdadero positivo", (preds_test == 1) & (y_test == 1)),
    ("Verdadero negativo", (preds_test == 0) & (y_test == 0)),
]:
    idxs = np.where(cond)[0]
    if len(idxs):
        i = int(idxs[np.argmax(np.abs(probs_test[idxs] - thr_real))])
        casos.append((etiqueta, i))

bloques_html = []
for etiqueta, i in casos:
    pid  = df_test.iloc[i]["Paciente_id"]
    text = df_test.iloc[i]["text"]
    print(f"  Atribuyendo {pid} ({etiqueta})...")
    tokens, attr = integrated_gradients(text, X_test_tab[i])
    words, wscores = merge_wordpieces(tokens, attr)
    bloques_html.append({
        "etiqueta": etiqueta, "pid": pid,
        "real": int(y_test[i]), "prob": float(probs_test[i]),
        "html": highlight_html(words, wscores),
    })

# Ranking global de palabras sobre todo el test
print("  Calculando ranking global de palabras sobre el test completo...")
acumulado = {}
for i in range(len(df_test)):
    tokens, attr = integrated_gradients(df_test.iloc[i]["text"], X_test_tab[i])
    words, wscores = merge_wordpieces(tokens, attr)
    if np.abs(wscores).max() > 0:
        wscores = wscores / np.abs(wscores).max()
    for w, s in zip(words, wscores):
        acumulado.setdefault(w.lower(), []).append(s)

ranking = sorted(acumulado.items(), key=lambda kv: -np.mean(np.abs(kv[1])))[:TOP_WORDS]

print(f"\n  Top {TOP_WORDS} palabras más influyentes (|atribución| media, test completo):")
print(f"    {'palabra':<20} {'|attr| media':>13} {'apariciones':>12}")
print("    " + "-" * 47)
for w, vals in ranking:
    print(f"    {w:<20} {np.mean(np.abs(vals)):>13.4f} {len(vals):>12d}")

# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

filas_control = "".join(
    f"<tr><td>{html.escape(r['etiqueta'])}</td><td>{r['auc_oof']:.3f}</td>"
    f"<td>{r['auc_val']:.3f}</td><td>{r['auc_test']:.3f}</td>"
    f"<td>{r['spec_test']:.3f}</td><td>{r['fp_test']}</td></tr>"
    for r in resumen_control
)

filas_ranking = "".join(
    f"<tr><td>{html.escape(w)}</td><td>{np.mean(np.abs(v)):.4f}</td><td>{len(v)}</td></tr>"
    for w, v in ranking
)

bloques = "".join(
    f"""
    <div class="caso">
      <h3>{html.escape(b['etiqueta'])}</h3>
      <p class="meta">Paciente <strong>{html.escape(str(b['pid']))}</strong> &middot;
         outcome real: <strong>{'mal outcome (1)' if b['real'] else 'buen outcome (0)'}</strong> &middot;
         probabilidad predicha: <strong>{b['prob']:.3f}</strong> (umbral {thr_real:.3f})</p>
      <div class="texto">{b['html']}</div>
    </div>"""
    for b in bloques_html
)

html_doc = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Diagnóstico de la rama de texto (PubMedBERT)</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 900px;
         margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #1a1a1a; }}
  h1 {{ font-size: 1.6rem; }} h2 {{ font-size: 1.2rem; margin-top: 2.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
  th, td {{ border: 1px solid #ddd; padding: 7px 10px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #f5f5f5; }}
  .texto {{ background: #fafafa; border: 1px solid #e5e5e5; border-radius: 6px;
            padding: 16px; line-height: 2.1; margin-top: 8px; }}
  .tok {{ padding: 2px 1px; border-radius: 3px; }}
  .meta {{ color: #555; font-size: 0.9rem; margin: 4px 0; }}
  .caso {{ margin-bottom: 2rem; }}
  .nota {{ background: #fff8e1; border-left: 4px solid #f0c040; padding: 12px 16px;
           margin: 1rem 0; font-size: 0.92rem; }}
  code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px; }}
</style></head><body>

<h1>¿La rama de texto aporta señal o ruido?</h1>
<p class="meta">PubMedBERT congelado + 5 variables tabulares &middot;
   109 train / 24 val / 24 test</p>

<h2>1. Control experimental</h2>
<p>El mismo modelo entrenado tres veces, cambiando solo qué texto recibe cada paciente.</p>
<table>
  <tr><th>Configuración</th><th>AUC OOF</th><th>AUC val</th><th>AUC test</th>
      <th>Especificidad test</th><th>FP test</th></tr>
  {filas_control}
</table>
<div class="nota">Si <em>texto real</em> no le gana claramente a <em>texto barajado</em>,
la rama de texto no está extrayendo información real del paciente: está ajustando ruido.</div>

<h2>2. Cuánto mueve el texto la predicción</h2>
<p>En test, anular el texto cambia la probabilidad predicha en promedio
   <strong>{delta.mean():.4f}</strong> (máximo {delta.max():.4f}).
   La correlación entre ese empujón y el outcome real es
   <strong>{corr_con_outcome:+.3f}</strong>.</p>

<h2>3. ¿BERT distingue un paciente de otro?</h2>
<p>Similitud coseno entre los {len(emb_np)} informes de train:
   media <strong>{triu.mean():.4f}</strong>, mín {triu.min():.4f}, máx {triu.max():.4f}.</p>
<table>
  <tr><th>Comparación</th><th>Similitud coseno media</th></tr>
  <tr><td>Dentro de "buen outcome"</td><td>{sim_intra_0:.4f}</td></tr>
  <tr><td>Dentro de "mal outcome"</td><td>{sim_intra_1:.4f}</td></tr>
  <tr><td>Entre las dos clases</td><td>{sim_inter:.4f}</td></tr>
</table>

<h2>4. ¿En qué palabras se apoya el modelo?</h2>
<p>Integrated Gradients sobre los tokens, promediado sobre los {N_SPLITS} modelos del ensemble.
   <span style="background: rgba(220,53,40,0.55); padding: 2px 6px; border-radius: 3px;">Rojo</span>
   empuja hacia "mal outcome",
   <span style="background: rgba(40,90,220,0.55); padding: 2px 6px; border-radius: 3px;">azul</span>
   hacia "buen outcome". La intensidad es la magnitud de la atribución.</p>
{bloques}

<h3>Palabras más influyentes en todo el test</h3>
<p class="meta">Si las primeras posiciones son artículos, preposiciones o puntuación en vez de
   NIHSS, edad o hallazgos clínicos, el modelo no está leyendo medicina: está leyendo ruido.</p>
<table>
  <tr><th>Palabra</th><th>|atribución| media</th><th>Apariciones</th></tr>
  {filas_ranking}
</table>

</body></html>"""

with open(HTML_OUT, "w", encoding="utf-8") as f:
    f.write(html_doc)

print(f"\n  Reporte HTML escrito en: {HTML_OUT}")
