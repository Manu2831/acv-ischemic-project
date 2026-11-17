"""Construye la tabla tabular limpia a partir del CSV crudo + correcciones verificadas en la nota de ingreso.

NO modifica el CSV crudo ni datos/df_*_acv.csv. Salidas (en esta carpeta):
  tabla_limpia.csv       una fila por paciente: valores del CSV (_csv), valores finales y marcas
  planilla_revision.csv  cada corrección o duda, con la frase de la nota, para que la valide un clínico
  ../datos/tabla_limpia.csv  versión mínima que lee el pipeline de imagen (04_imagen/train.py)

Regla para el NIHSS: el de llegada a la institución según la nota de ingreso (admission_record.txt).
Si la nota no lo menciona, queda el del CSV. Edad: solo se corrige si difiere en 2 años o más.
Uso: python construir_tabla.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parents[1]
RAW = ROOT.parent / "Base de datos pacientes Stroke IA - ACV.csv"

# (id, variable, valor_final, tipo, evidencia en la nota de ingreso)
# tipo: "claro" = la nota es inequívoca | "revisar" = corregido pero conviene que lo confirme un clínico
C = [
    # ---- NIHSS: diferencias grandes (>= 3 puntos) ----
    ("ACV-040", "nihss", 10, "claro", "NIHHS de ingreso 10 puntos (CSV: 120, valor imposible)"),
    ("ACV-014", "nihss", 17, "revisar", "disártrico y afásico con nihss de 17 puntos (la fila del CSV parece copiada de ACV-031)"),
    ("ACV-028", "nihss", 7, "claro", "NIHS SCORE 7 ... IDX ACV isquémico talámico - NIHS 7"),
    ("ACV-069", "nihss", 15, "claro", "con NIHSS 15 puntos se activa código stroke"),
    ("ACV-081", "nihss", 19, "claro", "presenta NIHSS de 19 ... NIHSS 19 puntos al ingreso"),
    ("ACV-094", "nihss", 28, "claro", "hemiparesia derecha 1.2. NIHSS: 28"),
    ("ACV-122", "nihss", 5, "revisar", "NIHSS de 5 puntos, se activa código ACV"),
    ("ACV-162", "nihss", 6, "claro", "déficit neurológico NIHSS 6 puntos ... -- NIHSS 6 puntos"),
    ("ACV-182", "nihss", 5, "revisar", "NIHSS actual: 5 puntos (remisión sin NIHSS especificado)"),
    ("ACV-208", "nihss", 3, "revisar", "con resolución de los síntomas, NIHSS 3 (CSV: 12, ¿NIHSS antes de la resolución?)"),
    ("ACV-247", "nihss", 3, "claro", "(NIHSS3) ... ACV isquémico de ACM derecha, NIHSS de 3"),
    ("ACV-296", "nihss", 2, "revisar", "paciente con nihss 2 ptos"),
    ("ACV-317", "nihss", 5, "revisar", "NIHSS 16 en el sitio de remisión antes de trombólisis; al llegar NIHSS de 5 puntos (CSV: 10, no coincide con ninguno)"),
    ("ACV-344", "nihss", 12, "claro", "Nihss de 12 puntos ... Nihss de ingreso de: 12 puntos"),
    ("ACV-041", "nihss", 7, "claro", "se activa código. NIHSS: 7 puntos"),
    ("ACV-192", "nihss", 9, "claro", "con NIHSS 9 pts y glucometría de 120 mg/dl al ingreso"),
    ("ACV-364", "nihss", 6, "claro", "ECV del despertar con nihss 6 ptos"),
    ("ACV-087", "nihss", 3, "claro", "no asimetría facial, no disartria: NIHSS 3 (CSV: 18)"),
    # ---- NIHSS: diferencias de 1-2 puntos ----
    ("ACV-055", "nihss", 6, "claro", "glasgow 15/15, NIHSS de 6"),
    ("ACV-062", "nihss", 2, "claro", "no disartria o afasia, NIHSS 2 puntos"),
    ("ACV-064", "nihss", 9, "claro", "desviación de comisura labial, NIHSS 9"),
    ("ACV-089", "nihss", 2, "claro", "disartria y desviación de comisura labial (nihss 2)"),
    ("ACV-090", "nihss", 11, "claro", "hemiparesia faciobraquiocrural derecha NIHSS 11"),
    ("ACV-151", "nihss", 5, "claro", "Síndrome neurovascular agudo -- NIHSS 5 puntos"),
    ("ACV-163", "nihss", 6, "claro", "NIHSS 6 de ingreso ... -- NIHSS ingreso 6"),
    ("ACV-181", "nihss", 3, "claro", "ECV isquémico? NIHSS: 3"),
    ("ACV-200", "nihss", 4, "claro", "ACV fuera de ventana en estudio - NIHSS: 4 puntos"),
    ("ACV-201", "nihss", 13, "claro", "escala de NIHSS 13 puntos"),
    ("ACV-254", "nihss", 3, "claro", "NIHSS 3 puntos (dos veces)"),
    ("ACV-270", "nihss", 4, "claro", "Nihss Score 4 puntos (la fila del CSV es idéntica a la de ACV-132)"),
    ("ACV-369", "nihss", 3, "claro", "nihhs de ingreso 3 puntos"),
    ("ACV-379", "nihss", 25, "claro", "NIHSS de ingreso de 25"),
    ("ACV-046", "nihss", 3, "claro", "Al examen físico: NIHSS 3 puntos"),
    ("ACV-036", "nihss", 3, "revisar", "vence gravedad pero no mantiene ... NIHSS 3"),
    ("ACV-123", "nihss", 17, "claro", "resultado de NIHSS de 17 puntos"),
    ("ACV-244", "nihss", 6, "claro", "NIHHS de 6 puntos (tres veces)"),
    ("ACV-291", "nihss", 2, "claro", "NIHSS de ingreso 2 puntos"),
    # ---- Edad (diferencia >= 2 años) ----
    ("ACV-014", "edad", 67, "revisar", "paciente masculino de 67 años (CSV: 77, igual que ACV-031)"),
    ("ACV-019", "edad", 64, "claro", "Paciente femenina de 64 años (CSV: 53)"),
    ("ACV-194", "edad", 66, "claro", "Paciente de 66 años de edad (CSV: 85)"),
    ("ACV-267", "edad", 55, "claro", "femenina adulta media de 55 años (la fila del CSV es idéntica a la de ACV-254)"),
    ("ACV-149", "edad", 74, "claro", "paciente femenina de 74 años (CSV: 72)"),
    ("ACV-180", "edad", 20, "claro", "Paciente de 20 años (CSV: 22)"),
    ("ACV-212", "edad", 58, "claro", "Paciente masculino de 58 años (CSV: 63)"),
    ("ACV-281", "edad", 63, "claro", "Paciente masculino de 63 años (CSV: 68)"),
    # ---- Comorbilidades ----
    ("ACV-162", "fa", 1, "claro", "fibrilación auricular de novo con respuesta ventricular conservada"),
    ("ACV-182", "fa", 1, "claro", "FA paroxística causante de embolismo cerebral"),
    ("ACV-224", "fa", 1, "claro", "ACV isquémico cardioembólico por FA ... anticoagulación rivaroxabán"),
    ("ACV-028", "dm", 1, "claro", "antecedentes de HTA - DM tipo 2"),
    ("ACV-281", "dm", 1, "claro", "antecedente de HTA y DM2 de novo"),
    ("ACV-145", "hta", 1, "claro", "Hombre de 54 años con antecedente de HTA"),
    ("ACV-247", "hta", 1, "claro", "antecedentes de HTA"),
    ("ACV-267", "hta", 1, "claro", "antecedentes de obesidad, HTA, t. ansiedad"),
    ("ACV-267", "sexo", "F", "claro", "FEMENINA adulta media (CSV: masculino, copiado de ACV-254)"),
]

# Dudas que NO se corrigen: quedan con el valor del CSV y van a la planilla.
DUDAS = [
    ("ACV-011", "mrs_egreso", "Misma RM y misma nota que ACV-030, pero mRS 2 vs 5. Se excluyen los dos hasta saber cuál es el correcto."),
    ("ACV-030", "mrs_egreso", "Duplicado de ACV-011 con mRS distinto (5 vs 2). Excluido."),
    ("ACV-269", "fila", "Fila del CSV idéntica a ACV-256 (26 años, hombre, NIHSS 6); la nota no trae edad ni NIHSS y describe un AIT leve. Excluido por no verificable."),
    ("ACV-031", "imagen", "Misma RM que ACV-014, pero las notas son de pacientes distintos (77 años NIHSS 2 vs 67 años NIHSS 17). ¿De quién es la RM?"),
    ("ACV-270", "edad", "Fila idéntica a ACV-132 (76 años); la nota dice 'adulta mayor' sin edad. Además la nota refiere ACV previo a los 32 años (CSV: no)."),
    ("ACV-020", "nihss", "La nota dice 'NIHSS de 41' en un AIT; el CSV dice 4. Probable error de tipeo en la nota. Queda 4."),
    ("ACV-150", "nihss", "Nota: 17 puntos al inicio y luego 15. CSV: 17. Queda 17."),
    ("ACV-149", "nihss", "Nota: NIHSS inicial 10 (sitio de remisión) y 14 al llegar. CSV: 10. Queda 10; definir cuál se usa."),
    ("ACV-372", "nihss", "Nota: 8 en el sitio de remisión y 3 al llegar. CSV: 8. Queda 8; definir cuál se usa."),
    ("ACV-174", "dm", "Nota: 'Diabetes Mellitus tipo II ?' con HbA1c 6,6 %. CSV: no."),
    ("ACV-170", "hta", "Nota: hipertensa al ingreso y se indica losartán; sin antecedente explícito. CSV: no."),
]
EXCLUIR = {"ACV-011": "duplicado con mRS en conflicto", "ACV-030": "duplicado con mRS en conflicto",
           "ACV-269": "fila copiada, no verificable"}

r = pd.read_csv(RAW, encoding="utf-8-sig", dtype=str)
r = r[r.iloc[:, 0].fillna("").str.startswith("ACV")].copy()
r = r.apply(lambda s: s.str.strip())
num = lambda s: pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
sn = lambda s: s.map({"1": 1.0, "2": 0.0})
t = pd.DataFrame({
    "id": r.iloc[:, 0], "edad": num(r.iloc[:, 6]), "sexo": r.iloc[:, 7].map({"1": "F", "2": "M"}),
    "hta": sn(r.iloc[:, 8]), "tabaco": sn(r.iloc[:, 9]), "dm": sn(r.iloc[:, 10]), "dislipidemia": sn(r.iloc[:, 11]),
    "fa": sn(r.iloc[:, 12]), "acv_previo": sn(r.iloc[:, 13]), "tiempo_min": num(r.iloc[:, 15]), "nihss": num(r.iloc[:, 16]),
    "mrs_egreso": num(r.iloc[:, 28]),
}).set_index("id")

split = {}
for s in ("train", "val", "test"):
    for p in (ROOT / "acv_split" / s / "rawdata").iterdir():
        split[p.name] = s
t = t.loc[[i for i in t.index if i in split]].copy()
t.insert(0, "split", [split[i] for i in t.index])
for c in ("edad", "sexo", "hta", "dm", "fa", "nihss"):
    t[c + "_csv"] = t[c]

filas = []
for pid, var, val, tipo, ev in C:
    antes = t.loc[pid, var]
    t.loc[pid, var] = val
    filas.append({"id": pid, "split": split[pid], "variable": var, "valor_csv": antes, "valor_nota": val, "usado": val,
                  "estado": "corregido" if tipo == "claro" else "corregido, confirmar", "evidencia": ev})
for pid, var, ev in DUDAS:
    filas.append({"id": pid, "split": split[pid], "variable": var, "valor_csv": t.loc[pid, var] if var in t.columns else "",
                  "valor_nota": "", "usado": "excluido" if pid in EXCLUIR else "valor del CSV", "estado": "duda sin resolver",
                  "evidencia": ev})
t["excluido"] = [EXCLUIR.get(i, "") for i in t.index]
t["corregido"] = [", ".join(sorted({c[1] for c in C if c[0] == i})) for i in t.index]
t["mal_desenlace"] = (t.mrs_egreso > 2).where(t.mrs_egreso.notna()).astype(float)

t.to_csv(AQUI / "tabla_limpia.csv", encoding="utf-8-sig")
# Versión mínima para el pipeline (04_imagen/train.py la lee de datos/tabla_limpia.csv): solo lo que usa el modelo.
pipe = t.reset_index().rename(columns={"id": "Paciente_id", "edad": "Edad", "nihss": "NIHSS ingreso", "dm": "Diabetes",
                                       "fa": "Fibrilación Auricular", "hta": "Hipertensión", "mrs_egreso": "mRS egreso"})
pipe[["Paciente_id", "split", "Edad", "NIHSS ingreso", "Diabetes", "Fibrilación Auricular", "Hipertensión", "mRS egreso",
      "excluido"]].to_csv(ROOT / "text" / "datos" / "tabla_limpia.csv", index=False, encoding="utf-8-sig")
pl = pd.DataFrame(filas)
pl.to_csv(AQUI / "planilla_revision.csv", index=False, encoding="utf-8-sig")

d = (t.nihss - t.nihss_csv).abs()
print(f"pacientes: {len(t)} | excluidos: {len(EXCLUIR)} | con alguna corrección: {(t.corregido != '').sum()}")
print(f"NIHSS corregidos: {(d > 0).sum()} (>= 3 puntos: {(d >= 3).sum()}) | edad: {(t.edad != t.edad_csv).sum()} | "
      f"FA: {int(t.fa.sum())} (antes {int(t.fa_csv.sum())}) | DM +{int((t.dm - t.dm_csv).sum())} | HTA +{int((t.hta - t.hta_csv).sum())}")
print(f"correcciones a confirmar por un clínico: {(pl.estado == 'corregido, confirmar').sum()} | dudas sin resolver: {(pl.estado == 'duda sin resolver').sum()}")
print("por split:", pl[pl.estado != "duda sin resolver"].groupby("split").id.nunique().to_dict())
