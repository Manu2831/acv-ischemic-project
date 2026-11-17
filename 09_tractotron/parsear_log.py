"""Reconstruye las tablas de Tractotron (BCBToolkit) a partir del log de ejecución (bash -x).

Por cada lesión (ACV-xxx_lesion_MNI) y cada tracto del atlas BCB guarda:
  probabilidad = máximo del mapa de probabilidad del tracto dentro de la lesión (fslstats -R)
  proporcion   = vóxeles de lesión dentro del tracto umbralizado a 0.5 / volumen del tracto
Uso: python parsear_log.py <logTractotron.txt>  -> tractotron_probabilidad.csv, tractotron_proporcion.csv
"""
import re
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
log = Path(sys.argv[1])
re_pat = re.compile(r"^\+\(62\)printf '%s\\t' (\S+)_lesion_MNI$")
re_tract = re.compile(r"Tracts/(\S+?)\.nii(?:\.gz)? -mul ")
re_max = re.compile(r"^\+\(70\)max='\S+ (\S+) '")
re_les = re.compile(r"^\+\(77\)lesTracVol=(\S*)")
re_tv = re.compile(r"^\+\(74\)tractVol=(\S*)")

prob, prop, run = {}, {}, 0
pid = tract = None
tv = lv = None
for line in log.open(encoding="utf-8", errors="replace"):
    line = line.rstrip("\r\n")
    if line.startswith("+(33)proba="):
        run += 1
    m = re_pat.match(line)
    if m:
        pid = m.group(1); continue
    m = re_tract.search(line)
    if m and pid:
        tract = m.group(1); tv = lv = None; continue
    m = re_max.match(line)
    if m and pid and tract:
        prob[(run, pid, tract)] = float(m.group(1)); continue
    m = re_tv.match(line)
    if m: tv = m.group(1); continue
    m = re_les.match(line)
    if m and pid and tract:
        lv = m.group(1)
        try:
            prop[(run, pid, tract)] = float(lv) / float(tv) if tv and float(tv) > 0 else float("nan")
        except ValueError:
            prop[(run, pid, tract)] = float("nan")

def tabla(d):
    s = pd.Series(d); s.index.names = ["corrida", "paciente", "tracto"]
    return s.unstack("tracto")

P, Q = tabla(prob), tabla(prop)
print("corridas:", sorted(P.index.get_level_values(0).unique()))
for r, g in P.groupby(level=0):
    print(f"  corrida {r}: {len(g)} pacientes, {g.shape[1]} tractos")
# Si un paciente aparece en varias corridas, se queda la última.
P = P.reset_index().drop_duplicates("paciente", keep="last").drop(columns="corrida").set_index("paciente")
Q = Q.reset_index().drop_duplicates("paciente", keep="last").drop(columns="corrida").set_index("paciente")
P.sort_index().to_csv(OUT / "tractotron_probabilidad.csv")
Q.sort_index().to_csv(OUT / "tractotron_proporcion.csv")
print("pacientes únicos:", len(P), "| tractos:", P.shape[1])
print("tractos:", ", ".join(P.columns))
