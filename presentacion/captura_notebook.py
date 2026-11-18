"""Genera figuras/notebook_celda7.png: captura de la celda de síntesis de entrega3_resultados.ipynb.

Exporta esa celda (con su salida) a HTML y la fotografía con Chrome en modo headless; luego recorta la tabla y el
gráfico. Uso: python captura_notebook.py  (requiere nbconvert y Google Chrome)
"""
import subprocess
import tempfile
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from PIL import Image

AQUI = Path(__file__).resolve().parent
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
nb = nbformat.read(AQUI.parent / "entrega3_resultados.ipynb", 4)
i = next(k for k, c in enumerate(nb.cells) if c.cell_type == "code" and "sintesis = pd.DataFrame" in c.source)
sub = nbformat.v4.new_notebook()
sub.cells, sub.metadata = [nb.cells[i - 1], nb.cells[i]], nb.metadata
html, _ = HTMLExporter(template_name="lab").from_notebook_node(sub)
tmp = Path(tempfile.mkdtemp())
(tmp / "celda.html").write_text(html.replace("</head>", "<style>body{zoom:0.9}</style></head>"), encoding="utf-8")
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1100,1700",
                f"--screenshot={tmp / 'celda.png'}", (tmp / "celda.html").as_uri()], check=True, capture_output=True)
im = Image.open(tmp / "celda.png").convert("RGB")
# recorte: tabla de la síntesis + gráfico (debajo del código)
im.crop((70, 525, 790, 1135)).save(AQUI / "figuras" / "notebook_celda7.png")
print("ok:", AQUI / "figuras" / "notebook_celda7.png")
