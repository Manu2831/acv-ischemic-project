"""Arma el deck de la Entrega 3 a partir del de la Entrega 2 ("Avance_proyecto_mRS_en_ACV_copia_ajustada_v2.pptx").

Sigue el orden pedido para la entrega: título, objetivo, dataset (descripción y descarga), mejor modelo (pipeline
explicado), resultados (resumen de la Entrega 2 y luego lo nuevo), análisis y conclusiones, notebook.

- Corrige errores de maquetación (portada, escala NIHSS, matrices y leyenda de la antigua diap. 13)
- Corrige números de la tabla de fusiones y de la matriz de confusión contra resultados_acv_p/resumen.csv
- Agrega: dataset, mejor modelo, resumen de la Entrega 2, cambios de la Entrega 3, controles, auditoría de datos,
  ubicación, desconexión, síntesis y notebook de la Entrega 3
- Imágenes: pipeline del mejor modelo (de la diap. 4), diagrama nativo de la Entrega 3 y captura del notebook
- Los resultados detallados de la Entrega 2 pasan a un anexo al final

Uso: python mejorar_deck.py [entrada.pptx] [salida.pptx]
No modifica el archivo de entrada. Requiere python-pptx y Pillow.
"""
import copy
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Pt

AQUI = Path(__file__).resolve().parent
TXT = AQUI.parent  # carpeta raíz del repositorio (text/)
ENTRADA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads/Avance_proyecto_mRS_en_ACV_copia_ajustada_v2.pptx"
SALIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "Downloads/Avance_proyecto_mRS_en_ACV_entrega3.pptx"

NAVY, MORADO, MAGENTA, INDIGO = "040544", "5F2772", "CE57C1", "3D41B0"
TARJ, TEXTO, GRIS, VIOLETA, LILA = "F7F4FD", "1F1F45", "4A4A6E", "6A47C2", "A88BE0"
TIT, CUERPO = "League Spartan", "Spectral"
R_EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"

prs = Presentation(ENTRADA)
orig = list(prs.slides)
assert len(orig) == 16, f"se esperaban las 16 diapositivas de la Entrega 2, hay {len(orig)}"


def duplicar_marco(src):
    """Diapositiva nueva con el marco de `src` (barra, logo, pie, número, sección y título: shapes 2–8)."""
    sl = prs.slides.add_slide(src.slide_layout)
    for sh in list(sl.shapes):
        sh._element.getparent().remove(sh._element)
    for sh in src.shapes:
        if sh.shape_id > 8:
            continue
        el = copy.deepcopy(sh._element)
        for blip in el.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
            rid = blip.get(R_EMBED)
            parte = src.part.related_part(rid)
            blip.set(R_EMBED, sl.part.relate_to(parte, src.part.rels[rid].reltype))
        sl.shapes._spTree.append(el)
    return sl


# Diapositivas nuevas, en el orden en que se crean (el orden final se fija al final)
NOMBRES = ["controles", "auditoria", "ubicacion", "tractos", "sintesis",
           "dataset", "mejor", "resumen_e2", "cambios_e3", "notebook_e3"]
N = {k: duplicar_marco(orig[10]) for k in NOMBRES}
nuevas = [N[k] for k in NOMBRES[:5]]


def rgb(h):
    return RGBColor.from_string(h)


def caja(sl, x, y, w, h, parrafos, anchor=MSO_ANCHOR.TOP, nombre=None):
    """parrafos: lista de (lista de (texto, tamaño, negrita, color, fuente), alineación)."""
    t = sl.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h))
    if nombre:
        t.name = nombre
    tf = t.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, (runs, al) in enumerate(parrafos):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = al
        if i:
            p.space_before = Pt(4)
        for txt, sz, b, col, fnt in runs:
            r = p.add_run()
            r.text = txt
            r.font.size, r.font.bold, r.font.name = Pt(sz), b, fnt
            r.font.color.rgb = rgb(col)
    return t


def P(txt, sz=13, b=False, col=TEXTO, fnt=CUERPO, al=PP_ALIGN.LEFT):
    return ([(txt, sz, b, col, fnt)], al)


def rect(sl, x, y, w, h, fill, forma=MSO_SHAPE.RECTANGLE):
    s = sl.shapes.add_shape(forma, Pt(x), Pt(y), Pt(w), Pt(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def linea(sl, x, y, w, h, col=INDIGO):
    from pptx.enum.shapes import MSO_CONNECTOR
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Pt(x), Pt(y), Pt(x + w), Pt(y + h))
    c.line.color.rgb = rgb(col)
    c.line.width = Pt(3)
    return c


def tarjeta(sl, x, y, w, h, cifra, etiqueta, col=INDIGO):
    """Tarjeta con cifra grande, como las de la diap. 11."""
    rect(sl, x, y, w, h, TARJ)
    linea(sl, x, y + 1.5, w, 0, col)
    caja(sl, x + 10, y + 10, w - 20, 28, [P(cifra, 22, True, NAVY, TIT)])
    caja(sl, x + 10, y + 40, w - 20, h - 46, [P(etiqueta, 12, False, GRIS)])


def vineta(sl, x, y, w, h, negrita, resto, col=INDIGO):
    """Viñeta con línea vertical a la izquierda, como en la diap. 11."""
    linea(sl, x, y, 0, h, col)
    caja(sl, x + 12, y, w - 12, h, [([(negrita, 13, True, TEXTO, CUERPO), (resto, 13, False, TEXTO, CUERPO)], PP_ALIGN.LEFT)])


def set_texto(shape, txt):
    """Cambia el texto conservando el formato del primer run."""
    p = shape.text_frame.paragraphs[0]
    p.runs[0].text = txt
    for r in p.runs[1:]:
        r.text = ""
    for extra in shape.text_frame.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)


def reemplazar(sl, viejo, nuevo):
    for sh in sl.shapes:
        if not sh.has_text_frame:
            continue
        for p in sh.text_frame.paragraphs:
            if viejo in p.text:
                t = p.text.replace(viejo, nuevo)
                p.runs[0].text = t
                for r in p.runs[1:]:
                    r.text = ""


def por_id(sl, i):
    return next(s for s in sl.shapes if s.shape_id == i)


def mover(sh, x=None, y=None, w=None, h=None):
    if x is not None: sh.left = Pt(x)
    if y is not None: sh.top = Pt(y)
    if w is not None: sh.width = Pt(w)
    if h is not None: sh.height = Pt(h)


def estilo_grafico(ch, sz=11):
    ch.has_title = False
    for se in ch.plots[0].series:
        se.invert_if_negative = False
    ch.font.name, ch.font.size = CUERPO, Pt(sz)
    ch.font.color.rgb = rgb(TEXTO)


def notas(sl, txt):
    ns = sl.notes_slide
    if ns.notes_text_frame is None:
        # Las diapositivas duplicadas reciben notas sin cuerpo: se copia el cuerpo de las notas de la diap. 11.
        cuerpo = next(sh for sh in orig[10].notes_slide.shapes if sh.is_placeholder and sh.placeholder_format.type == 2)
        ns.shapes._spTree.append(copy.deepcopy(cuerpo._element))
    tf = ns.notes_text_frame
    tf.text = txt


# ---------- 2. Correcciones en diapositivas existentes ----------
# Portada: el título se salía por la izquierda
mover(por_id(orig[0], 7), x=40, y=110, w=880, h=80)

# Escala NIHSS: números y etiquetas angostos se partían
s2 = orig[1]
for tarjeta_id, num_id, lab_id in [(21, 22, 23), (24, 25, 26), (27, 28, 29), (30, 31, 32), (33, 34, 35)]:
    t = por_id(s2, tarjeta_id)
    for i in (num_id, lab_id):
        sh = por_id(s2, i)
        sh.left, sh.width = t.left, t.width
        sh.text_frame.word_wrap = False
        for p in sh.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER

# Etapa B: texto y números de la tabla según resumen.csv
reemplazar(orig[5], " variables clinicas, un token por variable", "5 variables clínicas, un token por variable,")
reemplazar(orig[5], " variables clinicas, un token por variable", "5 variables clínicas, un token por variable,")
s12 = orig[11]
set_texto(next(s for s in s12.shapes if s.has_text_frame and "Comparacion" in s.text_frame.text), "Etapa B: comparación de tipos de fusión")
tabla = next(s for s in s12.shapes if s.has_table).table
filas = {
    "Regresión logística (clínico)": ["0,855 [0,73–0,95]", "0,750", "0,415", "0,138"],
    "Transformer tabular": ["0,861 [0,74–0,95]", "0,750", "0,415", "0,132"],
    "Solo imagen": ["0,687 [0,50–0,86]", "0,500", "0,000", "0,182"],
    "Concat": ["0,858 [0,74–0,95]", "0,776", "0,452", "0,136"],
    "FiLM": ["0,850 [0,73–0,95]", "0,763", "0,434", "0,135"],
    "Cross-attention": ["0,842 [0,71–0,94]", "0,816", "0,513", "0,136"],
    "Gated · mejor": ["0,863 [0,74–0,96]", "0,842", "0,558", "0,124"],
}
trs = tabla._tbl.tr_lst
vacia = next(tr for tr in trs[1:] if not "".join(tr.itertext()).strip())
nueva = copy.deepcopy(next(tr for tr in trs if "Transformer tabular" in "".join(tr.itertext())))
vacia.addprevious(nueva)
vacia.getparent().remove(vacia)
for k, row in enumerate(tabla.rows):
    if k == 0:
        continue
    nombre = row.cells[0].text_frame.text.strip()
    if k == 1:
        nombre = "Regresión logística (clínico)"
    for j, v in enumerate([nombre] + filas[nombre]):
        tf = row.cells[j].text_frame
        tf.paragraphs[0].runs[0].text = v

# Diap. 13: matrices de confusión y leyenda
s13 = orig[12]
set_texto(por_id(s13, 43), "Modelo clínico · reg. logística")
for celda, num, lab in [(44, 45, 46), (47, 48, 49), (50, 51, 52), (53, 54, 55), (57, 58, 59), (60, 61, 62), (63, 64, 65), (66, 67, 68)]:
    c = por_id(s13, celda)
    for i in (num, lab):
        sh = por_id(s13, i)
        sh.left, sh.width = c.left, c.width
        sh.text_frame.word_wrap = False
        for p in sh.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
set_texto(por_id(s13, 58), "10")
set_texto(por_id(s13, 64), "12")
for cuadro, texto, x, w in [(33, 34, 64, 55), (35, 36, 150, 80), (37, 38, 256, 80), (39, 40, 362, 60)]:
    mover(por_id(s13, cuadro), x=x)
    mover(por_id(s13, texto), x=x + 19, w=w)
    por_id(s13, texto).text_frame.word_wrap = False
mover(por_id(s13, 41), y=410, w=300)
caja(s13, 488, 302, 408, 80, [
    ([("Gated: ", 13, True, TEXTO, CUERPO), ("los mismos 10 aciertos con 7 falsas alarmas menos (19 → 12).", 13, False, TEXTO, CUERPO)], PP_ALIGN.LEFT),
    ([("Pero con n = 48 no es concluyente, y los controles de la Entrega 3 muestran que la imagen casi no pesa.", 12, False, GRIS, CUERPO)], PP_ALIGN.LEFT),
])

# Diap. 14: conclusiones y próximos pasos
s14 = orig[13]
set_texto(next(s for s in s14.shapes if s.has_text_frame and s.text_frame.text.startswith("Lo que sabemos")), "Lo que sabemos y lo que sigue")
set_texto(por_id(s14, 9), "Lo que sabemos hoy")
mover(por_id(s14, 9), y=146)
grupos = [(10, 11, 12, 13), (14, 15, 16, 17), (18, 19, 20, 21)]
g4 = []
for i in grupos[2]:
    el = copy.deepcopy(por_id(s14, i)._element)
    el.xpath("./*[1]/p:cNvPr")[0].set("id", str(900 + i))
    s14.shapes._spTree.append(el)
    g4.append(s14.shapes[-1])
items_izq = [
    "El encoder sí ve la lesión en DWI (Dice 0,745); el ADC sigue rezagado (0,517)",
    "Ninguna forma de usar la imagen supera a edad + NIHSS: ni la fusión, ni la ubicación, ni los tractos",
    "Los controles lo confirman: la imagen barajada da el mismo AUC y la compuerta le da solo 9–15 %",
    "Corregir los datos (un NIHSS de 120) aportó más que la imagen: +0,05 de AUC",
]
filas_g = [[por_id(s14, i) for i in g] for g in grupos] + [g4]
for k, ((caja_, circ, ico, txt), t) in enumerate(zip(filas_g, items_izq)):
    y = 178 + k * 76
    mover(caja_, y=y, h=68)
    mover(circ, y=y + 20)
    mover(ico, y=y + 26)
    mover(txt, y=y + 6, h=56)
    txt.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    set_texto(txt, t)
set_texto(por_id(s14, 23), "Próximos pasos")
mover(por_id(s14, 23), y=166)
pasos = [
    "Escribir el resultado negativo, bien controlado (guía TRIPOD)",
    "Validar con el clínico las 8 correcciones y 11 dudas de la tabla limpia",
    "Subgrupo donde el NIHSS falla: sedados o NIHSS bajo con lesión grande",
    "Completar el mRS a 90 días (hoy 67 de 157) y desenlaces motor y de lenguaje",
    "Ampliar la cohorte: unas 3–4 veces más eventos para ver un efecto pequeño",
]
for k, (n_id, t_id) in enumerate([(24, 25), (26, 27), (28, 29), (30, 31), (32, 33)]):
    y = 200 + k * 56
    mover(por_id(s14, n_id), y=y - 3)
    mover(por_id(s14, t_id), y=y, h=44)
    set_texto(por_id(s14, t_id), pasos[k])
notas(s14, "[1 min] ¿Qué sabemos hoy? Uno: la Etapa A funciona; el encoder localiza la lesión en DWI con Dice 0,745, "
      "aunque el núcleo en ADC sigue en 0,517. Dos: ninguna forma de usar la imagen supera a edad más NIHSS; lo probamos "
      "con la fusión profunda, con la ubicación de la lesión y con la desconexión de tractos. Tres: los controles lo "
      "confirman; con la imagen barajada entre pacientes el AUC es el mismo, y la compuerta le da a la imagen solo entre "
      "9 y 15 % del peso. Cuatro: corregir un error de digitación en la tabla aportó más que toda la imagen. Próximos pasos: "
      "escribir esto como un resultado negativo bien controlado; cerrar la revisión de la tabla con el clínico; estudiar el "
      "subgrupo donde el NIHSS no es confiable; completar el mRS a 90 días y desenlaces más específicos; y ampliar la "
      "cohorte, porque con 33 eventos no podemos descartar un efecto pequeño.")

# ---------- 3. Diapositivas nuevas ----------
for sl in nuevas:
    for sh in list(sl.shapes):
        if sh.shape_id >= 9:
            sh._element.getparent().remove(sh._element)


def encabezado(sl, seccion, titulo):
    set_texto(por_id(sl, 7), seccion)
    set_texto(por_id(sl, 8), titulo)


# 3a. Controles
sl = nuevas[0]
encabezado(sl, "6 · ¿LA IMAGEN APORTA? CONTROLES", "Los controles: la imagen casi no se usa")
caja(sl, 64, 142, 480, 22, [P("AUC en ext (n = 48): imagen real frente a barajada", 14, True, MORADO, TIT)])
cats = ["Solo clínica · reg. logística", "Imagen + tabla · concat", "Imagen + tabla · imagen barajada", "Solo imagen", "Solo imagen · barajada"]
vals = [0.855, 0.858, 0.858, 0.687, 0.516]
cols = [NAVY, VIOLETA, MAGENTA, VIOLETA, MAGENTA]
cd = CategoryChartData()
cd.categories = cats[::-1]
cd.add_series("AUC ext", vals[::-1])
ch = sl.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Pt(64), Pt(168), Pt(480), Pt(270), cd).chart
estilo_grafico(ch)
ch.has_legend = False
pl = ch.plots[0]
pl.gap_width = 60
pl.has_data_labels = True
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = "0.000", False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size, pl.data_labels.font.bold = Pt(12), True
for i, c in enumerate(cols[::-1]):
    pt = pl.series[0].points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = rgb(c)
va = ch.value_axis
va.minimum_scale, va.maximum_scale, va.major_unit = 0.4, 0.9, 0.1
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = rgb("E4DEF2")
va.tick_labels.number_format, va.tick_labels.number_format_is_linked = "0.0", False
va.format.line.fill.background()
ch.category_axis.format.line.color.rgb = rgb("C9C3DA")
for x, c, t in [(64, NAVY, "solo clínica"), (170, VIOLETA, "imagen real"), (276, MAGENTA, "imagen barajada")]:
    rect(sl, x, 446, 11, 11, c)
    caja(sl, x + 16, 444, 100, 16, [P(t, 11, False, GRIS)])
tarjeta(sl, 590, 150, 306, 86, "0,858 = 0,858", "Concat con la imagen real y con la imagen de otro paciente: idénticos")
tarjeta(sl, 590, 250, 306, 86, "9–15 %", "Peso que la compuerta (Gated) le da a la imagen; el resto va a la tabla")
tarjeta(sl, 590, 350, 306, 100, "3 de 5 folds", "La mejor época es la 5, el fin del calentamiento: después la red solo sobreajusta")
notas(sl, "[45 s] Antes de creerle a la fusión hicimos controles. Si barajamos las imágenes entre pacientes, es decir, "
      "a cada paciente le damos la resonancia de otro, el modelo de concatenación da exactamente el mismo AUC, 0,858. "
      "La imagen sola baja a 0,52 con la imagen barajada, o sea que algo ve, pero en compañía de la tabla no lo usa. "
      "La compuerta lo dice explícitamente: le da a la imagen entre 9 y 15 % del peso. Y en 3 de los 5 folds la mejor "
      "época es la 5, el fin del calentamiento: después de eso la red solo memoriza train. Conclusión: el AUC de las "
      "fusiones viene de las variables clínicas.")

# 3b. Auditoría de datos
sl = nuevas[1]
encabezado(sl, "6 · ¿LA IMAGEN APORTA? AUDITORÍA DE DATOS", "Un error de digitación pesaba más que la imagen")
caja(sl, 64, 142, 400, 22, [P("Auditoría de la tabla contra las notas de ingreso", 14, True, MORADO, TIT)])
hall = [
    ("NIHSS = 120 ", "en un paciente de train; el máximo posible es 42 (la nota dice 10)"),
    ("Misma RM con dos IDs ", "y mRS en conflicto (2 frente a 5): excluidos"),
    ("Fila clínica copiada ", "con otra RM, no verificable: excluida"),
    ("46 pacientes corregidos ", "desde la nota (37 NIHSS, 8 edades, FA 7 → 10); 8 correcciones y 11 dudas esperan al clínico"),
]
for k, (b, r) in enumerate(hall):
    vineta(sl, 66, 174 + k * 66, 390, 56, b, r, INDIGO if k else MAGENTA)
caja(sl, 500, 142, 396, 22, [P("AUC de edad + NIHSS · CV repetida 5×50, todos", 14, True, MORADO, TIT)])
cd = CategoryChartData()
cd.categories = ["Tabla original", "Solo corregir el 120", "Tabla limpia"]
cd.add_series("AUC", [0.779, 0.827, 0.823])
ch = sl.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Pt(500), Pt(168), Pt(396), Pt(232), cd).chart
estilo_grafico(ch)
ch.has_legend = False
pl = ch.plots[0]
pl.gap_width = 80
pl.has_data_labels = True
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = "0.000", False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size, pl.data_labels.font.bold = Pt(13), True
for i, c in enumerate([LILA, NAVY, NAVY]):
    pt = pl.series[0].points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = rgb(c)
va = ch.value_axis
va.minimum_scale, va.maximum_scale, va.major_unit = 0.70, 0.85, 0.05
va.major_gridlines.format.line.color.rgb = rgb("E4DEF2")
va.tick_labels.number_format, va.tick_labels.number_format_is_linked = "0.00", False
va.format.line.fill.background()
caja(sl, 500, 408, 396, 60, [P("Solo el 120 mueve el AUC (+0,05). Las demás diferencias entre nota y tabla no cambian el desempeño: "
                              "pueden ser otro evaluador, no necesariamente errores.", 12, False, GRIS)])
notas(sl, "[45 s] En paralelo auditamos la tabla clínica contra las notas de ingreso de los 160 pacientes. Encontramos un "
      "NIHSS de 120 en un paciente de train, cuando el máximo posible es 42; la nota dice 10. Una misma resonancia "
      "aparecía con dos identificadores y mRS distintos, y una fila clínica estaba copiada; esos tres se excluyeron. En "
      "total corregimos 46 pacientes desde la nota, y 19 casos esperan revisión del clínico. Lo importante está a la "
      "derecha: corregir solo el 120 sube el AUC del modelo clínico de 0,78 a 0,83; las demás correcciones no lo mueven. "
      "Un solo dígito mal puesto pesaba más que cualquier cosa que aportó la imagen.")

# 3c. Ubicación de la lesión
sl = nuevas[2]
encabezado(sl, "6 · ¿LA IMAGEN APORTA? UBICACIÓN DE LA LESIÓN", "¿Importa dónde está la lesión?")
fig = TXT / "08_desconexion/resultados/mapa_frecuencia_mni.png"
rec = AQUI / "figuras/mapa_mni_recorte.png"
rec.parent.mkdir(exist_ok=True)
im = Image.open(fig)
im.crop((0, 75, im.width, im.height)).save(rec)
w_img = 560
h_img = w_img * (im.height - 75) / im.width
sl.shapes.add_picture(str(rec), Pt(64), Pt(146), Pt(w_img), Pt(h_img))
caja(sl, 64, 146 + h_img + 6, w_img, 30, [P("Frecuencia de lesión en MNI152 (registro afín, Dice mediano 0,948); arriba buen desenlace (n = 122), "
                                            "al medio mal (n = 33), abajo la diferencia. Celeste: corticoespinal. Descriptivo.", 10, False, GRIS)])
caja(sl, 646, 150, 250, 22, [P("Qué dice", 14, True, MORADO, TIT)])
for k, (b, r) in enumerate([
    ("El mapa: ", "el mal desenlace se concentra en lesiones grandes de la cerebral media izquierda (ínsula, opérculo frontal)"),
    ("Puntaje de ubicación: ", "AUC 0,41 en CV dentro de train; permutación p = 0,92"),
    ("Regiones + edad + NIHSS: ", "0,71–0,78 frente a 0,789 sin ellas"),
]):
    vineta(sl, 648, [182, 280, 346][k], 248, [86, 54, 54][k], b, r)
notas(sl, "[40 s] Si la red no usa la imagen, probamos descriptores interpretables. Llevamos todas las lesiones al espacio "
      "MNI con un registro afín, con buena calidad, Dice mediano 0,95, y promediamos por grupo. El mapa es plausible: los "
      "pacientes con mal desenlace tienen lesiones más grandes en la cerebral media izquierda, en ínsula y opérculo. Pero "
      "como predictor no sirve: un puntaje de ubicación armado dentro de cada fold da AUC 0,41, igual al azar según la "
      "prueba de permutación, y agregar regiones al modelo clínico lo empeora. El NIHSS ya mide el efecto de esa ubicación.")

# 3d. Desconexión de tractos
sl = nuevas[3]
encabezado(sl, "6 · ¿LA IMAGEN APORTA? DESCONEXIÓN DE TRACTOS", "Desconexión de tractos: JHU y Tractotron")
caja(sl, 64, 142, 470, 22, [P("% de pacientes con el tracto desconectado (Tractotron, p > 0,5)", 14, True, MORADO, TIT)])
tractos = [("Cuerpo calloso", 70, 85), ("Corticoespinal (algún lado)", 71, 85), ("Comisural frontal", 41, 61),
           ("Puente izquierdo", 47, 55), ("Fronto-estriatal izq.", 41, 48), ("Arcuato largo izq.", 26, 30)]
cd = CategoryChartData()
cd.categories = [t[0] for t in tractos][::-1]
cd.add_series("Buen desenlace (n = 122)", [t[1] / 100 for t in tractos][::-1])
cd.add_series("Mal desenlace (n = 33)", [t[2] / 100 for t in tractos][::-1])
ch = sl.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Pt(64), Pt(166), Pt(470), Pt(300), cd).chart
estilo_grafico(ch)
ch.has_legend = True
ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.BOTTOM, False
ch.legend.font.size = Pt(11)
pl = ch.plots[0]
pl.gap_width, pl.overlap = 50, -10
pl.has_data_labels = True
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = "0%", False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(10)
for s, c in zip(pl.series, [LILA, NAVY]):
    s.format.fill.solid()
    s.format.fill.fore_color.rgb = rgb(c)
va = ch.value_axis
va.minimum_scale, va.maximum_scale, va.major_unit = 0, 1, 0.25
va.major_gridlines.format.line.color.rgb = rgb("E4DEF2")
va.tick_labels.number_format, va.tick_labels.number_format_is_linked = "0%", False
va.format.line.fill.background()
caja(sl, 566, 146, 330, 20, [P("JHU · registro propio · 11 tractos", 13, True, MORADO, TIT)])
vineta(sl, 568, 172, 328, 54, "Carga corticoespinal ", "(fijada de antemano): 0,823 → 0,802; p = 0,14")
vineta(sl, 568, 232, 328, 38, "11 tractos: ", "ninguno pasa Bonferroni")
caja(sl, 566, 290, 330, 20, [P("Tractotron · 68 tractos del atlas BCB", 13, True, MORADO, TIT)])
vineta(sl, 568, 316, 328, 38, "Ninguno pasa FDR ", "(58 evaluables)")
vineta(sl, 568, 360, 328, 54, "Corticoespinal desconectado en el 74 %: ", "no discrimina, pero sí se asocia al NIHSS (6 vs. 3)")
vineta(sl, 568, 420, 328, 46, "El mal desenlace desconecta más ", "en todas partes: es tamaño, no un tracto", MAGENTA)
notas(sl, "[45 s] La hipótesis más fuerte era la desconexión: no importa solo dónde está la lesión, sino qué fibras corta. "
      "Lo probamos dos veces. Con nuestro registro y el atlas JHU, la carga sobre el corticoespinal, que fijamos de antemano, "
      "no mejora al modelo clínico. Y con Tractotron, del BCB Toolkit, sobre 68 tractos: ninguno pasa la corrección por "
      "comparaciones múltiples. El gráfico muestra por qué: los pacientes con mal desenlace tienen más tractos "
      "desconectados en todas partes, porque tienen lesiones más grandes, no un tracto específico. Y el corticoespinal "
      "aparece desconectado en tres de cada cuatro pacientes, así que no separa grupos, aunque sí se asocia con el NIHSS.")

# 3e. Síntesis
sl = nuevas[4]
encabezado(sl, "6 · ¿LA IMAGEN APORTA? SÍNTESIS", "Nada de la imagen mejora a edad + NIHSS")
caja(sl, 64, 142, 832, 22, [P("Cambio de AUC al agregar cada variable a edad + NIHSS (CV repetida 5×50)", 14, True, MORADO, TIT)])
sint = [("Volumen de lesión", -0.009), ("Región profunda (solo train)", -0.023), ("6 regiones (solo train)", -0.080),
        ("Carga corticoespinal · JHU", -0.020), ("Perfil de 11 tractos · JHU", -0.027),
        ("Nº de tractos desconectados · Tractotron", -0.009), ("Perfil de 68 tractos · Tractotron", -0.150),
        ("Trombectomía (tratamiento, no imagen)", 0.009)]
cd = CategoryChartData()
cd.categories = [s[0] for s in sint][::-1]
cd.add_series("ΔAUC", [s[1] for s in sint][::-1])
ch = sl.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Pt(64), Pt(166), Pt(620), Pt(300), cd).chart
estilo_grafico(ch)
ch.has_legend = False
pl = ch.plots[0]
pl.gap_width = 50
pl.has_data_labels = True
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = "+0.000;−0.000", False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size, pl.data_labels.font.bold = Pt(11), True
for i, (_, v) in enumerate(sint[::-1]):
    pt = pl.series[0].points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = rgb(NAVY if v > 0 else VIOLETA)
va = ch.value_axis
va.minimum_scale, va.maximum_scale, va.major_unit = -0.2, 0.05, 0.05
va.major_gridlines.format.line.color.rgb = rgb("E4DEF2")
va.tick_labels.number_format, va.tick_labels.number_format_is_linked = "+0.00;−0.00;0", False
va.format.line.fill.background()
ca = ch.category_axis
from pptx.enum.chart import XL_TICK_LABEL_POSITION
ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
tarjeta(sl, 712, 170, 184, 90, "0,823", "Edad + NIHSS (155 pacientes, 33 eventos): la referencia")
tarjeta(sl, 712, 274, 184, 90, "+0,008", "Mejor fusión profunda en ext; IC 95 % −0,10 a +0,13", MAGENTA)
caja(sl, 712, 378, 184, 90, [P("Cuatro formas de medir la lesión, la misma respuesta: el NIHSS ya mide el efecto de la lesión.", 12, True, TEXTO)])
notas(sl, "[40 s] Esta es la síntesis. Cada barra es cuánto cambia el AUC del modelo de edad más NIHSS, 0,823, al agregarle "
      "una variable de imagen: volumen, regiones, carga corticoespinal, perfiles de tractos. Todas quedan en cero o por "
      "debajo; las que más empeoran son las de muchas variables, que sobreajustan. La única barra positiva, la trombectomía, "
      "es tratamiento, no imagen, y no es significativa. La fusión profunda tampoco: +0,008 con un intervalo que va de "
      "−0,10 a +0,13. Cuatro maneras distintas de medir la lesión dan la misma respuesta: para el mRS al egreso, el NIHSS "
      "ya contiene lo que la resonancia sabe.")



# ---------- 3b. Diapositivas de la estructura de la Entrega 3 ----------
from pptx.enum.dml import MSO_LINE


def hueco(sl, x, y, w, h, que, detalle=""):
    """Recuadro punteado donde se pega a mano una imagen (pipeline o captura)."""
    r = rect(sl, x, y, w, h, "FBFAFE")
    r.name = "PEGAR AQUÍ: " + que
    r.line.fill.solid()
    r.line.fill.fore_color.rgb = rgb(LILA)
    r.line.width = Pt(1.5)
    r.line.dash_style = MSO_LINE.DASH
    caja(sl, x + 20, y + h / 2 - 30, w - 40, 60, [P("Pegar aquí: " + que, 14, True, VIOLETA, TIT, PP_ALIGN.CENTER),
                                                  P(detalle, 11, False, GRIS, CUERPO, PP_ALIGN.CENTER)],
         anchor=MSO_ANCHOR.MIDDLE, nombre="Texto guía (borrar al pegar la imagen)")


def imagen_en_caja(sl, x, y, w, h, fuente, nombre):
    """Inserta una imagen (ruta o bytes) ajustada y centrada dentro de la caja x, y, w, h (en puntos)."""
    import io
    datos = io.BytesIO(fuente) if isinstance(fuente, bytes) else open(fuente, "rb")
    ancho, alto = Image.open(datos).size
    datos.seek(0)
    k = min(w / ancho, h / alto)
    pw, ph = ancho * k, alto * k
    pic = sl.shapes.add_picture(datos, Pt(x + (w - pw) / 2), Pt(y + (h - ph) / 2), Pt(pw), Pt(ph))
    pic.name = nombre
    pic.line.color.rgb = rgb("E4DEF2")
    pic.line.width = Pt(0.75)
    return pic


def flecha(sl, x1, y1, x2, y2, col=GRIS):
    from pptx.enum.shapes import MSO_CONNECTOR
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Pt(x1), Pt(y1), Pt(x2), Pt(y2))
    c.line.color.rgb = rgb(col)
    c.line.width = Pt(1.25)
    ln = c.line._get_or_add_ln()
    punta = ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle"})
    ln.append(punta)
    return c


def nodo(sl, x, y, w, h, texto, fondo=TARJ, color_texto=TEXTO, negrita=False):
    r = rect(sl, x, y, w, h, fondo, MSO_SHAPE.ROUNDED_RECTANGLE)
    r.adjustments[0] = 0.15
    if fondo == TARJ:
        r.line.fill.solid()
        r.line.fill.fore_color.rgb = rgb(LILA)
        r.line.width = Pt(1)
    caja(sl, x + 6, y, w - 12, h, [P(texto, 11, negrita, color_texto, CUERPO, PP_ALIGN.CENTER)], anchor=MSO_ANCHOR.MIDDLE)
    return r


def diagrama_e3(sl, x, y, w, h):
    """Pipeline de análisis de la Entrega 3, con formas nativas (editable en PowerPoint)."""
    cw, gap, bh = (w - 30) / 2, 30, 34
    xa, xb = x, x + cw + gap
    caja(sl, xa, y, cw, 16, [P("Datos clínicos", 11, True, MORADO, TIT, PP_ALIGN.CENTER)])
    caja(sl, xb, y, cw, 16, [P("Imagen", 11, True, MORADO, TIT, PP_ALIGN.CENTER)])
    filas = [("Notas de ingreso (160)", "RM DWI + máscara de lesión"),
             ("Auditoría contra el CSV", "Registro afín a MNI152"),
             ("Tabla limpia · 155 pacientes", "Atlas: mapa · JHU · BCB (68)")]
    for k, (ta, tb_) in enumerate(filas):
        yy = y + 24 + k * 58
        nodo(sl, xa, yy, cw, bh, ta, NAVY if k == 2 else TARJ, "FFFFFF" if k == 2 else TEXTO, k == 2)
        nodo(sl, xb, yy, cw, bh, tb_, NAVY if k == 2 else TARJ, "FFFFFF" if k == 2 else TEXTO, k == 2)
        if k:
            flecha(sl, xa + cw / 2, yy - 24, xa + cw / 2, yy)
            flecha(sl, xb + cw / 2, yy - 24, xb + cw / 2, yy)
    ym = y + 24 + 3 * 58 + 10
    xm, wm = x + 50, w - 100
    flecha(sl, xa + cw / 2, ym - 34, xm + wm * 0.3, ym)
    flecha(sl, xb + cw / 2, ym - 34, xm + wm * 0.7, ym)
    nodo(sl, xm, ym, wm, bh, "Edad + NIHSS + 1 variable de la lesión")
    flecha(sl, x + w / 2, ym + bh, x + w / 2, ym + bh + 20)
    nodo(sl, xm, ym + bh + 20, wm, bh, "CV repetida 5×50 → ΔAUC frente a edad + NIHSS", MAGENTA, "FFFFFF", True)
    caja(sl, x, ym + 2 * bh + 30, w, 20, [P("Aparte: controles de la fusión en Kaggle (imagen barajada, peso de la compuerta)",
                                           10, False, GRIS, CUERPO, PP_ALIGN.CENTER)])


# Dataset
sl = N["dataset"]
encabezado(sl, "3 · DATASET", "Dataset: descripción y descarga")
fuentes = [
    ("Cohorte FOSCAL · privada",
     "160 pacientes con ACV isquémico: RM DWI + ADC (NIfTI), máscara manual de lesión (DWI) y de núcleo (ADC), "
     "nota de ingreso, variables clínicas y mRS al egreso, 30 y 90 días.",
     "No es pública. Se solicita al grupo BIVL²ab bajo el acuerdo de ética y uso de datos; no se publican imágenes ni IDs."),
    ("ISLES 2022 · pública",
     "250 casos multicéntricos con DWI, ADC, FLAIR y máscara de lesión en DWI. No trae mRS ni clínica: "
     "solo sirve para entrenar el encoder.",
     "isles22.grand-challenge.org (licencia CC BY 4.0). Hernandez Petzsche et al., Scientific Data, 2022."),
    ("Atlas · públicos",
     "MNI152 y tractos JHU-ICBM para registrar las lesiones y medir la carga corticoespinal; 68 tractos del "
     "BCB Toolkit para Tractotron.",
     "MNI152 y JHU vienen con FSL (fsl.fmrib.ox.ac.uk); BCB Toolkit en toolkit.bcblab.com."),
]
for k, (tit, desc, acc) in enumerate(fuentes):
    x = 64 + k * 284
    rect(sl, x, 150, 264, 300, TARJ)
    linea(sl, x, 151.5, 264, 0, MAGENTA if k == 0 else INDIGO)
    caja(sl, x + 14, 164, 236, 24, [P(tit, 15, True, NAVY, TIT)])
    caja(sl, x + 14, 196, 236, 140, [P(desc, 13, False, TEXTO)])
    caja(sl, x + 14, 340, 236, 100, [([("Descarga: " if k else "Acceso: ", 12, True, MORADO, CUERPO), (acc, 12, False, GRIS, CUERPO)], PP_ALIGN.LEFT)])
caja(sl, 64, 462, 832, 20, [P("Preprocesamiento: RAS · 1,75 × 1,75 × 3 mm · 128 × 128 × 64 vóxeles · z-score por canal. "
                              "Registro a MNI: afín sobre la máscara de cerebro (Dice mediano 0,948).", 11, False, GRIS)])
notas(sl, "[40 s] Usamos tres fuentes. La principal es la cohorte de FOSCAL: 160 pacientes con resonancia DWI y ADC, máscaras "
      "manuales, la nota de ingreso, las variables clínicas y el mRS; es privada, se accede a través del grupo bajo el "
      "acuerdo de ética, y por eso no publicamos imágenes ni identificadores. La segunda es ISLES 2022, 250 casos públicos "
      "con máscara de lesión, que solo usamos para entrenar el encoder porque no trae desenlace. Y la tercera son atlas "
      "públicos: el MNI152 y los tractos JHU de FSL, y los 68 tractos del BCB Toolkit, para el análisis de desconexión.")

# Mejor modelo
sl = N["mejor"]
encabezado(sl, "4 · MEJOR MODELO", "Mejor modelo: fusión con compuerta (Gated)")
pipeline = next(sh for sh in orig[3].shapes if sh.shape_type == 13 and sh.width > Pt(300))  # diagrama de la diap. 4
imagen_en_caja(sl, 64, 146, 470, 320, pipeline.image.blob, "Pipeline del mejor modelo")
pasos_m = [
    ("Imagen: ", "SwinUNETR congelado (Etapa A) → 256 tokens + máscara → encoder de imagen → un vector"),
    ("Clínica: ", "5 variables, un token cada una → encoder tabular → un vector"),
    ("Dos cabezas: ", "cada rama da su propio logit, l_img y l_tab"),
    ("Compuerta: ", "g = softmax(W·[img; tab]) y logit = g_img·l_img + g_tab·l_tab"),
]
for k, (b, r) in enumerate(pasos_m):
    vineta(sl, 558, 150 + k * 58, 338, 50, b, r)
tarjeta(sl, 558, 386, 164, 80, "0,868", "AUC en ext con una secuencia; clínico 0,855")
tarjeta(sl, 732, 386, 164, 80, "9–15 %", "peso que la compuerta le da a la imagen", MAGENTA)
notas(sl, "[1 min] El mejor modelo fue la fusión con compuerta. La imagen pasa por el SwinUNETR congelado de la Etapa A, que "
      "la convierte en 256 tokens; con la máscara manual y un encoder de imagen se resume en un vector. Las cinco variables "
      "clínicas entran como un token cada una a un encoder tabular. A diferencia de concatenar, aquí cada rama tiene su "
      "propia cabeza y da su propio logit, y una red pequeña, la compuerta, decide cuánto confiar en cada una: el logit final "
      "es la suma ponderada. Fue la mejor en AUC, 0,868, y en especificidad. Pero la compuerta nos dice algo importante: le "
      "da a la imagen solo entre 9 y 15 % del peso. Es decir, su ventaja viene de cómo usa la tabla, como veremos.")

# Resumen de la Entrega 2
sl = N["resumen_e2"]
encabezado(sl, "6 · RESULTADOS · ENTREGA 2 (PUNTO DE PARTIDA)", "Entrega 2 en una diapositiva")
for k, (c, e, col) in enumerate([
    ("0,745", "Dice del encoder en DWI: sí localiza la lesión", INDIGO),
    ("0,517", "Dice en el núcleo ADC: la entrada más débil", INDIGO),
    ("0,868 vs 0,855", "mejor fusión frente al modelo clínico; el IC de la diferencia incluye 0", MAGENTA),
    ("19 → 12", "falsos positivos de Gated frente al clínico, con los mismos 10 aciertos", INDIGO),
]):
    tarjeta(sl, 64 + k * 212, 156, 196, 116, c, e, col)
caja(sl, 64, 300, 832, 30, [P("La pregunta con la que cerró la Entrega 2", 14, True, MORADO, TIT)])
caja(sl, 64, 330, 832, 60, [P("¿La imagen aporta de verdad, o el AUC de las fusiones viene de las variables clínicas?", 20, False, NAVY, CUERPO)])
caja(sl, 64, 410, 832, 40, [P("El detalle (curva de Dice, tabla de fusiones, matriz de confusión) está en el anexo, al final.", 12, False, GRIS)])
notas(sl, "[30 s] Resumo la Entrega 2 como punto de partida. El encoder sí aprendió a ver la lesión en DWI, Dice 0,745, "
      "aunque el ADC quedó en 0,517. Las fusiones de imagen y tabla empataron con el modelo clínico: la mejor dio 0,868 frente "
      "a 0,855, con un intervalo de la diferencia que incluye el cero. Y Gated bajó los falsos positivos de 19 a 12. La "
      "pregunta que quedó abierta, y que guía toda esta entrega, es si la imagen aporta de verdad o si todo viene de la tabla.")

# Cambios de la Entrega 3
sl = N["cambios_e3"]
encabezado(sl, "6 · RESULTADOS · ENTREGA 3", "Entrega 3: qué cambiamos")
cambios = [
    ("Auditoría de la tabla: ", "160 notas de ingreso contra el CSV; tabla limpia de 155 pacientes"),
    ("Controles de la fusión: ", "imagen barajada, imagen sola y peso de la compuerta"),
    ("Ubicación de la lesión: ", "mapa de frecuencias en MNI y puntaje por región"),
    ("Registro a MNI + atlas JHU: ", "carga corticoespinal, fijada de antemano"),
    ("Tractotron (68 tractos): ", "desconexión cruzada con territorio, tratamiento y mRS a 90 días"),
]
for k, (b, r) in enumerate(cambios):
    vineta(sl, 66, 150 + k * 56, 390, 48, b, r, MAGENTA if k == 0 else INDIGO)
caja(sl, 64, 434, 400, 40, [P("Todo contra la misma referencia: edad + NIHSS, validación cruzada repetida 5×50.", 12, True, TEXTO)])
diagrama_e3(sl, 486, 146, 410, 320)
notas(sl, "[45 s] Para responder esa pregunta cambiamos el enfoque: en vez de una red más grande, controles y medidas "
      "interpretables. Uno, auditamos la tabla clínica contra las notas de ingreso. Dos, corrimos controles sobre la fusión. "
      "Tres, medimos dónde está la lesión. Cuatro, registramos las lesiones al espacio MNI y medimos cuánto tocan el tracto "
      "corticoespinal. Y cinco, con Tractotron, la desconexión de 68 tractos. Todo comparado contra la misma referencia: el "
      "modelo de edad más NIHSS con validación cruzada repetida.")

# Notebook de la Entrega 3
sl = N["notebook_e3"]
encabezado(sl, "8 · NOTEBOOK", "entrega3_resultados.ipynb")
celdas = [("[1]", "Configuración y rutas"), ("[2]", "Tabla limpia: AUC antes y después"), ("[3]", "Controles de la fusión (resumen.csv)"),
          ("[4]", "Ubicación de la lesión"), ("[5]", "Desconexión JHU"), ("[6]", "Tractotron + dataset_report"),
          ("[7]", "Síntesis: ΔAUC frente a edad + NIHSS")]
for k, (n, t) in enumerate(celdas):
    y = 152 + k * 40
    if k in (2, 6):
        rect(sl, 64, y - 4, 330, 34, TARJ)
    linea(sl, 66, y - 2, 0, 30, MAGENTA if k in (2, 6) else INDIGO)
    caja(sl, 78, y + 2, 40, 22, [P(n, 12, False, GRIS, "Courier New")])
    caja(sl, 122, y + 2, 270, 22, [P(t, 13, k in (2, 6), TEXTO)])
caja(sl, 64, 440, 330, 30, [P("Corre de arriba abajo; las salidas solo muestran resultados agregados, nunca IDs.", 11, False, GRIS)])
captura = AQUI / "figuras" / "notebook_celda7.png"  # se genera con captura_notebook.py
if captura.exists():
    imagen_en_caja(sl, 420, 146, 476, 324, str(captura), "Captura del notebook: celda [7]")
    caja(sl, 420, 474, 476, 14, [P("Salida de la celda [7] de entrega3_resultados.ipynb", 10, False, GRIS, CUERPO, PP_ALIGN.CENTER)])
else:
    hueco(sl, 420, 146, 476, 324, "captura del notebook", "corre captura_notebook.py para generarla")
notas(sl, "[1 min] Este es el notebook de la Entrega 3. Corre de arriba abajo y reproduce todo lo que mostramos: la "
      "comparación de la tabla original contra la tabla limpia, los controles de la fusión a partir del resumen de Kaggle, "
      "la ubicación, la desconexión con JHU y con Tractotron, y al final la síntesis con el cambio de AUC de cada variable. "
      "Las salidas solo tienen resultados agregados, nunca identificadores de pacientes. [Mostrar el notebook en vivo.]")

# PowerPoint invierte (blanco) las barras negativas si el punto no dice invertIfNegative = 0
from lxml import etree
from pptx.oxml.ns import qn
for sl in N.values():
    for sh in sl.shapes:
        if sh.has_chart:
            for dpt in sh.chart._chartSpace.xpath(".//c:dPt"):
                if dpt.find(qn("c:invertIfNegative")) is None:
                    e = etree.SubElement(dpt, qn("c:invertIfNegative"))
                    e.set("val", "0")
                    dpt.find(qn("c:idx")).addnext(e)

# ---------- 4. Orden, secciones y numeración ----------
o = {i + 1: s for i, s in enumerate(orig)}
E3 = "6 · RESULTADOS · ENTREGA 3"
final = [
    (o[1], None), (o[2], "1 · INTRODUCCIÓN"), (o[3], "2 · OBJETIVO"),
    (o[7], "3 · DATASET"), (N["dataset"], "3 · DATASET"), (o[8], "3 · DATASET"), (o[9], "3 · DATASET"),
    (o[4], "4 · MEJOR MODELO · PIPELINE"), (o[5], "4 · MEJOR MODELO · ZOOM"), (o[6], "4 · MEJOR MODELO · ZOOM"),
    (N["mejor"], "4 · MEJOR MODELO"), (o[10], "5 · CONFIGURACIÓN EXPERIMENTAL"),
    (N["resumen_e2"], "6 · RESULTADOS · ENTREGA 2 (PUNTO DE PARTIDA)"), (N["cambios_e3"], E3),
    (N["controles"], E3 + " · CONTROLES"), (N["auditoria"], E3 + " · AUDITORÍA DE DATOS"),
    (N["ubicacion"], E3 + " · UBICACIÓN"), (N["tractos"], E3 + " · DESCONEXIÓN"), (N["sintesis"], E3 + " · SÍNTESIS"),
    (o[14], "7 · ANÁLISIS DE RESULTADOS Y CONCLUSIONES"),
    (N["notebook_e3"], "8 · NOTEBOOK"), (o[15], "8 · NOTEBOOK · CÓDIGO DEL MODELO"), (o[16], None),
    (o[11], "ANEXO · ENTREGA 2"), (o[12], "ANEXO · ENTREGA 2"), (o[13], "ANEXO · ENTREGA 2"),
]
lst = prs.slides._sldIdLst
el_de = {id(s): el for s, el in zip(prs.slides, list(lst))}
assert len(final) == len(el_de), (len(final), len(el_de))
for el in list(lst):
    lst.remove(el)
for i, (sl, etiqueta) in enumerate(final, start=1):
    lst.append(el_de[id(sl)])
    for sh in sl.shapes:
        if sh.has_text_frame and sh.name == "Text 3" and sh.text_frame.text.strip().isdigit():
            set_texto(sh, str(i))
    if etiqueta:
        set_texto(por_id(sl, 7), etiqueta)
prs.save(SALIDA)
print("Escrito:", SALIDA, "|", len(prs.slides), "diapositivas")
