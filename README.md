
# ACV isquémico: predicción del mRS al egreso con imagen y datos clínicos


**Objetivo:** predecir el mal desenlace al egreso (mRS > 2) en ACV isquémico combinando RM DWI/ADC con variables clínicas, y saber si la imagen aporta más allá de la clínica.

> Este repositorio debe ser **privado**: contiene datos clínicos. No subir imágenes, IDs de pacientes ni tablas por paciente (ver `.gitignore`).

## Entrega 3

**Notebook:** [`entrega3_resultados.ipynb`](entrega3_resultados.ipynb). Corre de arriba abajo y reproduce todos los resultados de la entrega; sus salidas solo muestran resultados agregados.

| Paso | Qué responde | Código |
|---|---|---|
| Auditoría de la tabla | ¿Los datos clínicos tienen errores? ¿Cuánto pesan? | `07_tabla_limpia/` |
| Controles de la fusión | ¿El modelo multimodal usa la imagen? (imagen barajada, peso de la compuerta) | `05_kaggle/`, `04_imagen/` |
| Ubicación de la lesión | ¿Importa dónde está la lesión? | `06_mapa_lesiones/` |
| Desconexión JHU | ¿La carga sobre el corticoespinal agrega algo a edad + NIHSS? | `08_desconexion/` |
| Tractotron | ¿Algún tracto de 68 (atlas BCB) separa buen y mal desenlace? | `09_tractotron/` |
| Presentación | Arma el deck de la entrega a partir del de la Entrega 2 | `presentacion/mejorar_deck.py` |

**Resultado principal:** ninguna forma de medir la lesión (fusión profunda, ubicación, carga corticoespinal, perfiles de tractos) mejora a edad + NIHSS (AUC 0,823, validación cruzada repetida 5×50, 155 pacientes, 33 eventos). Con la imagen barajada la fusión da el mismo AUC (0,858) y la compuerta le da a la imagen solo 9–15 % del peso. Corregir un NIHSS mal digitado (120) subió el AUC clínico de 0,779 a 0,827.

### Datos

- **Cohorte FOSCAL** (privada): 160 pacientes con RM DWI + ADC, máscaras manuales, nota de ingreso, variables clínicas y mRS. Se solicita al grupo BIVL²ab bajo el acuerdo de ética y uso de datos. Va en la carpeta superior al repositorio (`../acv_split/`, `../dataset_report.csv`).
- **ISLES 2022** (pública, CC BY 4.0): 250 casos con máscara de lesión DWI, solo para entrenar el encoder. Descarga en [isles22.grand-challenge.org](https://isles22.grand-challenge.org/). Hernandez Petzsche et al., *Scientific Data*, 2022.
- **Atlas** (públicos): MNI152 y tractos JHU-ICBM (incluidos en [FSL](https://fsl.fmrib.ox.ac.uk/)) en `../atlas/`; tractos del [BCB Toolkit](http://toolkit.bcblab.com/) para Tractotron. El log de Tractotron va en `../logTractotron.txt`.
- **Resultados de Kaggle**: `../../resultados_acv_p/resumen.csv` (o la variable `ACV_RESUMEN`).

## Descargar en otro PC

Inicia sesión en GitHub con la cuenta que tiene acceso al repositorio y ejecuta:

```powershell
git clone https://github.com/Manu2831/acv-ischemic-project.git
cd acv-ischemic-project
```


También puedes usar **Code > Download ZIP** en GitHub después de iniciar sesión.


## Archivos que debes transferir aparte

- `license.env`: contiene la licencia/configuración privada. Cópialo desde el PC original a la raíz de esta carpeta mediante un medio privado. No se publica en GitHub.
- El respaldo original contiene ese archivo: se incluye una copia del ZIP con sufijo `_sin_credenciales` que lo omite.
- Esta copia abarca únicamente `text/`. No incluye archivos de su carpeta superior, como `acv_split/`, `dataset_report.csv`, `logTractotron.txt`, `atlas/` o `isles2022/`. Los experimentos que los usan requieren copiarlos aparte.
- No se incluyen entornos virtuales, bytecode de Python ni configuración local de Claude.

## Organización

- `01_tabular/`: experimentos tabulares.
- `02_texto_bert/`: modelos de texto y multimodales con BERT.
- `03_texto_reglas_vs_bert/`: comparación de reglas y BERT.
- `04_imagen/`: preprocesamiento y entrenamiento de imagen (encoder SwinUNETR y clasificador multimodal).
- `05_kaggle/`: generación del paquete para Kaggle.
- `06_mapa_lesiones/`: mapas y descriptores de ubicación de la lesión.
- `07_tabla_limpia/`: auditoría, construcción y evaluación de la tabla corregida.
- `08_desconexion/`: registro a MNI y desconexión con el atlas JHU.
- `09_tractotron/`: tablas de Tractotron (desde su log) y cruce con datos clínicos, territorio y tratamiento.
- `presentacion/`: script que arma el deck de la Entrega 3.
- `datos/` y carpetas de resultados: datos y artefactos existentes.


Instala las dependencias que requiera el experimento que vayas a ejecutar. Los scripts usan, entre otras, PyTorch, NumPy, pandas, scikit-learn, Transformers, TabPFN, matplotlib, seaborn, SciPy, nibabel, SimpleITK, timm, python-pptx y python-dotenv; algunos flujos también usan MONAI o Weights & Biases. Esta copia no fija versiones ni garantiza reproducibilidad del entorno original.

`04_imagen/rutas.py` permite configurar `ACV_DATA_ROOT`, `ACV_TABULAR_DIR`, `ACV_TABLA_LIMPIA`, `ACV_WORK_DIR` y `ACV_ISLES_DIR`. Por defecto busca los datos originales en la carpeta superior a este repositorio. Revisa las rutas locales de cada script antes de ejecutarlo en el otro PC.

## Guardar cambios posteriores

```powershell
git add .
git commit -m "Actualizar proyecto"
git push
```

En el otro PC, ejecuta `git pull` para recibir los cambios después de guardar tus propios cambios locales.
