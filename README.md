# ACV isqu?mico ? texto, tabular e imagen

Copia de la carpeta `text` para continuar el trabajo en otro PC. Incluye c?digo, tablas, cach?s, modelos y resultados disponibles en esta carpeta. Mantener este repositorio privado: contiene datos cl?nicos.

## Descargar en otro PC

Inicia sesi?n en GitHub con la cuenta que tiene acceso al repositorio y ejecuta:

```powershell
git clone https://github.com/Manu2831/acv-ischemic-text.git
cd acv-ischemic-text
```

Tambi?n puedes usar **Code > Download ZIP** en GitHub despu?s de iniciar sesi?n.

## Archivos que debes transferir aparte

- `license.env`: contiene la licencia/configuraci?n privada. C?pialo desde el PC original a la ra?z de esta carpeta mediante un medio privado. No se publica en GitHub.
- El respaldo original contiene ese archivo: se incluye una copia del ZIP con sufijo `_sin_credenciales` que lo omite.
- Esta copia abarca ?nicamente `text/`. No incluye archivos de su carpeta superior, como `acv_split/`, `dataset_report.csv` o `isles2022/`. Los experimentos que los usan requieren copiarlos aparte.
- No se incluyen entornos virtuales, bytecode de Python ni configuraci?n local de Claude.

## Organizaci?n

- `01_tabular/`: experimentos tabulares.
- `02_texto_bert/`: modelos de texto y multimodales con BERT.
- `03_texto_reglas_vs_bert/`: comparaci?n de reglas y BERT.
- `04_imagen/`: preprocesamiento y entrenamiento de imagen.
- `05_kaggle/`: generaci?n del paquete para Kaggle.
- `06_mapa_lesiones/`: mapas y descriptores de lesiones.
- `07_tabla_limpia/`: construcci?n y evaluaci?n de la tabla corregida.
- `08_desconexion/`: c?digo adicional de desconexi?n.
- `datos/` y carpetas de resultados: datos y artefactos existentes.

## Configuraci?n de rutas y Python

Instala las dependencias que requiera el experimento que vayas a ejecutar. Los scripts usan, entre otras, PyTorch, NumPy, pandas, scikit-learn, Transformers, TabPFN, matplotlib, seaborn, SciPy, nibabel, timm y python-dotenv; algunos flujos tambi?n usan MONAI o Weights & Biases. Esta copia no fija versiones ni garantiza reproducibilidad del entorno original.

`04_imagen/rutas.py` permite configurar `ACV_DATA_ROOT`, `ACV_TABULAR_DIR`, `ACV_TABLA_LIMPIA`, `ACV_WORK_DIR` y `ACV_ISLES_DIR`. Por defecto busca los datos originales en la carpeta superior a este repositorio. Revisa las rutas locales de cada script antes de ejecutarlo en el otro PC.

## Guardar cambios posteriores

```powershell
git add .
git commit -m "Actualizar proyecto"
git push
```

En el otro PC, ejecuta `git pull` para recibir los cambios despu?s de guardar tus propios cambios locales.
