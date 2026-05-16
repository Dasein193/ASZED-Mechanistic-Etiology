---

### 2. El Filtro de Basura y Privacidad (CRÍTICO)
* **Nombre exacto:** `.gitignore`
* **Extensión:** `.(ninguna)` (El archivo empieza con un punto, no tiene nombre, solo extensión).
* **Para qué sirve:** Le dice a GitHub qué archivos **NO** debe subir. Esto es vital para no subir los archivos `.edf` gigantes, tus datos personales, o el archivo `Censo_ASZED_Limpio.csv` (por protección de datos médicos).

**Copia y pega este contenido:**
```text
# Datos y resultados locales (¡NO SUBIR DATOS DE PACIENTES!)
*.edf
*.csv
*.xlsx
Censo_ASZED_Limpio.csv

# Gráficos generados
*.png
*.jpg
*.pdf

# Python cache y entornos virtuales
__pycache__/
*.py[cod]
*$py.class
venv/
env/
.env

# Archivos del sistema operativo
.DS_Store
Thumbs.db