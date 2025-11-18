# 🔍 Cómo Buscar y Agregar Datasets de Roboflow

Esta guía te enseña cómo encontrar datasets públicos de baloncesto en Roboflow Universe y agregarlos a tu proyecto.

## 📊 Dataset Verificado Incluido

Actualmente el script incluye **1 dataset verificado**:

- **Basketball Detection (Roboflow 100)** - Dataset oficial de alta calidad

## 🔍 Buscar Datasets en Roboflow Universe

### 1. Ir a Roboflow Universe

Visita: [https://universe.roboflow.com/](https://universe.roboflow.com/)

### 2. Buscar Datasets de Baloncesto

Usa estas búsquedas:
- [Basketball](https://universe.roboflow.com/search?q=basketball)
- [Basketball Ball](https://universe.roboflow.com/search?q=basketball%20ball)
- [Ball Detection](https://universe.roboflow.com/search?q=ball%20detection)
- [Sports Ball](https://universe.roboflow.com/search?q=sports%20ball)

### 3. Filtrar por Tipo

- **Object Detection** - Para detectar balón y jugadores
- **Licencia pública** - Datasets gratuitos y abiertos
- **Número de imágenes** - Más de 100 imágenes recomendado

## 📝 Identificar Información del Dataset

Cuando encuentres un dataset, necesitas estos datos:

### Ejemplo: Basketball Detection

URL: `https://universe.roboflow.com/roboflow-100/basketball-detection`

Extraer:
- **Workspace:** `roboflow-100` (primera parte de la URL)
- **Project:** `basketball-detection` (segunda parte de la URL)
- **Version:** `1` (o la que quieras, se muestra en la página)

## ➕ Agregar Dataset al Script

### Método 1: Editar el Script

Abre `scripts/download_roboflow_dataset.py` y edita la lista `RECOMMENDED_DATASETS`:

```python
RECOMMENDED_DATASETS = [
    {
        'name': 'Basketball Detection (Roboflow 100)',
        'workspace': 'roboflow-100',
        'project': 'basketball-detection',
        'version': 1,
        'description': 'Dataset oficial de Roboflow 100 - Alta calidad, bien anotado'
    },
    # Agregar tu nuevo dataset aquí
    {
        'name': 'Mi Nuevo Dataset',
        'workspace': 'nombre-workspace',
        'project': 'nombre-proyecto',
        'version': 1,
        'description': 'Descripción del dataset'
    },
]
```

### Método 2: Descargar Directamente

Sin editar el script, usa:

```bash
python scripts/download_roboflow_dataset.py \
    --api-key TU_API_KEY \
    --workspace nombre-workspace \
    --project nombre-proyecto \
    --version 1
```

## 🔍 Datasets Públicos Populares de Baloncesto

Aquí hay algunos datasets públicos que puedes explorar (verificar disponibilidad):

### 1. Basketball Detection (Roboflow 100) ✅ Verificado
```
Workspace: roboflow-100
Project: basketball-detection
URL: https://universe.roboflow.com/roboflow-100/basketball-detection
Imágenes: ~600
Descripción: Dataset oficial, alta calidad
```

### 2. Basketball Court Detection
```
Workspace: basketball-court
Project: basketball-court-detection
URL: https://universe.roboflow.com/basketball-court/basketball-court-detection
Descripción: Detección de canchas y líneas
Nota: Verificar si es público
```

### 3. Otros Datasets

Busca en Universe con términos como:
- "basketball game"
- "basketball court"
- "sports detection"
- "ball tracking"

## ✅ Verificar si un Dataset es Accesible

### Prueba Rápida

```bash
python scripts/download_roboflow_dataset.py \
    --api-key TU_API_KEY \
    --workspace nombre-workspace \
    --project nombre-proyecto \
    --version 1 \
    --output-dir /tmp/test_download
```

Si funciona:
- ✅ El dataset es público y accesible
- Puedes agregarlo a `RECOMMENDED_DATASETS`

Si falla con error 404:
- ❌ El dataset no es público o requiere permisos
- Busca otro dataset

## 💡 Consejos para Encontrar Buenos Datasets

### Características de un Buen Dataset:

1. **Cantidad suficiente:**
   - Mínimo: 200+ imágenes
   - Ideal: 500+ imágenes
   - Excelente: 1000+ imágenes

2. **Bien anotado:**
   - Todas las imágenes tienen anotaciones
   - Bounding boxes precisas
   - Clase consistente ("basketball", "ball", etc.)

3. **Diversidad:**
   - Múltiples ángulos de cámara
   - Diferentes iluminaciones
   - Interior y exterior
   - Diferentes distancias

4. **Calidad:**
   - Imágenes nítidas
   - Buena resolución (640x640 mínimo)
   - Balón visible en las anotaciones

### Evaluar un Dataset en Roboflow:

1. **Ver el Preview:**
   - Haz click en "View Dataset"
   - Revisa algunas imágenes de ejemplo
   - Verifica la calidad de las anotaciones

2. **Revisar las Estadísticas:**
   - Número de imágenes
   - Distribución train/valid/test
   - Clases detectadas

3. **Leer la Descripción:**
   - Fuente del dataset
   - Propósito original
   - Condiciones de uso

## 🔄 Combinar Múltiples Datasets

El script `train_basketball_detector_simple.py` combina automáticamente todos los datasets en `data/basketball_training/`.

**Estrategia recomendada:**

1. **Descarga 2-3 datasets diferentes:**
   ```bash
   python scripts/download_roboflow_dataset.py --api-key KEY --download-all
   python scripts/download_roboflow_dataset.py --api-key KEY --workspace workspace1 --project proyecto1
   python scripts/download_roboflow_dataset.py --api-key KEY --workspace workspace2 --project proyecto2
   ```

2. **El script de entrenamiento los combina:**
   ```bash
   python scripts/train_basketball_detector_simple.py
   ```

3. **Resultado:**
   - Mayor diversidad de datos
   - Mejor generalización del modelo
   - Más robustez en diferentes condiciones

## 📚 Crear tu Propio Dataset Público

Si tienes tus propias anotaciones y quieres compartir:

1. **Sube a Roboflow:**
   - Crea proyecto en roboflow.com
   - Sube tus imágenes anotadas
   - Genera dataset

2. **Hacer Público:**
   - Project Settings → Make Public
   - Elegir licencia (MIT, CC0, etc.)

3. **Publicar en Universe:**
   - El dataset aparecerá en Roboflow Universe
   - Otros podrán usarlo

## 🔧 Solución de Problemas

### Error: "Dataset not found" (404)

**Causas:**
- Dataset no es público
- Workspace/project mal escrito
- Dataset fue removido

**Solución:**
1. Verifica la URL en el navegador
2. Confirma que el dataset es público
3. Revisa mayúsculas/minúsculas en nombres

### Error: "Missing permissions"

**Causa:** Dataset requiere autorización

**Solución:**
- Busca datasets con etiqueta "Public"
- O solicita acceso al dueño del dataset

### Descarga muy lenta

**Causa:** Dataset muy grande o conexión lenta

**Solución:**
- Descarga datasets más pequeños primero
- Usa `--max-frames` en extracción de frames
- Considera descargar de noche

## 📖 Recursos Adicionales

- [Roboflow Universe](https://universe.roboflow.com/)
- [Roboflow Docs](https://docs.roboflow.com/)
- [Buscar Datasets](https://universe.roboflow.com/search)
- [Public Datasets](https://universe.roboflow.com/browse/object-detection)

## 💬 Compartir tus Hallazgos

Si encuentras buenos datasets públicos de baloncesto:

1. Prueba que funcionan
2. Agrega un issue en GitHub con la información:
   - Workspace
   - Project
   - Version
   - Breve descripción
   - Número de imágenes

3. O crea un Pull Request agregándolo al script

---

**Última actualización:** Noviembre 2024

**¿Encontraste un buen dataset?** ¡Compártelo con la comunidad! 🏀
