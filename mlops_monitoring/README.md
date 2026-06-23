# MLOps Monitoring Pipeline

Estrategia de monitoreo E2E para el modelo de scoring, construida sobre los scripts del curso usando **Prefect** y **paralelismo CPU (mínimo 2 workers)**.

---

## Arquitectura del pipeline

```
Prefect Flow: monitoring_pipeline(payload)
│
├── [PARALELO — .submit() en Prefect]
│   ├── task: compute_raw_drift           → PSI sobre datos crudos (raw)
│   ├── task: compute_preprocessed_drift  → PSI sobre datos preprocesados
│   ├── task: compute_score_drift         → PSI sobre scores puros
│   └── task: compute_postprocessed_drift → PSI sobre scores posprocesados
│
├── task: compute_shap_task          → SHAP importance (ProcessPoolExecutor ≥2 workers)
├── task: update_shap_history        → Historial mes a mes en JSON
├── task: save_monitoring_results    → Persistencia CSV/JSON
└── task: generate_dashboard_task    → Tablero HTML interactivo E2E
```

### Paralelismo CPU

- **Nivel 1 — Prefect**: las 4 tasks de drift se lanzan con `.submit()` para ejecutarse de forma concurrente en el `ConcurrentTaskRunner` de Prefect.
- **Nivel 2 — ProcessPoolExecutor**: dentro de `monitoring_utils.py`, el cálculo de PSI por columna y los chunks de SHAP usan `concurrent.futures.ProcessPoolExecutor` con `n_workers = max(2, cpu_count() - 1)`, garantizando mínimo 2 workers.

---

## Indicadores de monitoreo

| Indicador | Etapa | Descripción |
|---|---|---|
| PSI (Population Stability Index) | RAW | Drift sobre datos crudos de entrada |
| PSI | Preprocesado | Drift sobre variables después de `preprocessing.py` |
| PSI | Score puro | Drift sobre probabilidades de `inference.py` |
| PSI | Score posprocesado | Drift sobre `puntuacion_tlv` y `grupo_ejec_tlv` de `posprocessing.py` |
| SHAP importance | Actual | Importancia media \|SHAP\| del período actual |
| SHAP variation | Mes a mes | Variación % de importancia respecto al período anterior |
| SHAP history | Histórico | Evolución de top-10 features a través de todas las particiones |

**Umbrales PSI:**
- `PSI < 0.10` → 🟢 OK (sin drift relevante)
- `0.10 ≤ PSI < 0.20` → 🟡 WARN (drift moderado, monitorear)
- `PSI ≥ 0.20` → 🔴 ALERT (drift alto, revisar pipeline)

---

## Estructura de archivos

```
mlops_monitoring/
├── scripts/
│   └── monitoring_utils.py      ← ÚNICO script modificado (según reglas de la tarea)
├── notebooks/
│   └── 10_Monitoring_Pipeline_Prefect.ipynb  ← Notebook principal para Colab
└── README.md
```

Los demás scripts del curso (`preprocessing.py`, `inference.py`, `posprocessing.py`, `training.py`, `hpo_training.py`) **no fueron modificados**.

---

## Instrucciones para ejecutar en Google Colab

### 1. Subir los scripts a Google Drive

Copia **todos** los scripts del curso a una carpeta en tu Drive, por ejemplo:
```
/content/drive/MyDrive/MLOps 202601/scripts/
```

Los scripts necesarios son:
- `preprocessing.py`
- `inference.py`
- `posprocessing.py`
- `training.py`
- `monitoring_utils.py`  ← usar la versión de este repositorio

### 2. Abrir el notebook en Colab

Abre `notebooks/10_Monitoring_Pipeline_Prefect.ipynb` en Google Colab.

### 3. Configurar el payload

En la **celda 5** del notebook, ajusta las rutas según tu Google Drive:

```python
BASE = '/content/drive/MyDrive/MLOps 202601/Dataset'

payload = {
    # Datos actuales (período a monitorear)
    'DIR_RAWDATA'       : f'{BASE}/DataSet Parcial',
    'DIR_PROCESSED'     : f'{BASE}/data_preprocesada',
    'SCORE_DIR'         : f'{BASE}/models/Scores/scr_model_puro',
    'DIR_OUTPUT'        : f'{BASE}/models/Scores/scr_model_prospro',
    'MODEL_DIR'         : f'{BASE}/models',

    # Datos de referencia (período anterior o de entrenamiento)
    'DIR_REF_RAWDATA'   : f'{BASE}/DataSet Referencia',
    'DIR_REF_PROCESSED' : f'{BASE}/data_preprocesada_ref',
    'DIR_REF_SCORES'    : f'{BASE}/models/Scores/scr_model_puro_ref',
    'DIR_REF_OUTPUT'    : f'{BASE}/models/Scores/scr_model_prospro_ref',

    # Monitoreo
    'MONITORING_DIR'    : f'{BASE}/monitoring',
    'SHAP_HISTORY_PATH' : f'{BASE}/monitoring/shap_history.json',

    # Parámetros
    'params': {
        'model_name' : 'extrac',   # mismo model_name que en el pipeline base
        'partition'  : 10,         # partición/mes a monitorear (ej: 10 = p10)
        'dd_metric'  : 'PSI',      # 'PSI' (recomendado) o 'KL'
        'quantils'   : 10,         # buckets para el cálculo de PSI
        'n_workers'  : 2,          # mínimo 2 workers (paralelismo CPU)
        'shap_sample': 500         # filas para SHAP (aumentar para más precisión)
    }
}
```

### 4. Ejecutar el pipeline de monitoreo

**Solo monitoreo** (asumiendo que la inferencia ya corrió):
```python
results = monitoring_pipeline(payload)
```

**Pipeline E2E completo** (inferencia + monitoreo):
```python
payload['params']['mode_type'] = 'inference'
results_e2e = mlops_pipeline_with_monitoring(payload)
```

### 5. Ver el tablero

El dashboard HTML se genera en `MONITORING_DIR/dashboard_{partition}.html` y se muestra automáticamente en el notebook con un `IFrame`.

---

## Estructura esperada de directorios en Drive

```
Dataset/
├── DataSet Parcial/           ← datos crudos actuales (*.csv)
├── DataSet Referencia/        ← datos crudos de referencia (*.csv)
├── data_preprocesada/
│   ├── preprocessed/
│   │   └── vars_10_extrac.csv
│   └── postprocessed/
│       └── post_10_extrac.csv
├── data_preprocesada_ref/
│   ├── preprocessed/          ← CSV de referencia preprocesados
│   └── postprocessed/
├── models/
│   ├── 2026-04-25_21-58-30/  ← carpeta con timestamp del modelo
│   │   ├── xgb_model.pkl
│   │   └── xgb_metadata.json
│   └── Scores/
│       ├── scr_model_puro/
│       │   └── inference_extrac_10.csv
│       ├── scr_model_puro_ref/    ← scores de referencia
│       ├── scr_model_prospro/
│       │   └── scr_extrac_10.txt
│       └── scr_model_prospro_ref/ ← scores posprocesados de referencia
└── monitoring/                ← generado automáticamente
    ├── dashboard_10.html
    ├── shap_history.json
    ├── drift_raw_10.csv
    ├── drift_preprocessed_10.csv
    ├── drift_score_puro_10.csv
    ├── drift_score_posprocesado_10.csv
    ├── shap_importance_10.csv
    └── shap_variation_10.csv
```

---

## Tablero de control (Dashboard)

El dashboard HTML generado contiene las siguientes secciones:

1. **🚦 Resumen ejecutivo** — semáforos por etapa del pipeline (verde/amarillo/rojo)
2. **Tablas de drift** — top features por PSI en cada etapa con nivel de alerta
3. **📉 Score Distribution Drift** — gráficas de barras de PSI por etapa
4. **📊 SHAP Importance actual** — importancia media |SHAP| del período
5. **Variación SHAP mes a mes** — tabla con variación % vs mes anterior
6. **📈 Evolución histórica SHAP** — líneas de evolución de top-10 features

---

## Dependencias

```
prefect>=2.16.0,<3.0.0
shap>=0.44.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost==1.2.8
dask>=2024.1.0
matplotlib>=3.8.0
scikit-learn>=1.4.0
pandas
numpy
```

Se instalan automáticamente en las primeras celdas del notebook.

---

## Notas técnicas

- **`monitoring_utils.py`** es el único script modificado respecto a los originales del curso.
- El cálculo de PSI admite `dd_metric='KL'` como alternativa a `'PSI'`.
- Para la variación SHAP mes a mes se necesita ejecutar el monitoreo al menos en 2 particiones distintas; el historial se acumula automáticamente en `shap_history.json`.
- En Colab gratuito `cpu_count()` retorna 2; en Colab Pro puede ser mayor.
