# Proyecto: Consumo de Energía y Huella de Carbono

## Objetivo

Predecir la emisión de CO2 en plantas de acero mediante algoritmos de regresión supervisada y modelizar series de tiempo para entender patrones de consumo energético.

## Estado del Proyecto (Revisión: 2026-05-06)

### Checklist de Etapas

- [x] **1. Setup** — Estructura del proyecto, dataset original y de trabajo creados
- [x] **2. Calidad de datos** — Análisis de nulidad, duplicados y tipos de datos completado
- [x] **3. EDA** — Análisis estadístico, gráficos y análisis de ceros en target completado
- [x] **4. Preprocesamiento (Experimento A)** — OHE + StandardScaler aplicados (pipeline base)
- [ ] **4b. Preprocesamiento (Experimento B)** — Pendiente: features temporales extraídas de `date`
- [x] **5. Entrenamiento** — GridSearchCV (279 configuraciones), RandomForest seleccionado
- [x] **6. Análisis de resultados** — RMSE=0.089, R²=0.992 (documentado en notebook)
- [x] **7. Series de tiempo** — ARIMA aplicado para CO2 (MAE=0.0217) y Usage_kWh
- [ ] **8. Selección de features** — Feature importances calculadas (pendiente exportar configuración)
- [ ] **9. Preproducción** — Pendiente (crear dataset de validación y métricas)
- [ ] **10. Producción** — Pendiente (exportar artefactos .joblib y pipeline completa)
- [ ] **11. Aplicación** — Pendiente (desarrollar app.py)

### Etapa Actual

**Rediseño de preprocesamiento:** El EDA confirmó que los ceros en CO2 son estados operacionales estructurados, no errores. El Experimento A (pipeline base) está completo; el Experimento B incorpora las features temporales derivadas del análisis de ceros y está pendiente de implementar.

### Próximos Pasos

1. Implementar Experimento B en `04_Transformacion_de_Variable.ipynb`:
   - Extraer `hora` y `turno` desde columna `date`
   - Codificar `NSM` como features cíclicas: `sin(2π·NSM/86400)`, `cos(2π·NSM/86400)`
   - Reentrenar con el mismo pipeline de `05_Modelizacion_Supervisada.ipynb`
2. Comparar métricas Exp. A vs Exp. B (R², RMSE)
3. Exportar modelos y transformadores a `03_Modelos/01_Historial/`
4. Crear `04_Resultados/01_Analisis/` con `feature_importances.json` y `metricas_entrenamiento.json`
5. Crear dataset de validación en `01_Datos/02_Validacion/`
6. Desarrollar pipeline de producción y aplicación Python en `05_Aplicacion/`

---

## Hallazgos del EDA

### Variables clave

| Variable | Correlación con CO2 | Significancia |
|---|---|---|
| `Usage_kWh` | r = 0.988 | p < 1e-300 |
| `Lagging_Current_Reactive_Power_kVarh` | r = 0.887 | p < 1e-300 |

### Análisis de ceros en CO2_tCO2

El 59.9% de los registros tienen `CO2_tCO2 = 0`. El análisis muestra que corresponden a **estados operacionales reales** de la planta, no a errores de medición.

**Distribución de ceros por turno horario**

| Turno | Rango | % Registros con CO2 = 0 |
|---|---|---|
| Noche | 22h – 6h | 94.6% |
| Mañana | 6h – 14h | 49.5% |
| Tarde | 14h – 22h | 35.6% |

El turno nocturno corresponde a planta prácticamente detenida; el turno de tarde es el más productivo.

**Distribución de ceros por día de la semana**

| Grupo | % Registros con CO2 = 0 |
|---|---|
| Lunes – Viernes | ~51% |
| Sábado | 73% |
| Domingo | 89% |

El patrón de fin de semana refleja un modo de stand-by operacional.

**Distribución de ceros por Load_Type**

| Load_Type | % Registros con CO2 = 0 |
|---|---|
| `Light_Load` | 90.4% |
| `Medium_Load` | 41.0% |
| `Maximum_Load` | 9.2% |

`Light_Load` actúa como proxy de planta inactiva; `Maximum_Load` casi siempre genera emisiones.

**Validación estadística:** Test chi-cuadrado turno vs CO2=0 → chi²=9.266, p < 1e-300. El patrón es completamente estructurado y no aleatorio.

---

## Diseño de Experimentos

### Experimento A — Pipeline base (completado)

Preprocesamiento con las variables disponibles directamente en el dataset.

- **Categóricas:** OneHotEncoding sobre `WeekStatus`, `Day_of_week`, `Load_Type`
- **Numéricas:** StandardScaler sobre todas las numéricas incluyendo `NSM` como valor lineal
- **Features temporales:** ninguna extraída de `date`
- **Dataset resultante:** 35.040 × 20 columnas (`df_tablon.pickle`)
- **Resultado:** RandomForest R²=0.992, RMSE=0.089

### Experimento B — Con features temporales (pendiente)

Preprocesamiento que incorpora la señal operacional identificada en el análisis de ceros.

- **Nuevas features desde `date`:** `hora` (0–23), `turno` (Noche/Mañana/Tarde como dummy)
- **NSM cíclico:** reemplaza `NSM` lineal por `NSM_sin = sin(2π·NSM/86400)` y `NSM_cos = cos(2π·NSM/86400)`
- **Rationale:** el turno y la hora son los predictores más fuertes del estado CO2=0 (94.6% nocturno); representarlos explícitamente debe mejorar la calibración del modelo en registros de bajo consumo
- **Hipótesis:** mejora en RMSE para el rango CO2 ∈ (0, 0.02] sin degradar R² global

---

## Fuente de datos

Steel Industry Energy Consumption Dataset
UCI Machine Learning Repository — ID 851
DAEWOO Steel Co. Ltd, Gwangyang, Corea del Sur
Período: 1 enero 2018 – 31 diciembre 2018, granularidad 15 minutos
35.040 registros
https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption

---

## Notebooks Desarrollados

| Notebook | Descripción |
|---|---|
| `01_Set Up.ipynb` | Inicialización y creación de estructura |
| `02_Calidad_datos.ipynb` | Validación de datos |
| `03_EDA.ipynb` | Análisis exploratorio, correlaciones y análisis de ceros en CO2 |
| `04_Transformacion_de_Variable.ipynb` | Codificación y escalado (Experimento A) |
| `05_Modelizacion_Supervisada.ipynb` | Optimización de hiperparámetros |
| `06_Series_de_tiempo.ipynb` | ARIMA para CO2 y consumo energético |

## Resultados Principales

- **Modelo supervisado (Exp. A):** RandomForest con R² = 0.992, RMSE = 0.089
- **Variable más importante:** `Usage_kWh` (99.75% de importancia)
- **Predicción ARIMA CO2:** MAE = 0.0217
- **Correlación CO2–Usage_kWh:** r = 0.988
