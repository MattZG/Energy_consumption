# Proyecto: Consumo de Energía y Huella de Carbono

## Objetivo

Predecir la emisión de CO2 en plantas de acero mediante algoritmos de regresión supervisada y modelizar series de tiempo para entender patrones de consumo energético.

## Estado del Proyecto (Revisión: 2026-05-05)

### Checklist de Etapas

- [x] **1. Setup** — Estructura del proyecto, dataset original y de trabajo creados
- [x] **2. Calidad de datos** — Análisis de nulidad, duplicados y tipos de datos completado
- [x] **3. EDA** — Análisis estadístico y gráficos con conclusiones por variable
  - **Variables clave**: `Usage_kWh` (r=0.988 con CO2), `Lagging_Current_Reactive_Power_kVarh` (r=0.887)
  - **Hallazgo principal**: El 59.9% de los registros tienen CO2=0, correspondiendo a estados operacionales reales (turno nocturno, fines de semana, `Load_Type=Light_Load`)
  - **Decisión**: No aplicar modelo zero-inflation — incorporar features temporales (`hora`, `turno`) al preprocesamiento
  - **Gráficos**: `04_Resultados/01_Analisis/` (9 archivos)
- [x] **4. Preprocesamiento** — OneHotEncoding y StandardScaler aplicados
- [x] **5. Entrenamiento** — GridSearchCV (279 configuraciones), RandomForest seleccionado
- [x] **6. Análisis de resultados** — RMSE=0.089, R²=0.992 (documentado en notebook)
- [x] **7. Series de tiempo** — ARIMA aplicado para CO2 (MAE=0.0217) y Usage_kWh
- [ ] **8. Selección de features** — Feature importances calculadas (pendiente exportar configuración)
- [ ] **9. Preproducción** — Pendiente (crear dataset de validación y métricas)
- [ ] **10. Producción** — Pendiente (exportar artefactos .joblib y pipeline completa)
- [ ] **11. Aplicación** — Pendiente (desarrollar app.py)

### Etapa Actual

**En transición:** El ciclo de desarrollo está completo (análisis → modelado → series de tiempo). Pendiente la etapa de industrialización: exportar artefactos, estructurar resultados y crear la aplicación.

### Próximos Pasos

1. Exportar modelos y transformadores a `03_Modelos/01_Historial/`:
   - `modelo_random_forest.joblib`
   - `encoder_onehot.joblib`
   - `scaler_standard.joblib`
2. Crear `04_Resultados/01_Analisis/` con `feature_importances.json` y `metricas_entrenamiento.json`
3. Crear dataset de validación en `02_Datos/02_Validacion/`
4. Desarrollar pipeline de producción y aplicación Python en `05_Aplicacion/`

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
| `03_EDA.ipynb` | Análisis exploratorio con gráficos y conclusiones |
| `04_Transformacion_de_Variable.ipynb` | Codificación y escalado |
| `05_Modelizacion_Supervisada.ipynb` | Optimización de hiperparámetros |
| `06_Series_de_tiempo.ipynb` | ARIMA para CO2 y consumo energético |

## Resultados Principales

- **Modelo supervisado:** RandomForest con R² = 0.992
- **Variable más importante:** Usage_kWh (99.75% de importancia)
- **Predicción ARIMA CO2:** MAE = 0.0217
- **Correlación CO2–Usage_kWh:** 0.988
