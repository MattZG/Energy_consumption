# Proyecto: Consumo de Energía y Huella de Carbono

## Objetivo

Predecir la emisión de CO2 en plantas de acero mediante algoritmos de regresión supervisada y modelizar series de tiempo para entender patrones de consumo energético.

## Estado del Proyecto (Revisión: 2026-05-11)

### Checklist de Etapas

- [x] **1. Setup** — Estructura de carpetas, dataset original, trabajo y validación creados
- [x] **2. Calidad de datos** — Análisis de nulidad, duplicados y tipos de datos completado; pickles generados en `01_Datos/03_Trabajo/`
- [x] **3. EDA** — Análisis estadístico, gráficos exportados a `04_Resultados/01_Analisis/` y análisis de ceros en target completado con conclusiones documentadas
- [x] **4. Preprocesamiento (Experimento A)** — OHE + StandardScaler aplicados; `df_tablon.pickle` generado (pipeline base sin features temporales)
- [x] **4b. Preprocesamiento (Experimento B)** — Features temporales implementadas (`hora`, `turno`, `NSM_sin`/`NSM_cos`, `es_fin_de_semana`); `trabajo_preprocesado_B.pickle` generado
- [x] **5. Entrenamiento** — GridSearchCV sobre 279 configuraciones ejecutado; RandomForest seleccionado (R²=0.992, RMSE=0.089)
- [x] **6. Series de tiempo** — ARIMA aplicado para CO2 (MAE=0.0217) y Usage_kWh; ejecutado con conclusiones en notebook
- [x] **7. Análisis de resultados** — JSONs de métricas e importancias exportados a `04_Resultados/01_Analisis/` para ambos experimentos
- [x] **8. Selección de features** — Modelo minimal B seleccionado: 7 features (Load_Type, factores de potencia, NSM cíclico); R²=0.8715 en validación, degradación -0.34% vs modelo completo; `config_modelo_minimal.json` generado
- [x] **9. Exportar artefactos** — Encoders, scalers y modelos exportados como `.joblib` en `03_Modelos/01_Historial/`
- [x] **10. Preproducción** — Ambos experimentos evaluados sobre 10.512 registros no vistos; métricas guardadas en `04_Resultados/02_Preproduccion/`; ningún modelo muestra overfitting
- [x] **11. Producción** — Pipeline de producción (Exp B minimal, 3 inputs del operador) y pipeline de referencia (Exp A) exportados a `03_Modelos/02_Produccion/`; reporte en `04_Resultados/03_Produccion/`
- [ ] **12. Aplicación** — Pendiente: desarrollar `app.py` en `05_Aplicacion/`

### Etapa Actual

**Aplicación — pipeline de producción completado y listo para integrar.**

El modelo minimal (Exp B, 7 features, 3 inputs del operador) alcanza R²=0.8715 en validación con una degradación de solo -0.34% respecto al modelo completo de 20 features. Los artefactos de producción están en `03_Modelos/02_Produccion/`. El siguiente paso es la aplicación Streamlit.

### Próximos Pasos

1. **Aplicación**: desarrollar `app.py` en `05_Aplicacion/` con interfaz Streamlit — el operador ingresa Load_Type, Leading_Current_Power_Factor y Lagging_Current_Power_Factor; la app calcula NSM_sin/cos desde el reloj del sistema y retorna la predicción de CO₂ en tCO₂

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

### Justificación del diseño

El dataset contiene dos variables con correlación física directa con CO2: `Usage_kWh` (r=0.988) y `Lagging_Current_Reactive_Power_kVarh` (r=0.887). La planta genera CO2 en función de la energía eléctrica consumida, por lo que incluir estas variables resuelve el problema de predicción de forma casi trivial. El modelo aprende la relación física, no los patrones operacionales. El resultado es preciso pero no accionable: no responde cuándo ni por qué la planta consume más.

El diseño de dos experimentos responde a preguntas distintas:

- **Exp A**: ¿cuál es el techo de rendimiento con información completa? Si el modelo falla incluso aquí, hay un problema fundamental con los datos o el enfoque. Es el baseline de referencia.
- **Exp B**: ¿puede el modelo identificar cuándo y por qué la planta consume más, usando solo variables que el operador puede conocer antes de que ocurra el consumo? Esta es la pregunta de negocio real.

**Criterio de éxito**: si Exp B degrada el RMSE en ≤20% respecto a Exp A, las features temporales y operacionales son suficientes para predicción accionable. Si supera ese umbral, se necesitan datos adicionales del proceso productivo (temperatura de horno, toneladas producidas, tipo de colada).

---

### Experimento A — Pipeline base (completado)

Preprocesamiento con las variables disponibles directamente en el dataset.

- **Categóricas:** OneHotEncoding sobre `WeekStatus`, `Day_of_week`, `Load_Type`
- **Numéricas:** StandardScaler sobre todas las numéricas incluyendo `Usage_kWh` y `Lagging_Current_Reactive_Power_kVarh`
- **Dataset resultante:** 35.040 × 18 features (`df_tablon.pickle`)
- **Modelo:** RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_leaf=4, min_samples_split=2)
- **Métricas test (30%, random_state=42):** R²=0.9923, RMSE=0.0875, MAE=0.0062
- **Feature importance top:** `Usage_kWh_std_` = 99.65%

### Experimento B — Sin variables físicas directas (completado)

Preprocesamiento que excluye las variables con correlación física directa y fuerza al modelo a encontrar señales operacionales accionables.

- **Variables excluidas:** `Usage_kWh`, `Lagging_Current_Reactive_Power_kVarh`
- **Nuevas features desde `date`:** `hora` (0–23), `turno` (Mañana/Tarde/Noche como dummies), `es_fin_de_semana`
- **NSM cíclico:** `NSM_sin = sin(2π·NSM/86400)` y `NSM_cos = cos(2π·NSM/86400)`
- **Dataset resultante:** 35.040 × 20 features (`trabajo_preprocesado_B.pickle`)
- **Modelo:** RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=4, min_samples_split=2)
- **Métricas test:** R²=0.8470, RMSE=0.3910, MAE=0.1933
- **Feature importance top:** `Load_Type_Light_Load`=37.3%, `Leading_Current_Power_Factor_std_`=36.3%, `Lagging_Current_Power_Factor_std_`=13.0%

### Tabla comparativa A vs B

| Métrica | Exp A (con variables físicas) | Exp B (sin variables físicas) |
|---|---|---|
| R² | 0.9923 | 0.8470 |
| RMSE | 0.0875 | 0.3910 |
| MAE | 0.0062 | 0.1933 |
| Degradación RMSE | — | 347% |
| Variables | 18 | 20 |

### Modelo minimal de producción

Selección de features sobre el Exp B: se evaluaron 3 subconjuntos por ablación, buscando el mínimo que mantiene R² dentro del 20% del modelo completo (umbral: R² ≥ 0.6996).

| Subconjunto | Features | R² validación | Degradación |
|---|---|---|---|
| Top 3 variables | 5 | 0.8097 | -7.4% |
| Top 3 + NSM cíclico | 7 | 0.8715 | -0.34% |
| Top 5 variables | 8 | 0.8698 | -0.54% |

**Seleccionado**: Top 3 + NSM cíclico (7 features). NSM_sin y NSM_cos se derivan automáticamente del reloj del sistema — el operador ingresa solo 3 campos. La degradación de -0.34% vs el modelo completo de 20 features justifica la reducción.

**Inputs del operador en la app**:
- `Load_Type` — tipo de carga programada para el turno (Light / Medium / Maximum)
- `Leading_Current_Power_Factor` — factor de potencia adelantada, lectura del sensor (0–100)
- `Lagging_Current_Power_Factor` — factor de potencia retrasada, lectura del sensor (0–100)

### Conclusión del ciclo de experimentación

El criterio de éxito (≤20% de degradación) no se cumple: **FAIL con 347%**. `Usage_kWh` es un proxy casi perfecto del CO2 (correlación r=0.988, importancia 99.65%) y no puede ser reemplazado por las features temporales disponibles. El Experimento B sí captura estructura real (R²=0.847) — Load_Type y los factores de potencia son señales informativas — pero con precisión insuficiente para uso operacional.

La siguiente etapa es **preproducción**: evaluar el Exp A (modelo de referencia) sobre el dataset de validación para confirmar que el rendimiento se mantiene fuera de la muestra de entrenamiento. Si se requiere un modelo accionable, se necesitan datos adicionales del proceso productivo.

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

### Entrenamiento (test 30%, random_state=42)
| Experimento | R² | RMSE | MAE |
|---|---|---|---|
| A — con variables físicas | 0.9923 | 0.0875 | 0.0062 |
| B — sin variables físicas | 0.8470 | 0.3910 | 0.1933 |

### Preproducción (validación, n=10.512, datos no vistos)
| Experimento | R² | RMSE | MAE | Diagnóstico |
|---|---|---|---|---|
| A — con variables físicas | 0.9959 | 0.0644 | 0.0048 | Generaliza correctamente |
| B — sin variables físicas | 0.8745 | 0.3557 | 0.1732 | Generaliza correctamente |
| B minimal — 7 features (producción) | 0.8715 | 0.3598 | 0.1790 | Generaliza correctamente |

- **Variable más importante Exp A:** `Usage_kWh` (99.65% de importancia)
- **Variables más importantes Exp B:** `Load_Type_Light_Load` (37.3%), `Leading_Current_Power_Factor` (36.3%), `Lagging_Current_Power_Factor` (13.0%)
- **Predicción ARIMA CO2:** MAE = 0.0217
- **Correlación CO2–Usage_kWh:** r = 0.988
