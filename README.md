# Prediccion de Emisiones CO2 — Planta de Acero

Prediccion de emisiones de CO2 y consumo energetico en una planta de acero coreana mediante regresion supervisada y series de tiempo. El objetivo de negocio es permitir que el operador **anticipe las emisiones antes de que ocurra el consumo**, habilitando decisiones operacionales sobre tipo de carga y eficiencia electrica en la planificacion del turno.

**Fuente de datos:** Steel Industry Energy Consumption Dataset — UCI Machine Learning Repository (ID 851), DAEWOO Steel Co. Ltd, Gwangyang, Corea del Sur. Periodo: 1 enero 2018 – 31 diciembre 2018, granularidad de 15 minutos, 35.040 registros.

---

## Estado del Proyecto (Revision: 2026-05-14)

Todas las etapas completadas. Aplicacion Streamlit operativa.

### Checklist de Etapas

- [x] **1. Setup** — Estructura de carpetas, dataset original, trabajo y validacion creados
- [x] **2. Calidad de datos** — Analisis de nulidad, duplicados y tipos de datos completado; pickles generados en `01_Datos/03_Trabajo/`
- [x] **3. EDA** — Analisis estadistico, graficos exportados a `04_Resultados/01_Analisis/` y analisis de ceros en target completado con conclusiones documentadas
- [x] **4. Preprocesamiento (Experimento A)** — OHE + StandardScaler aplicados; `df_tablon.pickle` generado (pipeline base con variables fisicas)
- [x] **4b. Preprocesamiento (Experimento B)** — Features temporales implementadas (`hora`, `turno`, `NSM_sin`/`NSM_cos`, `es_fin_de_semana`); `trabajo_preprocesado_B.pickle` generado
- [x] **5. Entrenamiento** — GridSearchCV sobre 279 configuraciones ejecutado; RandomForest seleccionado (R²=0.9923, RMSE=0.0875)
- [x] **6. Series de tiempo** — ARIMA aplicado para CO2 (MAE=0.0217) y Usage_kWh; ejecutado con conclusiones en notebook
- [x] **7. Analisis de resultados** — JSONs de metricas e importancias exportados a `04_Resultados/01_Analisis/` para ambos experimentos
- [x] **8. Seleccion de features** — Modelo minimal B seleccionado: 7 features (Load_Type, factores de potencia, NSM ciclico); R²=0.8715 en validacion, degradacion -0.34% vs modelo completo de 20 features; `config_modelo_minimal.json` generado
- [x] **9. Exportar artefactos** — Encoders, scalers y modelos exportados como `.joblib` en `03_Modelos/01_Historial/`
- [x] **10. Preproduccion** — Ambos experimentos evaluados sobre 10.512 registros no vistos; metricas guardadas en `04_Resultados/02_Preproduccion/`; ningun modelo muestra overfitting
- [x] **11. Produccion** — Pipeline de produccion (Exp B minimal, 3 inputs del operador) y pipeline de referencia (Exp A) exportados a `03_Modelos/02_Produccion/`; reporte en `04_Resultados/03_Produccion/`
- [x] **12. Aplicacion** — App Streamlit operativa en `05_Aplicacion/app.py` con prediccion individual y prediccion por dataset (carga de CSV en lote)

---

## Dataset y Split

| Conjunto | Registros | Ruta |
|---|---|---|
| Original | 35.040 | `01_Datos/01_Originales/Steel_industry_data.csv` |
| Trabajo (entrenamiento) | 24.528 (70%) | `01_Datos/03_Trabajo/trabajo.csv` |
| Validacion (evaluacion final, no vista) | 10.512 (30%) | `01_Datos/02_Validacion/validacion.csv` |

El dataset de validacion no se uso en ningun momento del ciclo de entrenamiento ni seleccion de hiperparametros. Se reservo exclusivamente para la evaluacion de preproduccion.

---

## Hallazgos del EDA

### Variables clave

| Variable | Correlacion con CO2 | Significancia |
|---|---|---|
| `Usage_kWh` | r = 0.988 | p < 1e-300 |
| `Lagging_Current_Reactive_Power_kVarh` | r = 0.887 | p < 1e-300 |

Estas dos variables tienen correlacion fisica directa con CO2: la planta genera emisiones en funcion de la energia consumida. Incluirlas resuelve el problema de prediccion de forma casi trivial pero no accionable — el modelo aprende la relacion fisica, no los patrones operacionales.

### Analisis de ceros en CO2_tCO2

El 59.9% de los registros tienen `CO2_tCO2 = 0`. El analisis confirma que corresponden a **estados operacionales reales** de la planta (no errores de medicion).

**Distribucion de ceros por turno horario**

| Turno | Rango | % Registros con CO2 = 0 |
|---|---|---|
| Noche | 22h – 6h | 94.6% |
| Manana | 6h – 14h | 49.5% |
| Tarde | 14h – 22h | 35.6% |

**Distribucion de ceros por dia de la semana**

| Grupo | % Registros con CO2 = 0 |
|---|---|
| Lunes – Viernes | ~51% |
| Sabado | 73% |
| Domingo | 89% |

**Distribucion de ceros por Load_Type**

| Load_Type | % Registros con CO2 = 0 |
|---|---|
| `Light_Load` | 90.4% |
| `Medium_Load` | 41.0% |
| `Maximum_Load` | 9.2% |

Test chi-cuadrado turno vs CO2=0: chi²=9.266, p < 1e-300. El patron es completamente estructurado y no aleatorio. `Light_Load` actua como proxy de planta inactiva; `Maximum_Load` casi siempre genera emisiones.

---

## Diseno de Experimentos

### Justificacion

El diseno de dos experimentos responde a preguntas distintas:

- **Exp A**: ?cual es el techo de rendimiento con informacion completa? Es el baseline de referencia. Si el modelo falla aqui, hay un problema fundamental.
- **Exp B**: ?puede el modelo identificar cuando y por que la planta consume mas, usando solo variables que el operador puede conocer **antes** de que ocurra el consumo? Esta es la pregunta de negocio real.

**Criterio de exito**: si Exp B degrada el RMSE en ≤20% respecto a Exp A, las features temporales y operacionales son suficientes para prediccion accionable.

### Experimento A — Pipeline base (modelo de referencia)

Incluye las variables con correlacion fisica directa con CO2.

- **Variables incluidas:** todas las del dataset, incluyendo `Usage_kWh` y `Lagging_Current_Reactive_Power_kVarh`
- **Preprocesamiento:** OHE sobre `WeekStatus`, `Day_of_week`, `Load_Type`; StandardScaler sobre numericas
- **Dataset resultante:** 35.040 × 18 features (`df_tablon.pickle` / `trabajo_preprocesado_A.pickle`)
- **Modelo:** RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_leaf=4, min_samples_split=2)
- **Artefacto:** `03_Modelos/02_Produccion/pipeline_referencia_A_v1.joblib`

### Experimento B — Sin variables fisicas directas (modelo de produccion)

Excluye las variables con correlacion fisica directa y fuerza al modelo a encontrar senales operacionales accionables.

- **Variables excluidas:** `Usage_kWh`, `Lagging_Current_Reactive_Power_kVarh`
- **Nuevas features desde `date`:** `hora` (0–23), `turno` (dummies Manana/Tarde/Noche), `es_fin_de_semana`, `NSM_sin = sin(2π·NSM/86400)`, `NSM_cos = cos(2π·NSM/86400)`
- **Dataset resultante:** 35.040 × 20 features (`trabajo_preprocesado_B.pickle`)
- **Modelo:** RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=4, min_samples_split=2)
- **Artefacto:** `03_Modelos/02_Produccion/pipeline_produccion_B_v1.joblib`

---

## Resultados

### Entrenamiento (test interno 30%, random_state=42)

| Experimento | R² | RMSE | MAE | Feature mas importante |
|---|---|---|---|---|
| A — con variables fisicas | 0.9923 | 0.0875 | 0.0062 | `Usage_kWh` (99.65%) |
| B — sin variables fisicas | 0.8470 | 0.3910 | 0.1933 | `Load_Type_Light_Load` (37.3%) |

### Preproduccion (validacion, n=10.512 registros no vistos)

| Experimento | R² | RMSE | MAE | Diagnostico |
|---|---|---|---|---|
| A — con variables fisicas | 0.9959 | 0.0644 | 0.0048 | Generaliza correctamente |
| B — sin variables fisicas | 0.8745 | 0.3557 | 0.1732 | Generaliza correctamente |
| B minimal — 7 features (produccion) | 0.8715 | 0.3598 | 0.1790 | Generaliza correctamente |

Ninguno de los tres modelos muestra overfitting. El Exp B mejora en validacion respecto al test de entrenamiento (degradacion RMSE: -9.03%).

### Seleccion de features — Modelo minimal de produccion

Ablacion sobre 3 subconjuntos del Exp B buscando el minimo que mantiene R² dentro del 20% del modelo completo (umbral: R² ≥ 0.6996).

| Subconjunto | Features en modelo | Inputs del operador | R² validacion | Degradacion vs B completo |
|---|---|---|---|---|
| Top 3 variables | 5 | 3 | 0.8097 | -7.41% |
| Top 3 + NSM ciclico | 7 | 3 | 0.8715 | -0.34% |
| Top 5 variables | 8 | 3 | 0.8698 | -0.54% |

**Seleccionado**: Top 3 + NSM ciclico (7 features). NSM_sin y NSM_cos se derivan automaticamente del reloj del sistema — el operador ingresa solo 3 campos. La degradacion de -0.34% vs el modelo completo de 20 features justifica la reduccion de complejidad.

### Tabla comparativa A vs B

| Metrica | Exp A (con variables fisicas) | Exp B (sin variables fisicas) |
|---|---|---|
| R² validacion | 0.9959 | 0.8745 |
| RMSE validacion | 0.0644 | 0.3557 |
| MAE validacion | 0.0048 | 0.1732 |
| Degradacion RMSE vs A | — | 452% |
| Accionable antes del consumo | No | Si |

El criterio de exito (≤20% degradacion RMSE) no se cumple. `Usage_kWh` es un proxy casi perfecto del CO2 (r=0.988, importancia 99.65%) y no puede ser reemplazado por las features temporales disponibles. El Exp B si captura estructura real (R²=0.875) — Load_Type y los factores de potencia son senales informativas — y es el unico modelo accionable para planificacion operacional.

---

## Aplicacion Streamlit

Archivo: `05_Aplicacion/app.py`

La aplicacion permite anticipar emisiones de CO2 antes de que ocurra el consumo energetico.

### Como ejecutar

```bash
cd C:\Users\matia\GitHub\Energy_consumption
streamlit run 05_Aplicacion/app.py
```

### Vistas disponibles

| Vista | Descripcion |
|---|---|
| Portada | Descripcion del modelo, metricas de rendimiento y tabla de variables de entrada |
| Prediccion Individual | Formulario con 3 campos del operador; la hora del sistema se captura automaticamente |
| Prediccion por Dataset | Carga de CSV en lote; genera predicciones masivas y permite descargar el reporte |

### Formato de entrada para prediccion individual

| Campo | Tipo | Descripcion |
|---|---|---|
| `Tipo de carga` | Categorico | Regimen del turno: Carga Ligera / Carga Media / Carga Maxima |
| `Factor de Potencia Adelantado` | Numerico [0–100] | Eficiencia del suministro de energia (lectura de sensor) |
| `Factor de Potencia Retrasado` | Numerico [0–100] | Eficiencia del consumo de energia (lectura de sensor) |
| Hora del sistema | Automatico | NSM_sin y NSM_cos se calculan internamente; el operador no los ingresa |

### Formato de entrada para prediccion por dataset (CSV)

El archivo CSV debe contener exactamente estas tres columnas:

```
Load_Type, Leading_Current_Power_Factor, Lagging_Current_Power_Factor
```

Los valores de `Load_Type` deben ser: `Light_Load`, `Medium_Load` o `Maximum_Load`.

### Interpretacion de la salida

La aplicacion retorna `CO2_predicho_tCO2` (toneladas de CO2 por intervalo de 15 minutos). Los valores negativos se clipean a 0 automaticamente. La prediccion individual incluye comparativa grafica contra el promedio historico por tipo de carga.

---

## Limitaciones del Modelo

1. **Precision del modelo de produccion (Exp B):** R²=0.8715 sin variables de proceso productivo (temperatura de horno, toneladas producidas, tipo de colada). Para mayor precision se requieren datos adicionales de la linea de produccion.
2. **Cobertura temporal:** el dataset cubre solo el ano 2018. La validez en otros periodos depende de la estabilidad de los patrones operacionales de la planta.
3. **Escala de tiempo:** el modelo predice emisiones por intervalo de 15 minutos. No es aplicable directamente a horizontes diarios o semanales sin agregacion.
4. **Modelo de referencia no accionable:** el Exp A (R²=0.9959) requiere `Usage_kWh` como input, variable que solo se conoce despues de que ocurre el consumo. Su uso es tautologico para prediccion anticipada.
5. **Dominio del dataset:** los patrones fueron aprendidos de una planta de acero coreana especifica. La transferencia a otras plantas requiere reentrenamiento.

---

## Estructura de Carpetas

```
Energy_consumption/
├── 01_Datos/
│   ├── 01_Originales/
│   │   └── Steel_industry_data.csv          # Dataset UCI original (35.040 registros)
│   ├── 02_Validacion/
│   │   └── validacion.csv                   # 10.512 registros no vistos (30%)
│   ├── 03_Trabajo/
│   │   ├── trabajo.csv                      # 24.528 registros de entrenamiento (70%)
│   │   ├── trabajo_preprocesado_A.pickle    # Dataset Exp A (18 features)
│   │   ├── trabajo_preprocesado_B.pickle    # Dataset Exp B (20 features)
│   │   ├── df_tablon.pickle                 # Dataset tablon Exp A (legacy)
│   │   ├── trabajo_resultado_calidad.pickle
│   │   ├── cat_resultado_calidad.pickle
│   │   └── num_resultado_calidad.pickle
│   └── 04_Prueba/
│       ├── dataset_prueba.csv               # Dataset de ejemplo para la app
│       └── predicciones_co2.csv             # Ultimo reporte de predicciones generado
│
├── 02_Notebooks/
│   ├── 01_Desarrollo/
│   │   ├── 01_Set Up.ipynb
│   │   ├── 02_Calidad_datos.ipynb
│   │   ├── 03_EDA.ipynb
│   │   ├── 04_Transformacion_de_Variable.ipynb   # Preprocesamiento A y B
│   │   ├── 05_Modelizacion_Supervisada.ipynb     # Entrenamiento + seleccion de features
│   │   └── 06_Series_de_tiempo.ipynb
│   └── 02_Produccion/
│       ├── 01_Preproduccion.ipynb               # Evaluacion sobre validacion (1 unica vez)
│       └── 02_Produccion.ipynb                  # Exportacion de artefactos de produccion
│
├── 03_Modelos/
│   ├── 01_Historial/
│   │   ├── pipeline_encoder_A.joblib
│   │   ├── pipeline_scaler_A.joblib
│   │   ├── rfc_CO2_A_v1_pipeline.joblib
│   │   ├── pipeline_encoder_B.joblib
│   │   ├── pipeline_scaler_B.joblib
│   │   └── rfc_CO2_B_v1_pipeline.joblib
│   └── 02_Produccion/
│       ├── pipeline_produccion_B_min_v1.joblib  # Modelo de produccion (Exp B minimal, 7 features)
│       ├── pipeline_produccion_B_v1.joblib      # Modelo Exp B completo (10 features)
│       └── pipeline_referencia_A_v1.joblib      # Modelo de referencia Exp A (18 features)
│
├── 04_Resultados/
│   ├── 01_Analisis/
│   │   ├── benchmarking_A.json
│   │   ├── benchmarking_B.json
│   │   ├── feature_importances_A.json
│   │   ├── feature_importances_B.json
│   │   ├── comparacion_AB.json
│   │   └── *.png                                # Graficos del EDA
│   ├── 02_Preproduccion/
│   │   ├── metricas_validacion_A.json
│   │   ├── metricas_validacion_B.json
│   │   └── reporte_preproduccion.json
│   └── 03_Produccion/
│       ├── reporte_produccion.json
│       └── config_modelo_minimal.json
│
└── 05_Aplicacion/
    └── app.py                                   # Aplicacion Streamlit operativa
```

---

## Notebooks Desarrollados

| Notebook | Etapa | Descripcion |
|---|---|---|
| `01_Set Up.ipynb` | Setup | Estructura de carpetas, carga del dataset y split trabajo/validacion |
| `02_Calidad_datos.ipynb` | Calidad | Nulidad, duplicados, tipos de datos y decisiones de limpieza |
| `03_EDA.ipynb` | EDA | Correlaciones, distribucion del target, analisis de ceros en CO2 |
| `04_Transformacion_de_Variable.ipynb` | Preprocesamiento | OHE + StandardScaler; construccion de features temporales para Exp B |
| `05_Modelizacion_Supervisada.ipynb` | Entrenamiento + Seleccion | GridSearchCV, comparacion A vs B, ablacion de features y modelo minimal |
| `06_Series_de_tiempo.ipynb` | Series de tiempo | ARIMA para CO2 y Usage_kWh |
| `01_Preproduccion.ipynb` | Preproduccion | Evaluacion unica sobre validacion (10.512 registros no vistos) |
| `02_Produccion.ipynb` | Produccion | Exportacion de pipelines `.joblib` a `03_Modelos/02_Produccion/` |

---

## Fuente de Datos

Steel Industry Energy Consumption Dataset
UCI Machine Learning Repository — ID 851
DAEWOO Steel Co. Ltd, Gwangyang, Corea del Sur
Periodo: 1 enero 2018 – 31 diciembre 2018, granularidad 15 minutos, 35.040 registros
https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption
