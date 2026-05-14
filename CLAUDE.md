# Energy Consumption

Predicción de emisiones de CO2 y consumo energético en plantas de acero 
mediante regresión supervisada y series de tiempo.

## Estructura
- `01_Datos/` — originales, validación, trabajo y prueba
- `02_Notebooks/` — desarrollo y producción
- `03_Modelos/` — historial y producción
- `04_Resultados/` — análisis, preproducción y producción
- `05_Aplicacion/` — app Streamlit

## Skills disponibles
- (por definir cuando se generalicen)

## Convenciones
- Datos de entrenamiento en `01_Datos/03_Trabajo/`
- Artefactos en `03_Modelos/01_Historial/`
- Dataset de validación en `01_Datos/02_Validacion/`

## Convención de invocación de agentes

Siempre cargar la Skill del experto antes de ejecutar cualquier sub-agente:

@nombre-agente lee primero .claude/agents/Experto_Energetico.md 
para entender el contexto del dominio y luego [tarea]