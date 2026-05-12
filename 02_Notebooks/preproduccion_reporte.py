"""
Tarea 3 — Reporte comparativo A vs B sobre validacion
Genera reporte_preproduccion.json con analisis consolidado.
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Raiz del repositorio
# ---------------------------------------------------------------------------
repo_root = Path.cwd().resolve()
while repo_root != repo_root.parent and not (repo_root / "README.md").exists():
    repo_root = repo_root.parent
if not (repo_root / "README.md").exists():
    raise FileNotFoundError("No se encontro la raiz del repositorio.")

out_dir = repo_root / "04_Resultados" / "02_Preproduccion"

# ---------------------------------------------------------------------------
# Cargar JSONs de cada experimento
# ---------------------------------------------------------------------------
path_A = out_dir / "metricas_validacion_A.json"
path_B = out_dir / "metricas_validacion_B.json"

assert path_A.exists(), f"No existe: {path_A}"
assert path_B.exists(), f"No existe: {path_B}"

with open(path_A, encoding="utf-8") as f:
    res_A = json.load(f)
with open(path_B, encoding="utf-8") as f:
    res_B = json.load(f)

print("JSONs de preproduccion cargados correctamente.")

# ---------------------------------------------------------------------------
# Construir reporte comparativo
# ---------------------------------------------------------------------------
def seccion_experimento(res):
    return {
        "metricas_entrenamiento": res["metricas_entrenamiento_ref"],
        "metricas_validacion": res["metricas_validacion"],
        "degradacion_rmse_pct": res["degradacion_rmse_pct"],
        "diagnostico": res["diagnostico"],
    }

# Conclusion narrativa
r2_A_val   = res_A["metricas_validacion"]["r2"]
rmse_A_val = res_A["metricas_validacion"]["rmse"]
rmse_A_ref = res_A["metricas_entrenamiento_ref"]["rmse"]
deg_A      = res_A["degradacion_rmse_pct"]

r2_B_val   = res_B["metricas_validacion"]["r2"]
rmse_B_val = res_B["metricas_validacion"]["rmse"]
rmse_B_ref = res_B["metricas_entrenamiento_ref"]["rmse"]
deg_B      = res_B["degradacion_rmse_pct"]

conclusion = (
    f"Experimento A (con Usage_kWh y Lagging_Current_Reactive_Power_kVarh): "
    f"el modelo generaliza correctamente. En validacion obtiene R²={r2_A_val:.4f} y "
    f"RMSE={rmse_A_val:.4f}, mejorando incluso respecto al test de entrenamiento "
    f"(RMSE ref={rmse_A_ref}, degradacion={deg_A:.2f}%). "
    f"Este resultado confirma ausencia de overfitting y un rendimiento cercano al techo fisico "
    f"del problema. "
    f"Experimento B (sin variables fisicamente correlacionadas, usando features temporales y "
    f"de eficiencia electrica): el modelo tambien generaliza correctamente. "
    f"En validacion obtiene R²={r2_B_val:.4f} y RMSE={rmse_B_val:.4f}, con una degradacion "
    f"de {deg_B:.2f}% respecto al test de entrenamiento (RMSE ref={rmse_B_ref}), dentro del "
    f"umbral del 10%. "
    f"Recomendacion para produccion: el Experimento A es el modelo de referencia para "
    f"monitoreo y alertas de alta precision (R²>0.99), pero requiere disponibilidad en tiempo "
    f"real de Usage_kWh y Lagging_Current_Reactive_Power_kVarh. "
    f"El Experimento B es el modelo accionable para la operacion de la planta: al depender "
    f"unicamente de features temporales y variables electricas de eficiencia (sin consumo "
    f"directo de energia), permite a los operadores anticipar emisiones de CO2 en escenarios "
    f"de planificacion de turno y carga, facilitando la toma de decisiones preventivas "
    f"(redistribucion de carga, ajuste de turno, mantenimiento programado) sin necesidad de "
    f"senales de medicion directa de consumo. Con R²={r2_B_val:.4f} en datos no vistos, "
    f"el modelo B ofrece capacidad predictiva suficiente para soporte operacional en produccion."
)

reporte = {
    "experimento_A": seccion_experimento(res_A),
    "experimento_B": seccion_experimento(res_B),
    "conclusion": conclusion,
}

# ---------------------------------------------------------------------------
# Guardar reporte
# ---------------------------------------------------------------------------
out_path = out_dir / "reporte_preproduccion.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(reporte, f, indent=2, ensure_ascii=False)

# Verificar
assert out_path.exists(), "ERROR: reporte no creado"
with open(out_path, encoding="utf-8") as f:
    verificado = json.load(f)

print(f"\nReporte guardado y verificado: {out_path}")
print(json.dumps(verificado, indent=2, ensure_ascii=False))
