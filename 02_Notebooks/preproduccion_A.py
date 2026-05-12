"""
Preproduccion — Experimento A
Evaluacion del pipeline entrenado sobre datos de validacion (nunca .fit).
"""
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------------
# Raiz del repositorio
# ---------------------------------------------------------------------------
repo_root = Path.cwd().resolve()
while repo_root != repo_root.parent and not (repo_root / "README.md").exists():
    repo_root = repo_root.parent
if not (repo_root / "README.md").exists():
    raise FileNotFoundError("No se encontro la raiz del repositorio.")

print(f"Repositorio: {repo_root}")

# ---------------------------------------------------------------------------
# 1. Cargar validacion.csv y renombrar columnas inconsistentes
# ---------------------------------------------------------------------------
val_path = repo_root / "01_Datos" / "02_Validacion" / "validacion.csv"
df = pd.read_csv(val_path)

df = df.rename(columns={
    "Lagging_Current_Reactive.Power_kVarh": "Lagging_Current_Reactive_Power_kVarh",
    "CO2(tCO2)": "CO2_tCO2",
})

print(f"Validacion cargada: {df.shape[0]} registros, {df.shape[1]} columnas")
print("Columnas:", df.columns.tolist())

# ---------------------------------------------------------------------------
# 2. Cargar artefactos Exp A
# ---------------------------------------------------------------------------
hist_dir = repo_root / "03_Modelos" / "01_Historial"

encoder_A = joblib.load(hist_dir / "pipeline_encoder_A.joblib")
scaler_A  = joblib.load(hist_dir / "pipeline_scaler_A.joblib")
modelo_A  = joblib.load(hist_dir / "rfc_CO2_A_v1_pipeline.joblib")

print("\nArtefactos Exp A cargados.")

# ---------------------------------------------------------------------------
# 3. Preparar datos de validacion — mismo orden que entrenamiento
# ---------------------------------------------------------------------------
# Columnas categoricas (OHE Exp A): WeekStatus, Day_of_week, Load_Type
cat_A = df[["WeekStatus", "Day_of_week", "Load_Type"]]

# Transformar sin fit
cat_ohe_arr = encoder_A.transform(cat_A)
cat_ohe_cols = encoder_A.get_feature_names_out()
cat_ohe = pd.DataFrame(cat_ohe_arr, columns=cat_ohe_cols, index=df.index)

# Columnas numericas Exp A (orden exacto del entrenamiento)
num_cols_A = [
    "Usage_kWh",
    "Lagging_Current_Reactive_Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "CO2_tCO2",
]
num_A = df[num_cols_A]

# Transformar sin fit
num_std_arr = scaler_A.transform(num_A)
# Nombres: agregar sufijo _std_ para CO2_tCO2
num_std_cols = [c + "_std_" for c in num_cols_A]
num_std = pd.DataFrame(num_std_arr, columns=num_std_cols, index=df.index)

# Construir tablon de validacion
tablon_A = pd.concat([cat_ohe, num_std], axis=1)

# Separar X e y
X_val_A = tablon_A.drop(columns=["CO2_tCO2_std_"])
y_val_A = tablon_A["CO2_tCO2_std_"]

print(f"\nTablon A: {tablon_A.shape}")
print(f"X_val_A: {X_val_A.shape} | y_val_A: {y_val_A.shape}")
print("Features X_val_A:", X_val_A.columns.tolist())

# ---------------------------------------------------------------------------
# 4. Prediccion
# ---------------------------------------------------------------------------
y_pred_A = modelo_A.predict(X_val_A)

# ---------------------------------------------------------------------------
# 5. Metricas
# ---------------------------------------------------------------------------
r2_val   = r2_score(y_val_A, y_pred_A)
rmse_val = np.sqrt(mean_squared_error(y_val_A, y_pred_A))
mae_val  = mean_absolute_error(y_val_A, y_pred_A)

# Referencias entrenamiento
r2_train   = 0.992343
rmse_train = 0.08747
mae_train  = 0.00616

degradacion_rmse = (rmse_val - rmse_train) / rmse_train * 100
diagnostico = "Señal de overfitting" if degradacion_rmse > 10 else "Generaliza correctamente"

print("\n=== METRICAS EXPERIMENTO A — VALIDACION ===")
print(f"R²   : {r2_val:.6f}  (ref train: {r2_train})")
print(f"RMSE : {rmse_val:.6f}  (ref train: {rmse_train})")
print(f"MAE  : {mae_val:.6f}  (ref train: {mae_train})")
print(f"Degradacion RMSE: {degradacion_rmse:.2f}%")
print(f"Diagnostico: {diagnostico}")

# ---------------------------------------------------------------------------
# 6. Guardar JSON
# ---------------------------------------------------------------------------
out_dir = repo_root / "04_Resultados" / "02_Preproduccion"
out_dir.mkdir(parents=True, exist_ok=True)

resultado_A = {
    "experimento": "A",
    "dataset": "validacion",
    "n_registros": int(df.shape[0]),
    "metricas_validacion": {
        "r2":   round(r2_val, 6),
        "rmse": round(rmse_val, 6),
        "mae":  round(mae_val, 6),
    },
    "metricas_entrenamiento_ref": {
        "r2":   r2_train,
        "rmse": rmse_train,
        "mae":  mae_train,
    },
    "degradacion_rmse_pct": round(degradacion_rmse, 4),
    "diagnostico": diagnostico,
}

out_path = out_dir / "metricas_validacion_A.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(resultado_A, f, indent=2, ensure_ascii=False)

# Verificar
assert out_path.exists(), "ERROR: archivo no creado"
with open(out_path, encoding="utf-8") as f:
    verificado = json.load(f)
print(f"\nArchivo guardado y verificado: {out_path}")
print(json.dumps(verificado, indent=2, ensure_ascii=False))
