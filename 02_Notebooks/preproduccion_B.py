"""
Preproduccion — Experimento B
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

# ---------------------------------------------------------------------------
# 2. Cargar artefactos Exp B
# ---------------------------------------------------------------------------
hist_dir = repo_root / "03_Modelos" / "01_Historial"

encoder_B = joblib.load(hist_dir / "pipeline_encoder_B.joblib")
scaler_B  = joblib.load(hist_dir / "pipeline_scaler_B.joblib")
modelo_B  = joblib.load(hist_dir / "rfc_CO2_B_v1_pipeline.joblib")

print("Artefactos Exp B cargados.")

# ---------------------------------------------------------------------------
# 3. Construir features Exp B desde datos de validacion
# ---------------------------------------------------------------------------
# Convertir date a datetime
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# Features temporales
df["hora"] = df["date"].dt.hour
df["turno"] = df["date"].dt.hour.map(
    lambda h: "Mañana" if 6 <= h < 14 else ("Tarde" if 14 <= h < 22 else "Noche")
)
df["es_fin_de_semana"] = (df["WeekStatus"] == "Weekend").astype(int)
df["NSM_sin"] = np.sin(2 * np.pi * df["NSM"] / 86400)
df["NSM_cos"] = np.cos(2 * np.pi * df["NSM"] / 86400)

# Columnas categoricas para OHE Exp B (mismo orden que entrenamiento)
cat_B = df[["Day_of_week", "Load_Type", "turno"]]

# Columnas numericas para scaler Exp B (mismo orden que entrenamiento)
num_cols_B = [
    "Leading_Current_Reactive_Power_kVarh",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM_sin",
    "NSM_cos",
    "hora",
    "CO2_tCO2",
]
num_B = df[num_cols_B]

print(f"\nDistribucion de turnos:\n{df['turno'].value_counts()}")
print(f"\nCat B shape: {cat_B.shape}")
print(f"Num B shape: {num_B.shape}")

# ---------------------------------------------------------------------------
# 4. Transformar (solo .transform, nunca .fit)
# ---------------------------------------------------------------------------
cat_ohe_arr_B = encoder_B.transform(cat_B)
cat_ohe_cols_B = encoder_B.get_feature_names_out()
cat_ohe_B = pd.DataFrame(cat_ohe_arr_B, columns=cat_ohe_cols_B, index=df.index)

num_std_arr_B = scaler_B.transform(num_B)  # CO2_tCO2 es la ultima columna (indice 6)

# Nombres con sufijo _std_
num_std_cols_B = [c + "_std_" for c in num_cols_B]
num_std_B = pd.DataFrame(num_std_arr_B, columns=num_std_cols_B, index=df.index)

# es_fin_de_semana como columna independiente
es_fin_col = df[["es_fin_de_semana"]].copy()

# Construir tablon Exp B: cat_ohe + es_fin_de_semana + num_std
tablon_B = pd.concat([cat_ohe_B, es_fin_col, num_std_B], axis=1)

# Separar X e y
y_val_B = num_std_arr_B[:, 6]  # CO2_tCO2_std_ es el indice 6
X_val_B = tablon_B.drop(columns=["CO2_tCO2_std_"])

print(f"\nTablon B: {tablon_B.shape}")
print(f"X_val_B: {X_val_B.shape} | y_val_B shape: {y_val_B.shape}")
print("Features X_val_B:", X_val_B.columns.tolist())

# ---------------------------------------------------------------------------
# 5. Prediccion
# ---------------------------------------------------------------------------
y_pred_B = modelo_B.predict(X_val_B)

# ---------------------------------------------------------------------------
# 6. Metricas
# ---------------------------------------------------------------------------
r2_val_B   = r2_score(y_val_B, y_pred_B)
rmse_val_B = np.sqrt(mean_squared_error(y_val_B, y_pred_B))
mae_val_B  = mean_absolute_error(y_val_B, y_pred_B)

# Referencias entrenamiento
r2_train_B   = 0.846997
rmse_train_B = 0.390998
mae_train_B  = 0.193321

degradacion_rmse_B = (rmse_val_B - rmse_train_B) / rmse_train_B * 100
diagnostico_B = "Señal de overfitting" if degradacion_rmse_B > 10 else "Generaliza correctamente"

print("\n=== METRICAS EXPERIMENTO B — VALIDACION ===")
print(f"R²   : {r2_val_B:.6f}  (ref train: {r2_train_B})")
print(f"RMSE : {rmse_val_B:.6f}  (ref train: {rmse_train_B})")
print(f"MAE  : {mae_val_B:.6f}  (ref train: {mae_train_B})")
print(f"Degradacion RMSE: {degradacion_rmse_B:.2f}%")
print(f"Diagnostico: {diagnostico_B}")

# ---------------------------------------------------------------------------
# 7. Guardar JSON
# ---------------------------------------------------------------------------
out_dir = repo_root / "04_Resultados" / "02_Preproduccion"
out_dir.mkdir(parents=True, exist_ok=True)

resultado_B = {
    "experimento": "B",
    "dataset": "validacion",
    "n_registros": int(df.shape[0]),
    "metricas_validacion": {
        "r2":   round(r2_val_B, 6),
        "rmse": round(rmse_val_B, 6),
        "mae":  round(mae_val_B, 6),
    },
    "metricas_entrenamiento_ref": {
        "r2":   r2_train_B,
        "rmse": rmse_train_B,
        "mae":  mae_train_B,
    },
    "degradacion_rmse_pct": round(degradacion_rmse_B, 4),
    "diagnostico": diagnostico_B,
}

out_path = out_dir / "metricas_validacion_B.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(resultado_B, f, indent=2, ensure_ascii=False)

# Verificar
assert out_path.exists(), "ERROR: archivo no creado"
with open(out_path, encoding="utf-8") as f:
    verificado = json.load(f)
print(f"\nArchivo guardado y verificado: {out_path}")
print(json.dumps(verificado, indent=2, ensure_ascii=False))
