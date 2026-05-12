"""
TAREA 3 — Implementar Experimento B (features accionables, sin Usage_kWh ni Lagging_Reactive)
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
import warnings
warnings.filterwarnings('ignore')

# ── Resolver raíz del repositorio ──────────────────────────────────────────
repo_root = Path.cwd().resolve()
while repo_root != repo_root.parent and not (repo_root / "README.md").exists():
    repo_root = repo_root.parent
if not (repo_root / "README.md").exists():
    raise FileNotFoundError("No se encontró la raíz del repositorio.")

print(f"repo_root: {repo_root}")

# ── 1. Cargar pickles fuente ────────────────────────────────────────────────
cat = pd.read_pickle(repo_root / "01_Datos/03_Trabajo/cat_resultado_calidad.pickle")
num = pd.read_pickle(repo_root / "01_Datos/03_Trabajo/num_resultado_calidad.pickle")

print(f"cat: {cat.shape}, num: {num.shape}")

# ── 1b. Crear features temporales ───────────────────────────────────────────
cat['hora'] = cat['date'].dt.hour                          # 0-23
cat['turno'] = cat['date'].dt.hour.map(
    lambda h: 'Mañana' if 6 <= h < 14
              else ('Tarde' if 14 <= h < 22 else 'Noche')
)
cat['es_fin_de_semana'] = (cat['WeekStatus'] == 'Weekend').astype(int)

num['NSM_sin'] = np.sin(2 * np.pi * num['NSM'] / 86400)
num['NSM_cos'] = np.cos(2 * np.pi * num['NSM'] / 86400)

print("Features temporales creadas:")
print(f"  cat extra: hora, turno, es_fin_de_semana")
print(f"  num extra: NSM_sin, NSM_cos")

# ── 2. Separar date y es_fin_de_semana ──────────────────────────────────────
date_series   = cat[['date']].copy().reset_index(drop=True)
efds_series   = cat[['es_fin_de_semana']].copy().reset_index(drop=True)

# Categóricas para OHE en Exp B: Day_of_week, Load_Type, turno
cat_B = cat[['Day_of_week', 'Load_Type', 'turno']].copy()

# ── 3. OHE sobre categóricas de Exp B ───────────────────────────────────────
ohe_B = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_ohe_B_arr = ohe_B.fit_transform(cat_B)
cat_ohe_B_cols = [f"{col}_{val}" for col, vals in zip(cat_B.columns, ohe_B.categories_) for val in vals]
cat_ohe_B = pd.DataFrame(cat_ohe_B_arr, columns=cat_ohe_B_cols, index=cat_B.index).reset_index(drop=True)

print(f"cat_ohe_B shape: {cat_ohe_B.shape}  |  columnas: {list(cat_ohe_B.columns)}")

# ── 4. StandardScaler sobre numéricas de Exp B ──────────────────────────────
# 'hora' viene de cat; la unimos temporalmente al dataframe numérico
num_B_source = num.copy().reset_index(drop=True)
num_B_source['hora'] = cat['hora'].values      # hora viene del cat procesado

num_B_cols = [
    'Leading_Current_Reactive_Power_kVarh',
    'Lagging_Current_Power_Factor',
    'Leading_Current_Power_Factor',
    'NSM_sin',
    'NSM_cos',
    'hora',
    'CO2_tCO2'
]

scaler_B = StandardScaler()
num_B_std_arr = scaler_B.fit_transform(num_B_source[num_B_cols])
num_B_std_cols = [f"{c}_std_" if c != 'CO2_tCO2' else 'CO2_tCO2_std_' for c in num_B_cols]
# Normalizar nombres de columnas con sufijo _std_
num_B_std_cols = []
for c in num_B_cols:
    if c == 'CO2_tCO2':
        num_B_std_cols.append('CO2_tCO2_std_')
    else:
        num_B_std_cols.append(f"{c}_std_")

num_std_B = pd.DataFrame(num_B_std_arr, columns=num_B_std_cols, index=num.index).reset_index(drop=True)

print(f"num_std_B shape: {num_std_B.shape}  |  columnas: {list(num_std_B.columns)}")

# ── 5. Concatenar: date + cat_ohe_B + es_fin_de_semana + num_std_B ──────────
df_tablon_B = pd.concat([
    date_series,
    cat_ohe_B,
    efds_series,
    num_std_B
], axis=1)

print(f"df_tablon_B shape: {df_tablon_B.shape}  |  columnas: {list(df_tablon_B.columns)}")

# ── 5b. X_B, y_B ────────────────────────────────────────────────────────────
X_B = df_tablon_B.drop(columns=['date', 'CO2_tCO2_std_'])
y_B = df_tablon_B['CO2_tCO2_std_']

print(f"X_B shape: {X_B.shape}  |  y_B shape: {y_B.shape}")
print(f"Features B: {list(X_B.columns)}")

# ── 6. Train-test split (mismo random_state=42) ──────────────────────────────
X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(
    X_B, y_B, test_size=0.3, random_state=42
)
print(f"Train: {X_B_train.shape}, Test: {X_B_test.shape}")

# ── 7. GridSearchCV (solo RandomForest) ─────────────────────────────────────
pipe_B = Pipeline([('algoritmo', RandomForestRegressor(random_state=42))])

grid_B = [{
    'algoritmo': [RandomForestRegressor(random_state=42)],
    'algoritmo__n_jobs': [-1],
    'algoritmo__n_estimators': [50, 100, 200],
    'algoritmo__max_depth': [None, 5, 10, 15],
    'algoritmo__min_samples_split': [2, 5],
    'algoritmo__min_samples_leaf': [1, 2, 4]
}]

grid_search_B = GridSearchCV(
    estimator=pipe_B,
    param_grid=grid_B,
    cv=5,
    scoring=make_scorer(mean_squared_error, greater_is_better=False),
    verbose=1,
    n_jobs=-1
)

print("\nEjecutando GridSearchCV Exp B...")
grid_search_B.fit(X_B_train, y_B_train)
print("GridSearchCV completado.")
print(f"Mejores parámetros: {grid_search_B.best_params_}")
print(f"Mejor score CV: {grid_search_B.best_score_:.6f}")

# ── 8. Evaluar mejor modelo sobre test ──────────────────────────────────────
best_model_B = grid_search_B.best_estimator_
y_B_pred = best_model_B.predict(X_B_test)

r2_B   = float(r2_score(y_B_test, y_B_pred))
rmse_B = float(np.sqrt(mean_squared_error(y_B_test, y_B_pred)))
mae_B  = float(mean_absolute_error(y_B_test, y_B_pred))

print(f"\n=== MÉTRICAS EXPERIMENTO B ===")
print(f"R²   = {r2_B:.4f}")
print(f"RMSE = {rmse_B:.4f}")
print(f"MAE  = {mae_B:.4f}")

# ── 9. Feature importances B ────────────────────────────────────────────────
rf_best_B = best_model_B.named_steps['algoritmo']
importances_B = rf_best_B.feature_importances_
feature_names_B = list(X_B.columns)
fi_B_list = sorted(
    [{"feature": name, "importance": float(imp)} for name, imp in zip(feature_names_B, importances_B)],
    key=lambda x: x["importance"],
    reverse=True
)

print("\nTop 5 features B:")
for fi in fi_B_list[:5]:
    print(f"  {fi['feature']}: {fi['importance']:.6f}")

# ── Comparación A vs B ───────────────────────────────────────────────────────
# Cargar métricas A
bench_A_path = repo_root / "04_Resultados/01_Analisis/benchmarking_A.json"
with open(bench_A_path, "r", encoding="utf-8") as f:
    bench_A = json.load(f)

r2_A   = bench_A["metricas_test"]["r2"]
rmse_A = bench_A["metricas_test"]["rmse"]
mae_A  = bench_A["metricas_test"]["mae"]
n_feat_A = bench_A["n_features"]

degradacion_pct = float((rmse_B - rmse_A) / rmse_A * 100)
criterio = "PASS" if rmse_B <= rmse_A * 1.20 else "FAIL"

print(f"\n{'='*55}")
print(f"Experimento A: R²={r2_A:.4f}, RMSE={rmse_A:.4f}, MAE={mae_A:.4f}")
print(f"Experimento B: R²={r2_B:.4f}, RMSE={rmse_B:.4f}, MAE={mae_B:.4f}")
print(f"Degradación RMSE: {degradacion_pct:.1f}%")
print(f"Criterio de exito (<=20% degradacion): {criterio}")
print(f"{'='*55}")

# ── Guardar variables para Tarea 4 en un archivo intermedio ─────────────────
# Se guardan los objetos que necesita Tarea 4
resultados_intermedios = {
    "r2_B": r2_B,
    "rmse_B": rmse_B,
    "mae_B": mae_B,
    "r2_A": r2_A,
    "rmse_A": rmse_A,
    "mae_A": mae_A,
    "n_feat_A": n_feat_A,
    "n_feat_B": len(feature_names_B),
    "degradacion_pct": degradacion_pct,
    "criterio": criterio,
    "fi_B_list": fi_B_list,
    "feature_names_B": feature_names_B,
    "best_params_B": str(grid_search_B.best_params_)
}

import pickle
with open(repo_root / "01_Datos/03_Trabajo/_temp_tarea3_resultados.pickle", "wb") as f:
    pickle.dump(resultados_intermedios, f)

# Guardar también los objetos principales para Tarea 4
joblib.dump(ohe_B,          repo_root / "03_Modelos/01_Historial/pipeline_encoder_B.joblib")
joblib.dump(scaler_B,       repo_root / "03_Modelos/01_Historial/pipeline_scaler_B.joblib")
joblib.dump(best_model_B,   repo_root / "03_Modelos/01_Historial/rfc_CO2_B_v1_pipeline.joblib")
df_tablon_B.to_pickle(      repo_root / "01_Datos/03_Trabajo/trabajo_preprocesado_B.pickle")

print("\n=== MODELOS Y TABLON B GUARDADOS ===")
for p in [
    repo_root / "03_Modelos/01_Historial/pipeline_encoder_B.joblib",
    repo_root / "03_Modelos/01_Historial/pipeline_scaler_B.joblib",
    repo_root / "03_Modelos/01_Historial/rfc_CO2_B_v1_pipeline.joblib",
    repo_root / "01_Datos/03_Trabajo/trabajo_preprocesado_B.pickle"
]:
    exists = p.exists()
    size   = p.stat().st_size if exists else 0
    print(f"  {'OK' if exists else 'FALTA'} | {size:>10} bytes | {p.relative_to(repo_root)}")

print("\nTAREA 3 COMPLETADA.")
