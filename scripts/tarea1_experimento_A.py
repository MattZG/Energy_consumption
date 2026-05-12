"""
TAREA 1 — Exportar artefactos del Experimento A
Replica el pipeline del Experimento A con random_state=42 y guarda artefactos.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

print(f"cat shape: {cat.shape}  |  columnas: {list(cat.columns)}")
print(f"num shape: {num.shape}  |  columnas: {list(num.columns)}")

# ── 2. Separar date del dataframe categórico ────────────────────────────────
date_series = cat[['date']].copy()
cat_features = cat.drop(columns=['date'])   # WeekStatus, Day_of_week, Load_Type

# ── 3. OHE sobre cat_features ───────────────────────────────────────────────
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_ohe_arr = ohe.fit_transform(cat_features)
cat_ohe_cols = [f"{col}_{val}" for col, vals in zip(cat_features.columns, ohe.categories_) for val in vals]
cat_ohe = pd.DataFrame(cat_ohe_arr, columns=cat_ohe_cols, index=cat_features.index)

print(f"cat_ohe shape: {cat_ohe.shape}  |  columnas: {list(cat_ohe.columns)}")

# ── 4. StandardScaler sobre todas las numéricas (7 columnas) ─────────────────
num_cols = ['Usage_kWh', 'Lagging_Current_Reactive_Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh', 'Lagging_Current_Power_Factor',
            'Leading_Current_Power_Factor', 'NSM', 'CO2_tCO2']

scaler = StandardScaler()
num_std_arr = scaler.fit_transform(num[num_cols])
num_std_cols = [f"{c}_std_" for c in num_cols]
num_std = pd.DataFrame(num_std_arr, columns=num_std_cols, index=num.index)

print(f"num_std shape: {num_std.shape}  |  columnas: {list(num_std.columns)}")

# ── 5. Concatenar: date + cat_ohe + num_std → df_tablon ─────────────────────
df_tablon = pd.concat([date_series.reset_index(drop=True),
                        cat_ohe.reset_index(drop=True),
                        num_std.reset_index(drop=True)], axis=1)

print(f"df_tablon shape: {df_tablon.shape}  |  columnas: {list(df_tablon.columns)}")

# ── 6. X e y ────────────────────────────────────────────────────────────────
X = df_tablon.drop(columns=['date', 'CO2_tCO2_std_'])
y = df_tablon['CO2_tCO2_std_']

print(f"X shape: {X.shape}  |  y shape: {y.shape}")
print(f"Features: {list(X.columns)}")

# ── 7. Train-test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── 8. Entrenar RandomForest con los hiperparámetros del Exp A ───────────────
rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=None,
    min_samples_leaf=4,
    min_samples_split=2,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
print("Modelo entrenado.")

# ── 9. Evaluar sobre test ────────────────────────────────────────────────────
y_pred = rf.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print(f"\n=== MÉTRICAS EXPERIMENTO A ===")
print(f"R²   = {r2:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"MAE  = {mae:.4f}")

# ── 10. Guardar artefactos ───────────────────────────────────────────────────
historial_dir = repo_root / "03_Modelos/01_Historial"
trabajo_dir   = repo_root / "01_Datos/03_Trabajo"

historial_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(ohe,   historial_dir / "pipeline_encoder_A.joblib")
joblib.dump(scaler, historial_dir / "pipeline_scaler_A.joblib")
joblib.dump(rf,    historial_dir / "rfc_CO2_A_v1_pipeline.joblib")
df_tablon.to_pickle(trabajo_dir / "trabajo_preprocesado_A.pickle")

print("\n=== ARTEFACTOS GUARDADOS ===")
for p in [
    historial_dir / "pipeline_encoder_A.joblib",
    historial_dir / "pipeline_scaler_A.joblib",
    historial_dir / "rfc_CO2_A_v1_pipeline.joblib",
    trabajo_dir   / "trabajo_preprocesado_A.pickle"
]:
    exists = p.exists()
    size   = p.stat().st_size if exists else 0
    print(f"  {'OK' if exists else 'FALTA'} | {size:>10} bytes | {p.relative_to(repo_root)}")

print("\nTAREA 1 COMPLETADA.")