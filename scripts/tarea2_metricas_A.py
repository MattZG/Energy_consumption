"""
TAREA 2 — Exportar métricas y feature importances del Experimento A a JSON
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
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

# ── Cargar df_tablon y artefactos ya entrenados ─────────────────────────────
df_tablon = pd.read_pickle(repo_root / "01_Datos/03_Trabajo/trabajo_preprocesado_A.pickle")
rf = joblib.load(repo_root / "03_Modelos/01_Historial/rfc_CO2_A_v1_pipeline.joblib")

# ── Reconstruir X, y y split exacto ─────────────────────────────────────────
X = df_tablon.drop(columns=['date', 'CO2_tCO2_std_'])
y = df_tablon['CO2_tCO2_std_']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ── Métricas sobre test ─────────────────────────────────────────────────────
y_pred = rf.predict(X_test)
r2   = float(r2_score(y_test, y_pred))
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae  = float(mean_absolute_error(y_test, y_pred))

print(f"R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

# ── Feature importances ─────────────────────────────────────────────────────
importances = rf.feature_importances_
feature_names = list(X.columns)
fi_list = sorted(
    [{"feature": name, "importance": float(imp)} for name, imp in zip(feature_names, importances)],
    key=lambda x: x["importance"],
    reverse=True
)

print("\nTop 5 features:")
for fi in fi_list[:5]:
    print(f"  {fi['feature']}: {fi['importance']:.6f}")

# ── Construir benchmarking_A.json ───────────────────────────────────────────
benchmarking_A = {
    "experimento": "A",
    "modelo": "RandomForestRegressor",
    "hiperparametros": {
        "n_estimators": 50,
        "max_depth": None,
        "min_samples_leaf": 4,
        "min_samples_split": 2
    },
    "metricas_test": {
        "r2": round(r2, 6),
        "rmse": round(rmse, 6),
        "mae": round(mae, 6)
    },
    "n_features": len(feature_names),
    "features": feature_names
}

# ── Guardar JSONs ───────────────────────────────────────────────────────────
analisis_dir = repo_root / "04_Resultados/01_Analisis"
analisis_dir.mkdir(parents=True, exist_ok=True)

bench_path = analisis_dir / "benchmarking_A.json"
fi_path    = analisis_dir / "feature_importances_A.json"

with open(bench_path, "w", encoding="utf-8") as f:
    json.dump(benchmarking_A, f, ensure_ascii=False, indent=2)

with open(fi_path, "w", encoding="utf-8") as f:
    json.dump(fi_list, f, ensure_ascii=False, indent=2)

print("\n=== ARCHIVOS JSON GUARDADOS ===")
for p in [bench_path, fi_path]:
    exists = p.exists()
    size   = p.stat().st_size if exists else 0
    print(f"  {'OK' if exists else 'FALTA'} | {size:>8} bytes | {p.relative_to(repo_root)}")

# ── Verificar contenido ─────────────────────────────────────────────────────
print("\n--- benchmarking_A.json (preview) ---")
with open(bench_path, "r", encoding="utf-8") as f:
    print(f.read())

print("\n--- feature_importances_A.json (top 3) ---")
with open(fi_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    for item in data[:3]:
        print(f"  {item}")

print("\nTAREA 2 COMPLETADA.")
