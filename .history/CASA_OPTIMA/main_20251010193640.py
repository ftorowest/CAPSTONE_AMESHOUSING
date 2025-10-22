import os
import numpy as np
import pandas as pd
from joblib import load

from src.preprocessing import load_and_prepare
from src.train_model import train_models
from src.interpretability import explain_model
from src.optimization import optimize_house


def main():
    # Configuración 
    DATA_PATH = "data/ames_dum.csv"
    SAVE_DIR = "models"
    MODEL_TO_EXPLAIN = "XGB_Optuna"   # "Linear" o "XGB_Optuna"
    os.makedirs(SAVE_DIR, exist_ok=True)

    #  Cargar y preparar datos 
    print("\n=== Cargando y preparando dataset ===")
    X, y = load_and_prepare(DATA_PATH)

    # ====== 2️⃣ Entrenamiento (solo si no existen modelos guardados) ======
    linear_path = os.path.join(SAVE_DIR, "linear_model.pkl")
    xgb_path = os.path.join(SAVE_DIR, "xgb_optuna_model.pkl")

    if os.path.exists(linear_path) and os.path.exists(xgb_path):
        print("\n✅ Modelos ya entrenados encontrados. Cargando desde disco...")
        fitted = {
            "Linear": load(linear_path),
            "XGB_Optuna": load(xgb_path)
        }
        results = pd.DataFrame([
            {"Model": "Linear", "Status": "Loaded"},
            {"Model": "XGB_Optuna", "Status": "Loaded"}
        ])
    else:
        print("\n⚙️ No se encontraron modelos guardados. Entrenando desde cero (esto puede tardar varios minutos)...")
        fitted, results = train_models(X, y, save_dir=SAVE_DIR)

    print("\n=== 3️⃣ Resultados de modelos ===")
    print(results)

    # ====== 3️⃣ Interpretabilidad SHAP ======
    print("\n=== 4️⃣ Interpretabilidad SHAP ===")

    model_to_explain = fitted[MODEL_TO_EXPLAIN]

    # Archivos esperados
    summary_path = os.path.join(SAVE_DIR, f"{MODEL_TO_EXPLAIN}_shap_summary.png")
    bar_path     = os.path.join(SAVE_DIR, f"{MODEL_TO_EXPLAIN}_shap_bar.png")
    csv_path     = os.path.join(SAVE_DIR, f"{MODEL_TO_EXPLAIN}_shap_values.csv")

    # Solo correr SHAP si no existen los outputs
    if not (os.path.exists(summary_path) and os.path.exists(bar_path)):
        print(f"⚙️ Generando interpretabilidad SHAP para {MODEL_TO_EXPLAIN}...")
        _ = explain_model(
            model_to_explain,
            X,
            model_name=MODEL_TO_EXPLAIN,
            save_csv=True
        )
        print(f"✅ SHAP completado y guardado en '{SAVE_DIR}'")
    else:
        print(f"🟢 Resultados SHAP ya existen en '{SAVE_DIR}', se omite cálculo.")
    
    # ====== 4️⃣ Optimización con Gurobi ======
    print("\n=== 5️⃣ Optimizando la Casa Óptima ===")
    result = optimize_house(
        model=fitted["XGB_Optuna"],
        X=X,
        y_log=y,
        trained_feats=X.columns.tolist(),
        trained_stats=pd.DataFrame({
            "q05": X.quantile(0.05),
            "median": X.median(),
            "q95": X.quantile(0.95),
            "max": X.max()
        }),
        baseline_idx=1,
        budget=200000
)
    baseline_X = X.iloc[1]

    print("Baseline features:\n")
    #ordenamos alfabeticamente para mejor lectura
    print(f"Baseline: {baseline_X.sort_index().to_dict()}")


# ====== Punto de entrada ======
if __name__ == "__main__":
    main()
