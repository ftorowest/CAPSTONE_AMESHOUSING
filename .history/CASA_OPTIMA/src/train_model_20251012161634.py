"""
train_model.py
---------------
Entrena modelos de regresión lineal y XGBoost (Optuna)
para el proyecto Casa Óptima. Evalúa con RMSE, MAE y R2.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from xgboost import XGBRegressor
import optuna
from joblib import dump


# Función de RMSE compatible 
def rmse_compat(y_true, y_pred):
    # Devuelve RMSE compatible con distintas versiones de sklearn.
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Cross-validation para cualquier modelo 
def cv_eval(estimator, X, y_log, folds=5, seed=42):
    # Evalúa un modelo con validación cruzada y devuelve métricas promedio.
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    y_true, y_pred = [], []
    for tr, te in kf.split(X):
        Xt, Xv = X.iloc[tr], X.iloc[te]
        yt, yv = y_log.iloc[tr], y_log.iloc[te]
        m = clone(estimator)
        m.fit(Xt, yt)
        pv = np.expm1(m.predict(Xv))  # vuelve a escala original
        y_true.append(np.expm1(yv))
        y_pred.append(pv)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return {
        "RMSE": rmse_compat(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


# ===== Optuna para XGBoost =====
def tune_xgb_with_optuna(X, y, n_trials=50):
    """Busca los mejores hiperparámetros de XGBRegressor con Optuna."""
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "n_jobs": -1,
            "tree_method": "hist",
            "random_state": 42
        }

        model = XGBRegressor(**params)
        pipe = Pipeline([("scaler", "passthrough"), ("est", model)])
        metrics = cv_eval(pipe, X, y)
        return metrics["RMSE"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n==== Mejores hiperparámetros Optuna ====")
    print(study.best_params)
    print(f"RMSE CV óptimo: {study.best_value:.4f}")

    return study.best_params


# ===== Entrenar ambos modelos =====
def train_models(X, y, save_dir="models"):
    """Entrena Linear y XGB (Optuna) y guarda los modelos."""
    from pathlib import Path
    Path(save_dir).mkdir(exist_ok=True)

    results = []
    fitted = {}

    # --- Modelo lineal ---
    lin_model = Pipeline([
        ("scaler", StandardScaler()),
        ("est", LinearRegression())
    ])
    metrics_lin = cv_eval(lin_model, X, y)
    lin_model.fit(X, y)
    dump(lin_model, f"{save_dir}/linear_model.pkl")
    results.append({"Model": "Linear", **metrics_lin})
    fitted["Linear"] = lin_model

    # --- Modelo XGBoost (Optuna) ---
    print("\n🔍 Optimizando hiperparámetros XGBoost con Optuna...")
    best_params = tune_xgb_with_optuna(X, y, n_trials=30)
    xgb_model = XGBRegressor(**best_params)
    xgb_model.fit(X, y)
    dump(xgb_model, f"{save_dir}/xgb_optuna_model.pkl")

    metrics_xgb = cv_eval(Pipeline([("scaler", "passthrough"), ("est", xgb_model)]), X, y)
    results.append({"Model": "XGB_Optuna", **metrics_xgb})
    fitted["XGB_Optuna"] = xgb_model

    # --- Comparar resultados ---
    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print("\n=== Resultados comparativos ===")
    print(results_df)

    best_name = results_df.iloc[0]["Model"]
    print(f"\n Mejor modelo por RMSE: {best_name}")

    return fitted, results_df
