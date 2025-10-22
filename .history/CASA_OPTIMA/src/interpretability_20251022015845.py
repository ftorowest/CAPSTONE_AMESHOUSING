"""
interpretability.py
-------------------
Analiza la interpretabilidad de los modelos usando SHAP.
Genera gráficos globales y (opcionalmente) exporta valores.
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def explain_model(model, X, model_name="XGB_Optuna", save_csv=False):
    """
    Parámetros
    ----------
    model : modelo entrenado (XGBRegressor o Pipeline)
    X : pd.DataFrame
        Datos usados para calcular los valores SHAP.
    model_name : str
        Nombre del modelo para etiquetas/gráficos.
    save_csv : bool
        Si True, guarda los valores SHAP en 'models/shap_values.csv'.
    """

    print(f"\nCalculando valores SHAP para {model_name}...")
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Gráfico beeswarm (impacto global) 
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"Impacto global de variables ({model_name})")
    plt.tight_layout()
    plt.savefig(f"models/{model_name}_shap_summary.png", dpi=300)
    plt.close()

    # Gráfico de barras (importancia promedio) 
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=len(X.columns))  # muestra todas las variables
    plt.title(f"Importancia promedio ({model_name})")
    plt.tight_layout()
    plt.savefig(f"models/{model_name}_shap_bar.png", dpi=300)
    plt.close()

    print(f"Gráficos SHAP guardados en carpeta 'models/'")

    if save_csv:
        shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
        shap_df.to_csv("models/shap_values.csv", index=False)
        print("SHAP values guardados en models/shap_values.csv")

    # Calcular la importancia promedio absoluta (igual que el gráfico de barras)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "Variable": X.columns,
        "MeanAbsSHAP": mean_abs_shap
    }).sort_values("MeanAbsSHAP", ascending=False)

    print("\nImportancia promedio de cada variable (SHAP):")
    print(shap_importance.head(10))  # muestra las 10 más influyentes

    # Guardar en CSV
    shap_importance.to_csv("models/shap_importance_summary.csv", index=False)

    return shap_values
