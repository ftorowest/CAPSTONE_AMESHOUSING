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

    return shap_values
