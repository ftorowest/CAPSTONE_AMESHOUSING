"""
Análisis de ROI para múltiples casas factibles con diferentes presupuestos
===========================================================================

Este script selecciona 10 casas factibles al azar del dataset Ames Housing
y calcula el ROI para presupuestos que van desde $10,000 hasta $100,000
con incrementos de $5,000.
"""

import os
import numpy as np
import pandas as pd
from joblib import load
import random

from preprocessing import load_and_prepare
from optimization import optimize_house
from check_feasible_houses import check_house_feasibility


def analyze_multiple_houses_roi(num_houses=10, budgets=None):
    """
    Analiza el ROI para múltiples casas factibles con diferentes presupuestos.

    Parameters:
    num_houses : int
        Número de casas factibles a seleccionar al azar
    budgets : list
        Lista de presupuestos a probar (en USD)

    Returns:
    pd.DataFrame: Resultados del análisis de ROI
    """
    if budgets is None:
        # Presupuestos desde $10K hasta $100K con saltos de $5K
        budgets = list(range(10_000, 105_000, 5_000))

    print(f"\n{'='*80}")
    print(f"ANÁLISIS DE ROI - {num_houses} CASAS FACTIBLES ALEATORIAS")
    print(f"Presupuestos: {len(budgets)} (desde ${budgets[0]:,} hasta ${budgets[-1]:,})")
    print(f"{'='*80}")

    # Cargar datos y modelo
    print("\nCargando datos y modelo...")
    DATA_PATH = "CASA_OPTIMA/data/ames_dum.csv"
    MODEL_PATH = "CASA_OPTIMA/models/xgb_optuna_model.pkl"

    X, y_log = load_and_prepare(DATA_PATH)
    trained_feats = X.columns.tolist()

    # Cargar modelo entrenado
    model = load(MODEL_PATH)

    # Calcular estadísticas para la optimización
    trained_stats = pd.DataFrame({
        "q05": X.quantile(0.05),
        "median": X.median(),
        "q95": X.quantile(0.95),
        "max": X.max()
    })

    # Encontrar casas factibles
    print("\nIdentificando casas factibles...")
    feasible_houses = []
    for idx in range(len(X)):
        house = X.iloc[idx]
        is_feasible, _ = check_house_feasibility(house)
        if is_feasible:
            feasible_houses.append(idx)

    print(f"Encontradas {len(feasible_houses)} casas factibles de {len(X)} totales")

    if len(feasible_houses) < num_houses:
        print(f"Advertencia: Solo hay {len(feasible_houses)} casas factibles. Usando todas.")
        num_houses = len(feasible_houses)

    # Seleccionar casas al azar
    selected_houses = random.sample(feasible_houses, num_houses)
    print(f"Casas seleccionadas: {selected_houses}")

    # Resultados
    results = []

    # Para cada casa seleccionada
    for house_idx in selected_houses:
        print(f"\n--- CASA {house_idx} ---")
        house = X.iloc[house_idx]
        price_original = np.exp(y_log.iloc[house_idx])  # Precio real

        # Para cada presupuesto
        for budget in budgets:
            print(f"  Presupuesto: ${budget:,.0f}", end="")

            try:
                # Ejecutar optimización
                result = optimize_house(
                    model=model,
                    X=X,
                    y_log=y_log,
                    trained_feats=trained_feats,
                    trained_stats=trained_stats,
                    baseline_idx=house_idx,
                    budget=budget,
                    pwl_k=25
                )

                if result is None:
                    print(" - No factible")
                    roi = None
                    profit = None
                    optimized_price = None
                    gasto = None
                else:
                    # Extraer resultados
                    optimized_price = result["price_after"]
                    profit = result["profit"]
                    roi = result["roi"]
                    gasto = result["total_cost"]

                    print(".1f")

                # Guardar resultado
                results.append({
                    "casa_id": house_idx,
                    "presupuesto": budget,
                    "precio_original": price_original,
                    "precio_optimizado": optimized_price,
                    "gasto_total": gasto,
                    "ganancia_neta": profit,
                    "roi": roi
                })

            except Exception as e:
                print(f" - Error: {str(e)}")
                results.append({
                    "casa_id": house_idx,
                    "presupuesto": budget,
                    "precio_original": price_original,
                    "precio_optimizado": None,
                    "gasto_total": None,
                    "ganancia_neta": None,
                    "roi": None
                })

    # Convertir a DataFrame
    results_df = pd.DataFrame(results)

    # Resumen
    valid_results = results_df.dropna(subset=["roi"])
    if len(valid_results) > 0:
        print("\nRESUMEN:")
        print(f"Total combinaciones analizadas: {len(results_df)}")
        print(f"Optimizaciones exitosas: {len(valid_results)}")
        print(".2f")
        print(".2f")

    return results_df


def main():
    """Función principal"""
    # Analizar ROI para 10 casas factibles aleatorias
    results = analyze_multiple_houses_roi(num_houses=10)

    # Guardar resultados
    output_file = "CASA_OPTIMA/models/multi_house_roi_analysis.csv"
    results.to_csv(output_file, index=False)
    print(f"\n✓ Resultados guardados en: {output_file}")
    print(f"✓ Total filas: {len(results)}")


if _name_ == "_main_":
    main()