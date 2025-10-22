"""
Análisis de ROI para diferentes presupuestos en la casa 667
===========================================================

Este script analiza el retorno de inversión (ROI) para diferentes presupuestos
de remodelación en la casa 667 del dataset Ames Housing.
"""

import os
import numpy as np
import pandas as pd
from joblib import load

from preprocessing import load_and_prepare
from optimization import optimize_house


def analyze_roi_for_house(house_idx=667, budgets=None):
    """
    Analiza el ROI para diferentes presupuestos en una casa específica.

    Parameters:
    house_idx : int
        Índice de la casa a analizar
    budgets : list
        Lista de presupuestos a probar (en USD)

    Returns:
    pd.DataFrame: Resultados del análisis de ROI
    """
    if budgets is None:
        # 10 presupuestos diferentes: desde $50K hasta $500K
        budgets = [50_000, 75_000, 100_000, 125_000, 150_000,
                  200_000, 250_000, 300_000, 400_000, 500_000]

    print(f"\n{'='*80}")
    print(f"ANÁLISIS DE ROI - CASA {house_idx}")
    print(f"{'='*80}")

    # Cargar datos y modelo
    print("\nCargando datos y modelo...")
    DATA_PATH = "../data/ames_dum.csv"
    MODEL_PATH = "../models/xgb_optuna_model.pkl"

    X, y_log = load_and_prepare(DATA_PATH)
    trained_feats = X.columns.tolist()

    # Cargar modelo entrenado
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")

    model = load(MODEL_PATH)
    print("✓ Modelo cargado exitosamente")

    # Estadísticas de las features (necesarias para optimization.py)
    trained_stats = pd.DataFrame({
        'q05': X.quantile(0.05),
        'median': X.median(),
        'q95': X.quantile(0.95),
        'max': X.max()
    })

    # Precio original de la casa
    baseline = X.iloc[house_idx]
    pred_log = float(model.predict(baseline.to_frame().T)[0])
    price_original = float(np.expm1(pred_log))

    print(f"\nCasa {house_idx}:")
    print(f"  Precio original predicho: ${price_original:,.0f}")

    results = []

    for budget in budgets:
        print(f"\n--- Presupuesto: ${budget:,.0f} ---")

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
                print(f "Optimización fallida - Casa no factible")
                roi = None
                profit = None
                optimized_price = None
            else:
                # Extraer resultados del diccionario
                optimized_price = result["price_after"]
                profit = result["profit"]
                roi = result["roi"]

                print(f"  Precio optimizado: ${optimized_price:,.0f}")
                print(f"  Beneficio: ${profit:,.0f}")
                print(f"  ROI: {roi:.2f}%")

        except Exception as e:
            print(f"Error en optimización: {str(e)}")
            optimized_price = None
            profit = None
            roi = None

        results.append({
            'presupuesto': budget,
            'precio_original': price_original,
            'precio_optimizado': optimized_price,
            'beneficio': profit,
            'roi': roi
        })

    # Crear DataFrame con resultados
    df_results = pd.DataFrame(results)

    # Mostrar resumen
    print(f"\n{'='*80}")
    print("RESUMEN DE RESULTADOS")
    print(f"{'='*80}")

    # Filtrar resultados válidos
    valid_results = df_results.dropna()

    if len(valid_results) > 0:
        print(f"\nMejor ROI: {valid_results['roi'].max():.2f}% "
              f"(presupuesto: ${valid_results.loc[valid_results['roi'].idxmax(), 'presupuesto']:,.0f})")

        print(f"Mejor beneficio absoluto: ${valid_results['beneficio'].max():,.0f} "
              f"(presupuesto: ${valid_results.loc[valid_results['beneficio'].idxmax(), 'presupuesto']:,.0f})")

        # Mostrar tabla completa
        print(f"\n{'Presupuesto':<12} {'Precio Opt':<12} {'Beneficio':<12} {'ROI':<8}")
        print("-" * 50)
        for _, row in valid_results.iterrows():
            print(f"${row['presupuesto']:,<10.0f} ${row['precio_optimizado']:,<10.0f} "
                  f"${row['beneficio']:,<10.0f} {row['roi']:>6.1f}%")
    else:
        print("No se pudieron calcular resultados válidos")

    return df_results


def main():
    """Función principal"""
    # Analizar ROI para la casa 667 con 10 presupuestos
    results = analyze_roi_for_house(house_idx=667)

    # Guardar resultados
    output_file = "../models/roi_analysis_casa_667.csv"
    results.to_csv(output_file, index=False)
    print(f"\n✓ Resultados guardados en: {output_file}")


if __name__ == "__main__":
    main()