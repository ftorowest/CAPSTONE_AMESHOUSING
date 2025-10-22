"""
Análisis de ROI para diferentes presupuestos en la casa 667
===========================================================

Este script analiza el retorno de inversión (ROI) para diferentes presupuestos
de remodelación en la casa 667 del dataset Ames Housing.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
                print("Optimización fallida - Casa no factible")
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


def plot_roi_analysis(results_df):
    """
    Crea gráficos para visualizar el análisis de ROI

    Args:
        results_df: DataFrame con los resultados del análisis
    """
    # Configurar estilo de seaborn
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # Filtrar resultados válidos
    valid_results = results_df.dropna()

    if len(valid_results) == 0:
        print("No hay resultados válidos para graficar")
        return

    # Crear figura con subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. ROI vs Presupuesto
    ax1.plot(valid_results['presupuesto'] / 1000, valid_results['roi'],
             marker='o', linewidth=2, markersize=8, color='blue')
    ax1.set_title('ROI vs Presupuesto', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Presupuesto (miles de $)')
    ax1.set_ylabel('ROI (%)')
    ax1.grid(True, alpha=0.3)

    # Marcar el máximo ROI
    max_roi_idx = valid_results['roi'].idxmax()
    max_roi_budget = valid_results.loc[max_roi_idx, 'presupuesto'] / 1000
    max_roi_value = valid_results.loc[max_roi_idx, 'roi']
    ax1.plot(max_roi_budget, max_roi_value, 'ro', markersize=12, label=f'Máx: {max_roi_value:.1f}%')
    ax1.legend()

    # 2. Beneficio vs Presupuesto
    ax2.plot(valid_results['presupuesto'] / 1000, valid_results['beneficio'] / 1000,
             marker='s', linewidth=2, markersize=8, color='green')
    ax2.set_title('Beneficio vs Presupuesto', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Presupuesto (miles de $)')
    ax2.set_ylabel('Beneficio (miles de $)')
    ax2.grid(True, alpha=0.3)

    # Marcar el máximo beneficio
    max_profit_idx = valid_results['beneficio'].idxmax()
    max_profit_budget = valid_results.loc[max_profit_idx, 'presupuesto'] / 1000
    max_profit_value = valid_results.loc[max_profit_idx, 'beneficio'] / 1000
    ax2.plot(max_profit_budget, max_profit_value, 'ro', markersize=12,
             label=f'Máx: ${max_profit_value*1000:,.0f}')
    ax2.legend()

    # 3. Precio Optimizado vs Presupuesto
    ax3.plot(valid_results['presupuesto'] / 1000, valid_results['precio_optimizado'] / 1000,
             marker='^', linewidth=2, markersize=8, color='orange')
    ax3.axhline(y=valid_results['precio_original'].iloc[0] / 1000, color='red', linestyle='--',
                label=f'Precio Original: ${valid_results["precio_original"].iloc[0]:,.0f}')
    ax3.set_title('Precio Optimizado vs Presupuesto', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Presupuesto (miles de $)')
    ax3.set_ylabel('Precio Optimizado (miles de $)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. ROI y Beneficio combinados
    ax4_twin = ax4.twinx()

    # ROI en el eje izquierdo
    line1 = ax4.plot(valid_results['presupuesto'] / 1000, valid_results['roi'],
                     marker='o', linewidth=2, markersize=6, color='blue', label='ROI (%)')
    ax4.set_xlabel('Presupuesto (miles de $)')
    ax4.set_ylabel('ROI (%)', color='blue')
    ax4.tick_params(axis='y', labelcolor='blue')

    # Beneficio en el eje derecho
    line2 = ax4_twin.plot(valid_results['presupuesto'] / 1000, valid_results['beneficio'] / 1000,
                          marker='s', linewidth=2, markersize=6, color='green', label='Beneficio (miles $)')
    ax4_twin.set_ylabel('Beneficio (miles $)', color='green')
    ax4_twin.tick_params(axis='y', labelcolor='green')

    ax4.set_title('ROI y Beneficio vs Presupuesto', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Combinar leyendas
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig('../models/roi_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Imprimir recomendaciones
    print(f"\n{'='*80}")
    print("RECOMENDACIONES DE INVERSIÓN")
    print(f"{'='*80}")

    max_roi_budget = valid_results.loc[valid_results['roi'].idxmax(), 'presupuesto']
    max_roi_value = valid_results['roi'].max()
    max_profit_budget = valid_results.loc[valid_results['beneficio'].idxmax(), 'presupuesto']
    max_profit_value = valid_results['beneficio'].max()

    print(f"• Mejor ROI: {max_roi_value:.2f}% con presupuesto de ${max_roi_budget:,.0f}")
    print(f"• Mejor beneficio absoluto: ${max_profit_value:,.0f} con presupuesto de ${max_profit_budget:,.0f}")

    # Analizar eficiencia
    roi_per_dollar = valid_results['roi'] / (valid_results['presupuesto'] / 1000)
    best_efficiency_idx = roi_per_dollar.idxmax()
    best_efficiency_budget = valid_results.loc[best_efficiency_idx, 'presupuesto']
    best_efficiency_roi = valid_results.loc[best_efficiency_idx, 'roi']

    print(f"• Mejor eficiencia (ROI por $1000 invertido): {roi_per_dollar.max():.3f}% con presupuesto de ${best_efficiency_budget:,.0f}")

    # Recomendación general
    if max_roi_budget == max_profit_budget:
        print(f"\n✓ RECOMENDACIÓN: Invierta ${max_roi_budget:,.0f} para maximizar tanto ROI como beneficio absoluto.")
    else:
        print(f"\n✓ RECOMENDACIÓN: Si busca máximo ROI, invierta ${max_roi_budget:,.0f}.")
        print(f"✓ Si busca máximo beneficio absoluto, invierta ${max_profit_budget:,.0f}.")


def main():
    """Función principal"""
    # Analizar ROI para la casa 667 con 10 presupuestos
    results = analyze_roi_for_house(house_idx=667)

    # Crear gráficos
    plot_roi_analysis(results)

    # Guardar resultados
    output_file = "../models/roi_analysis_casa_667.csv"
    results.to_csv(output_file, index=False)
    print(f"\n✓ Resultados guardados en: {output_file}")
    print("✓ Gráficos guardados en: ../models/roi_analysis_plots.png")


if _name_ == "_main_":
    main()