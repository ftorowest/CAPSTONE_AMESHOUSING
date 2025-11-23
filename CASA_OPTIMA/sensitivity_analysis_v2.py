#!/usr/bin/env python3
"""
ANÁLISIS DE SENSIBILIDAD V2 - REMODELACIONES
Análisis completo para el código nuevo (remodelaciones únicamente)

Ejecuta 5 análisis principales:
1. ROI vs Precio Base (5 presupuestos diferentes)
2. Sensibilidad Presupuesto (múltiples casas)
3. Sensibilidad PWL (parámetro de aproximación)
4. Predicción vs Real (tracking en todas las optimizaciones)
5. Comparación XGBoost vs Linear (mismas casas, ambos modelos)
"""

import pandas as pd
from joblib import load
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from src.preprocessing import load_and_prepare
from src.optimization import optimize_house
from tqdm import tqdm

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Parámetros generales
BUDGET_LEVELS = [50000, 100000, 150000, 200000, 250000]  # 5 presupuestos (Análisis 1)
BUDGET_LEVELS_EXTENDED = [50000, 100000, 150000, 200000, 250000, 500000]  # 6 presupuestos (Análisis 2)
PWL_VALUES = [10, 15, 20, 25, 30, 40, 50]  # Valores PWL a probar
DEFAULT_PWL = 20  # PWL por defecto

# Número de casas a analizar
N_HOUSES_PRICE_ANALYSIS = 30  # Para análisis 1 (ROI vs Precio Base)
N_HOUSES_BUDGET_ANALYSIS = 10  # Para análisis 2 (Sensibilidad Presupuesto)
N_HOUSES_COMPARISON = 20  # Para análisis 5 (XGBoost vs Linear) - "considerable pero no enorme"

# Directorio de resultados
RESULTS_DIR = Path("sensitivity_results_v2")
RESULTS_DIR.mkdir(exist_ok=True)

# Semilla para reproducibilidad
random.seed(42)
np.random.seed(42)

print("="*80)
print("ANÁLISIS DE SENSIBILIDAD V2 - REMODELACIONES")
print("="*80)
print(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Resultados en: {RESULTS_DIR.absolute()}")
print("="*80)

# ============================================================================
# CARGAR DATOS Y MODELOS
# ============================================================================

print("\n📂 Cargando datos y modelos...")
X, y_log = load_and_prepare('data/ames_dum.csv')

# Cargar modelos (usar joblib.load en vez de pickle.load)
model_xgb = load('models/xgb_optuna_model.pkl')
print("   ✓ Modelo XGBoost cargado")

model_linear = load('models/linear_model.pkl')
print("   ✓ Modelo Linear cargado")

# Metadata de modelos (formato DataFrame como espera optimize_house)
trained_feats = X.columns.tolist()
trained_stats = pd.DataFrame({
    'mean': X.mean(),
    'std': X.std(),
    'min': X.min(),
    'max': X.max()
})

# Seleccionar casas aleatorias
n_houses = len(X)
print(f"\n📊 Dataset: {n_houses} casas disponibles")

# ============================================================================
# ESTRUCTURA DE DATOS GLOBAL PARA TRACKING
# ============================================================================

# DataFrame global para tracking predicción vs real
prediction_tracking = []

def run_optimization(model, model_name, house_idx, budget, pwl_k=DEFAULT_PWL):
    """
    Ejecuta optimización y registra predicción vs real
    
    Returns:
        dict: Resultados de optimización + tracking
    """
    baseline_house = X.iloc[house_idx].copy()
    baseline_price = np.exp(y_log.iloc[house_idx])
    
    result = optimize_house(
        model=model,
        X=X,
        y_log=y_log,
        trained_feats=trained_feats,
        trained_stats=trained_stats,
        baseline_idx=house_idx,
        budget=budget,
        pwl_k=pwl_k
    )
    
    # Tracking predicción vs real (si la optimización fue exitosa)
    if 'profit' in result:
        prediction_tracking.append({
            'model': model_name,
            'house_idx': house_idx,
            'budget': budget,
            'pwl_k': pwl_k,
            'real_price_before': baseline_price,
            'predicted_price_before': result.get('price_before', baseline_price),
            'predicted_price_after': result.get('price_after', baseline_price),
            'price_diff_before': abs(baseline_price - result.get('price_before', baseline_price)),
            'price_diff_pct_before': abs(baseline_price - result.get('price_before', baseline_price)) / baseline_price * 100
        })
    
    return result


# ============================================================================
# ANÁLISIS 1: ROI vs Precio Base (Multi-Presupuesto)
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS 1: ROI vs Precio Base (5 Presupuestos)")
print("="*80)

# Seleccionar 30 casas con diferentes rangos de precio
price_bins = pd.qcut(y_log, q=6, labels=False, duplicates='drop')
houses_analysis1 = []

for bin_idx in range(6):
    bin_houses = X[price_bins == bin_idx].index.tolist()
    if len(bin_houses) >= 5:
        houses_analysis1.extend(random.sample(bin_houses, min(5, len(bin_houses))))

houses_analysis1 = houses_analysis1[:N_HOUSES_PRICE_ANALYSIS]
print(f"   Casas seleccionadas: {len(houses_analysis1)}")
print(f"   Presupuestos: {BUDGET_LEVELS}")

results_price_multi = []

for budget in tqdm(BUDGET_LEVELS, desc="Presupuestos"):
    for house_idx in tqdm(houses_analysis1, desc=f"  Presupuesto ${budget:,}", leave=False):
        result = run_optimization(model_xgb, 'XGBoost', house_idx, budget)
        
        if 'profit' in result:
            results_price_multi.append({
                'budget': budget,
                'house_idx': house_idx,
                'baseline_price': result['price_before'],
                'optimized_price': result['price_after'],
                'spent': result['spent'],
                'profit': result['profit'],
                'roi': result['roi'],
                'price_increase_pct': (result['price_after'] - result['price_before']) / result['price_before'] * 100
            })

df_price_multi = pd.DataFrame(results_price_multi)
df_price_multi.to_csv(RESULTS_DIR / "analysis1_roi_vs_price_multibudget.csv", index=False)
print(f"\n✓ Análisis 1 completado: {len(df_price_multi)} optimizaciones")
print(f"   Guardado: analysis1_roi_vs_price_multibudget.csv")


# ============================================================================
# ANÁLISIS 2: Sensibilidad Presupuesto (Multi-Casa)
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS 2: Sensibilidad Presupuesto (Múltiples Casas)")
print("="*80)

# Seleccionar 10 casas representativas
houses_analysis2 = random.sample(range(n_houses), N_HOUSES_BUDGET_ANALYSIS)
print(f"   Casas seleccionadas: {len(houses_analysis2)}")
print(f"   Presupuestos: {BUDGET_LEVELS_EXTENDED}")

results_budget_multi = []

for house_idx in tqdm(houses_analysis2, desc="Casas"):
    for budget in tqdm(BUDGET_LEVELS_EXTENDED, desc=f"  Casa #{house_idx}", leave=False):
        result = run_optimization(model_xgb, 'XGBoost', house_idx, budget)
        
        if 'profit' in result:
            results_budget_multi.append({
                'house_idx': house_idx,
                'budget': budget,
                'price_before': result['price_before'],
                'price_after': result['price_after'],
                'spent': result['spent'],
                'profit': result['profit'],
                'roi': result['roi'],
                'budget_used_pct': result['spent'] / budget * 100
            })

df_budget_multi = pd.DataFrame(results_budget_multi)
df_budget_multi.to_csv(RESULTS_DIR / "analysis2_budget_sensitivity_multihouse.csv", index=False)
print(f"\n✓ Análisis 2 completado: {len(df_budget_multi)} optimizaciones")
print(f"   Guardado: analysis2_budget_sensitivity_multihouse.csv")


# ============================================================================
# ANÁLISIS 3: Sensibilidad PWL (Mantener como está)
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS 3: Sensibilidad PWL")
print("="*80)

# Una casa representativa, presupuesto fijo
house_pwl = random.choice(range(n_houses))
budget_pwl = 125000
print(f"   Casa: #{house_pwl}")
print(f"   Presupuesto: ${budget_pwl:,}")
print(f"   Valores PWL: {PWL_VALUES}")

results_pwl = []

for pwl_k in tqdm(PWL_VALUES, desc="Valores PWL"):
    result = run_optimization(model_xgb, 'XGBoost', house_pwl, budget_pwl, pwl_k)
    
    if 'profit' in result:
        results_pwl.append({
            'pwl_k': pwl_k,
            'price_before': result['price_before'],
            'optimized_price': result['price_after'],
            'spent': result['spent'],
            'budget': budget_pwl,
            'profit': result['profit'],
            'roi': result['roi']
        })

df_pwl = pd.DataFrame(results_pwl)
df_pwl.to_csv(RESULTS_DIR / "analysis3_pwl_sensitivity.csv", index=False)
print(f"\n✓ Análisis 3 completado: {len(df_pwl)} optimizaciones")
print(f"   Guardado: analysis3_pwl_sensitivity.csv")


# ============================================================================
# ANÁLISIS 4: Tracking Predicción vs Real
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS 4: Predicción vs Real (Recopilado de todos los análisis)")
print("="*80)

# Guardar tracking acumulado
df_tracking = pd.DataFrame(prediction_tracking)
df_tracking.to_csv(RESULTS_DIR / "analysis4_prediction_vs_real.csv", index=False)

print(f"✓ Análisis 4 completado: {len(df_tracking)} registros de tracking")
print(f"   Guardado: analysis4_prediction_vs_real.csv")

# Estadísticas
if len(df_tracking) > 0:
    print(f"\n📊 ESTADÍSTICAS DE PREDICCIÓN:")
    print(f"   Error promedio precio base: ${df_tracking['price_diff_before'].mean():,.2f}")
    print(f"   Error % promedio precio base: {df_tracking['price_diff_pct_before'].mean():.2f}%")
    print(f"   Error máximo: ${df_tracking['price_diff_before'].max():,.2f}")


# ============================================================================
# ANÁLISIS 5: Comparación XGBoost vs Linear
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS 5: XGBoost vs Linear (Mismas Casas)")
print("="*80)

# Seleccionar 5 casas representativas
houses_comparison = random.sample(range(n_houses), N_HOUSES_COMPARISON)
budget_comparison = 150000
print(f"   Casas seleccionadas: {houses_comparison}")
print(f"   Presupuesto: ${budget_comparison:,}")

results_comparison = []
improvement_differences = []  # Para comparar mejoras entre modelos (todas las casas)

for house_idx in tqdm(houses_comparison, desc="Casas"):
    # XGBoost
    result_xgb = run_optimization(model_xgb, 'XGBoost', house_idx, budget_comparison)
    
    # Linear
    result_linear = run_optimization(model_linear, 'Linear', house_idx, budget_comparison)
    
    # Guardar ambos resultados
    for model_name, result in [('XGBoost', result_xgb), ('Linear', result_linear)]:
        if 'profit' in result:
            comparison_entry = {
                'model': model_name,
                'house_idx': house_idx,
                'budget': budget_comparison,
                'price_before': result['price_before'],
                'price_after': result['price_after'],
                'spent': result['spent'],
                'profit': result['profit'],
                'roi': result['roi']
            }
            
            # Extraer cambios específicos (atributos modificados)
            if 'changes' in result:
                changes = result['changes']
                comparison_entry['n_changes'] = len(changes)
                comparison_entry['top_changes'] = ', '.join([f"{k}:{v:.2f}" for k, v in list(changes.items())[:5]])
            
            results_comparison.append(comparison_entry)
    
    # Comparar DIFERENCIAS en mejoras entre modelos (para cada casa)
    if 'changes' in result_xgb and 'changes' in result_linear:
        changes_xgb = result_xgb['changes']
        changes_linear = result_linear['changes']
        
        # Todos los atributos modificados por al menos un modelo
        all_attributes = set(changes_xgb.keys()) | set(changes_linear.keys())
        
        for attr in all_attributes:
            delta_xgb = changes_xgb.get(attr, 0)
            delta_linear = changes_linear.get(attr, 0)
            difference = delta_xgb - delta_linear
            
            # Guardar TODAS las diferencias (incluso pequeñas)
            improvement_differences.append({
                'house_idx': house_idx,
                'attribute': attr,
                'xgb_change': delta_xgb,
                'linear_change': delta_linear,
                'difference': difference,
                'abs_difference': abs(difference)
            })

df_comparison = pd.DataFrame(results_comparison)
df_comparison.to_csv(RESULTS_DIR / "analysis5_xgb_vs_linear.csv", index=False)
print(f"\n✓ Análisis 5 completado: {len(df_comparison)} optimizaciones")
print(f"   Guardado: analysis5_xgb_vs_linear.csv")

# Analizar PATRONES ACUMULADOS en mejoras (no casos extremos)
if improvement_differences:
    df_improvements = pd.DataFrame(improvement_differences)
    
    # CALCULAR ESTADÍSTICAS POR ATRIBUTO (PATRONES GENERALES)
    patterns = df_improvements.groupby('attribute').agg({
        'difference': ['sum', 'mean', 'count'],  # Diferencia acumulada, promedio, frecuencia
        'abs_difference': 'mean',  # Diferencia absoluta promedio
        'xgb_change': 'mean',  # Cambio promedio XGBoost
        'linear_change': 'mean'  # Cambio promedio Linear
    }).reset_index()
    
    # Aplanar nombres de columnas
    patterns.columns = ['attribute', 'diff_acumulada', 'diff_promedio', 'frecuencia', 
                        'abs_diff_promedio', 'xgb_promedio', 'linear_promedio']
    
    # Ordenar por diferencia acumulada ABSOLUTA (patrones más importantes)
    patterns['abs_diff_acumulada'] = patterns['diff_acumulada'].abs()
    patterns = patterns.sort_values('abs_diff_acumulada', ascending=False)
    
    # Guardar CSV con diferencias detalladas (todas las casas)
    df_improvements.to_csv(RESULTS_DIR / "analysis5_improvement_differences_detailed.csv", index=False)
    print(f"   Guardado: analysis5_improvement_differences_detailed.csv ({len(df_improvements)} registros)")
    
    # Guardar CSV con PATRONES ACUMULADOS
    patterns.to_csv(RESULTS_DIR / "analysis5_improvement_patterns.csv", index=False)
    print(f"   Guardado: analysis5_improvement_patterns.csv (patrones por atributo)")
    
    # Mostrar TOP 10 PATRONES (no casos extremos)
    print(f"\n   🎯 TOP 10 PATRONES EN MEJORAS (Diferencia Acumulada en {len(houses_comparison)} casas):")
    print(f"   {'Atributo':<25} {'Frec':<5} {'XGB Prom':<12} {'Linear Prom':<12} {'Diff Acum':<12} {'Diff Prom':<12}")
    print(f"   {'-'*80}")
    
    for _, row in patterns.head(10).iterrows():
        attr = row['attribute'][:23]  # Truncar nombre largo
        freq = int(row['frecuencia'])
        xgb_avg = row['xgb_promedio']
        lin_avg = row['linear_promedio']
        diff_acc = row['diff_acumulada']
        diff_avg = row['diff_promedio']
        
        print(f"   {attr:<25} {freq:<5} {xgb_avg:>10.2f}   {lin_avg:>10.2f}   {diff_acc:>10.2f}   {diff_avg:>10.2f}")

# Comparación estadística
if len(df_comparison) > 0:
    print(f"\n📊 COMPARACIÓN XGBoost vs Linear:")
    for model in ['XGBoost', 'Linear']:
        model_data = df_comparison[df_comparison['model'] == model]
        print(f"\n   {model}:")
        print(f"      ROI promedio: {model_data['roi'].mean()*100:.1f}%")
        print(f"      Ganancia promedio: ${model_data['profit'].mean():,.0f}")
        print(f"      Gasto promedio: ${model_data['spent'].mean():,.0f}")
        if 'n_changes' in model_data.columns:
            print(f"      Cambios promedio: {model_data['n_changes'].mean():.1f} atributos")


# ============================================================================
# REPORTE FINAL
# ============================================================================

print("\n" + "="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80)

summary = f"""
RESUMEN DE ANÁLISIS DE SENSIBILIDAD V2
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ANÁLISIS 1: ROI vs Precio Base (Multi-Presupuesto)
   • Casas analizadas: {len(houses_analysis1)}
   • Presupuestos: {len(BUDGET_LEVELS)}
   • Total optimizaciones: {len(df_price_multi)}
   • Archivo: analysis1_roi_vs_price_multibudget.csv

ANÁLISIS 2: Sensibilidad Presupuesto (Multi-Casa) - CON $500K ADICIONAL
   • Casas analizadas: {len(houses_analysis2)}
   • Presupuestos por casa: {len(BUDGET_LEVELS_EXTENDED)} (incluye $500k)
   • Total optimizaciones: {len(df_budget_multi)}
   • Archivo: analysis2_budget_sensitivity_multihouse.csv

ANÁLISIS 3: Sensibilidad PWL
   • Valores PWL probados: {len(PWL_VALUES)}
   • Total optimizaciones: {len(df_pwl)}
   • Archivo: analysis3_pwl_sensitivity.csv

ANÁLISIS 4: Predicción vs Real
   • Registros de tracking: {len(df_tracking)}
   • Error promedio: ${df_tracking['price_diff_before'].mean() if len(df_tracking) > 0 else 0:,.2f}
   • Archivo: analysis4_prediction_vs_real.csv

ANÁLISIS 5: XGBoost vs Linear - CON PATRONES ACUMULADOS DE MEJORAS
   • Casas comparadas: {len(houses_comparison)}
   • Modelos: 2 (XGBoost, Linear)
   • Total optimizaciones: {len(df_comparison)}
   • Archivos: 
      - analysis5_xgb_vs_linear.csv (resultados ROI/ganancia)
      - analysis5_improvement_differences_detailed.csv (todas las diferencias)
      - analysis5_improvement_patterns.csv (patrones acumulados por atributo)

ARCHIVOS CSV GENERADOS:
   1. analysis1_roi_vs_price_multibudget.csv
   2. analysis2_budget_sensitivity_multihouse.csv
   3. analysis3_pwl_sensitivity.csv
   4. analysis4_prediction_vs_real.csv
   5. analysis5_xgb_vs_linear.csv
   6. analysis5_improvement_differences_detailed.csv (diferencias por casa)
   7. analysis5_improvement_patterns.csv (PATRONES ACUMULADOS) ← NUEVO
   8. summary_report.txt (este archivo)

TOTAL DE OPTIMIZACIONES: {len(df_price_multi) + len(df_budget_multi) + len(df_pwl) + len(df_comparison)}
"""

with open(RESULTS_DIR / "summary_report.txt", "w") as f:
    f.write(summary)

print(summary)
print(f"\n📁 Todos los resultados guardados en: {RESULTS_DIR.absolute()}")
print("="*80)
