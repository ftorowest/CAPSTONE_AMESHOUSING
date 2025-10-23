"""
Batch Optimization Script
Selecciona 10 casas aleatorias (con semilla fija) y 10 presupuestos,
optimiza cada combinación y guarda los resultados en CSV.
"""
import json
import numpy as np
import pandas as pd
from joblib import load
from preprocessing import load_and_prepare
from optimization import optimize_house

# Configuración
SEED = 42
N_HOUSES = 10
N_BUDGETS = 10
MIN_BUDGET = 10_000
MAX_BUDGET = 100_000

DATA_PATH = "data/ames_dum.csv"
MODEL_PATH = "models/xgb_optuna_model.pkl"
OUTPUT_CSV = "models/batch_optimization_results.csv"

print("=" * 70)
print("BATCH OPTIMIZATION - 10 CASAS x 10 PRESUPUESTOS")
print("=" * 70)

# Cargar datos y modelo
print("\n[1/4] Cargando datos y modelo...")
X, y = load_and_prepare(DATA_PATH)
model = load(MODEL_PATH)

trained_feats = X.columns.tolist()
trained_stats = pd.DataFrame({
    "q05": X.quantile(0.05),
    "median": X.median(),
    "q95": X.quantile(0.95),
    "max": X.max()
})

# Seleccionar 10 casas aleatorias con semilla fija
print(f"\n[2/4] Seleccionando {N_HOUSES} casas aleatorias (seed={SEED})...")
np.random.seed(SEED)
selected_houses = np.random.choice(len(X), size=N_HOUSES, replace=False)
print(f"Casas seleccionadas: {selected_houses.tolist()}")

# Generar 10 presupuestos lineales
budgets = np.linspace(MIN_BUDGET, MAX_BUDGET, N_BUDGETS)
print(f"\n[3/4] Presupuestos: {[f'${b:,.0f}' for b in budgets]}")

# Optimizar cada combinación casa x presupuesto
print(f"\n[4/4] Optimizando {N_HOUSES * N_BUDGETS} combinaciones...")
print("-" * 70)

results = []
total_combinations = N_HOUSES * N_BUDGETS
current = 0

for house_idx in selected_houses:
    for budget in budgets:
        current += 1
        print(f"\n[{current}/{total_combinations}] Casa {house_idx} | "
              f"Presupuesto ${budget:,.0f}")
        
        try:
            result = optimize_house(
                model=model,
                X=X,
                y_log=y,
                trained_feats=trained_feats,
                trained_stats=trained_stats,
                baseline_idx=int(house_idx),
                budget=float(budget)
            )
            
            if result is not None:
                # Extraer cambios (mejoras) como JSON
                changes_dict = result.get("changes", {})
                cost_breakdown_dict = result.get("cost_breakdown", {})
                
                row = {
                    "house_idx": int(house_idx),
                    "budget": float(budget),
                    "price_before": float(result["price_before"]),
                    "price_after": float(result["price_after"]),
                    "spent": float(result["spent"]),
                    "profit": float(result["profit"]),
                    "roi": float(result["roi"]),
                    "changes": json.dumps(changes_dict),
                    "cost_breakdown": json.dumps(cost_breakdown_dict)
                }
                results.append(row)
                print(f"✓ ROI: {result['roi']:.2f} | "
                      f"Gasto: ${result['spent']:,.0f}")
            else:
                print("✗ No se encontró solución factible")
                
        except Exception as e:
            print(f"✗ Error: {str(e)}")

# Guardar resultados en CSV
print("\n" + "=" * 70)
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)
print(f"✓ Resultados guardados en: {OUTPUT_CSV}")
print(f"✓ Total de optimizaciones exitosas: {len(results)}/{total_combinations}")
print("="*70)