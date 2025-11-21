import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from preprocessing import load_and_prepare
from joblib import load
from optimization import optimize_house

# ==============================================================
# CONFIG
# ==============================================================

PLOT_DIR = "results/plots/construction"
RESULT_CSV = "results/construcciones_optimas.csv"
os.makedirs(PLOT_DIR, exist_ok=True)

# ==============================================================
# 1. CARGAR CSV PRINCIPAL 
# ==============================================================

df = pd.read_csv("results/construction_results.csv")
df["roi"] = pd.to_numeric(df["roi"], errors="coerce")

print(f"\nDataset original: {len(df)} filas")


# ==============================================================
# 2. GRAFICO 0 — Mejor LOT AREA por ubicación *y por presupuesto*
# ==============================================================

tmp = (
    df.groupby(["budget", "latitude", "longitude", "lot_area"])["roi"]
      .mean()
      .reset_index()
)

# elegimos el mejor lot_area por (budget, lat, lon)
best_lot_by_loc_budget = (
    tmp.sort_values(["budget", "latitude", "longitude", "roi"],
                    ascending=[True, True, True, False])
       .groupby(["budget", "latitude", "longitude"], as_index=False)
       .first()
)

# texto formato (lat,lon)
best_lot_by_loc_budget["loc_str"] = best_lot_by_loc_budget.apply(
    lambda r: f"({r['latitude']:.3f}, {r['longitude']:.3f})",
    axis=1
)

pivot_lot = best_lot_by_loc_budget.pivot_table(
    index="loc_str",
    columns="budget",
    values="lot_area"
)

plt.figure(figsize=(10,6))
sns.heatmap(pivot_lot, annot=True, fmt=".0f", cmap="viridis")
plt.title("Mejor tamaño de lote por ubicación y presupuesto\n(según ROI máximo)")
plt.xlabel("Presupuesto")
plt.ylabel("Ubicación")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/best_lot_area_per_location_by_budget.png", dpi=300)
plt.close()


print("\n✔ Gráfico 'best_lot_area_per_location_by_budget.png' creado.")

# ==============================================================
# 2B. GRAFICO — Mejor LOT AREA por ubicación *y por presupuesto* (según PROFIT)
# ==============================================================

tmp_profit = (
    df.groupby(["budget", "latitude", "longitude", "lot_area"])["profit"]
      .mean()
      .reset_index()
)

best_lot_by_loc_budget_profit = (
    tmp_profit.sort_values(
        ["budget", "latitude", "longitude", "profit"],
        ascending=[True, True, True, False]
    )
    .groupby(["budget", "latitude", "longitude"], as_index=False)
    .first()
)

# etiqueta de ubicación
best_lot_by_loc_budget_profit["loc_str"] = best_lot_by_loc_budget_profit.apply(
    lambda r: f"({r['latitude']:.3f}, {r['longitude']:.3f})",
    axis=1
)

pivot_lot_profit = best_lot_by_loc_budget_profit.pivot_table(
    index="loc_str",
    columns="budget",
    values="lot_area"
)

plt.figure(figsize=(10,6))
sns.heatmap(pivot_lot_profit, annot=True, fmt=".0f", cmap="plasma")
plt.title("Mejor tamaño de lote por ubicación y presupuesto\n(según PROFIT máximo)")
plt.xlabel("Presupuesto")
plt.ylabel("Ubicación")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/best_lot_area_per_location_by_budget_profit.png", dpi=300)
plt.close()


print("\n✔ Gráfico 'best_lot_area_per_location_by_budget_profit.png' creado.")


# ==============================================================
# 3. Cargar dataset real + modelo solo UNA vez
# ==============================================================

print("\nCargando dataset y modelo...")

DATA_PATH = "data/ames_dum.csv"
MODEL_PATH = "models/xgb_optuna_model.pkl"

X, y_log = load_and_prepare(DATA_PATH)
model = load(MODEL_PATH)

trained_feats = X.columns.tolist()
trained_stats = pd.DataFrame({
    "q05":    X.quantile(0.05),
    "median": X.median(),
    "q95":    X.quantile(0.95),
    "max":    X.max()
})

extra_budget = 1_000_000


# ==============================================================
# 4. Filtrar SOLO el mejor lot_area por ubicación
# ==============================================================

best_lot_by_loc = (
    df.groupby(["latitude", "longitude", "lot_area"])["roi"]
      .mean()
      .reset_index()
      .sort_values(["latitude", "longitude", "roi"],
                    ascending=[True, True, False])
      .groupby(["latitude", "longitude"], as_index=False)
      .first()
)

df_f = df.merge(
    best_lot_by_loc[["latitude", "longitude", "lot_area"]],
    on=["latitude", "longitude", "lot_area"],
    how="inner"
)

print(f"\nDataset filtrado para análisis: {len(df_f)} filas")


# ==============================================================
# 5. ROI vs Budget
# ==============================================================

plt.figure(figsize=(10,6))
for (lat, lon), sub in df_f.groupby(["latitude", "longitude"]):
    plt.plot(sub["budget"], sub["roi"], "-o", label=f"({lat:.3f},{lon:.3f})")

plt.xlabel("Presupuesto")
plt.ylabel("ROI")
plt.title("ROI vs Presupuesto (mejor lot area por ubicación)")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/roi_vs_budget_bestLot.png", dpi=300)
plt.close()



# ==============================================================
# 6. Profit vs Budget
# ==============================================================

plt.figure(figsize=(10,6))
for (lat, lon), sub in df_f.groupby(["latitude", "longitude"]):
    plt.plot(sub["budget"], sub["profit"], "-o", label=f"({lat:.3f},{lon:.3f})")

plt.xlabel("Presupuesto")
plt.ylabel("Profit")
plt.title("Profit vs Presupuesto (mejor lot area por ubicación)")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/profit_vs_budget_bestLot.png", dpi=300)
plt.close()


# ==============================================================
# 7. Ejecutar OPTIMIZACIONES con presupuestos fijos
# ==============================================================

results_final = []

print("\n\n=== EJECUTANDO OPTIMIZACIONES DETALLADAS (PRESUPUESTOS FIJOS) ===\n")

presupuestos = [250000, 400000, 500000, 1000000]

for (lat, lon), sub in df_f.groupby(["latitude", "longitude"]):

    print("="*85)
    print(f"UBICACIÓN: ({lat}, {lon})")
    print("="*85)

    # lot_area óptimo para ESTA ubicación (ya filtrado en df_f)
    lot_optimo = sub["lot_area"].iloc[0]
    print(f"✔ Lot area óptimo para esta ubicación: {lot_optimo}")

    # Ejecutar optimización REAL para cada presupuesto fijo
    for B in presupuestos:

        print(f"\nEjecutando optimización | Budget={B} | Lot Area={lot_optimo}")

        res = optimize_house(
            model=model,
            X=X,
            y_log=y_log,
            trained_feats=trained_feats,
            trained_stats=trained_stats,
            budget=B,
            zero=True,
            LON=lon,
            LAT=lat,
            Lot_Area=lot_optimo
        )

        if res is None:
            print("❌ Optimización falló.")
            continue

        final_house = res["final_house"][0]

        results_final.append({
            "latitude": lat,
            "longitude": lon,
            "budget": B,
            "lot_area": lot_optimo,
            "price_before": res["price_before"],
            "price_after": res["price_after"],
            "spent": res["spent"],
            "profit": res["profit"],
            "roi": res["roi"],
            **final_house
        })

# ==============================================================
# 8. Guardar resultados
# ==============================================================

df_out = pd.DataFrame(results_final)
df_out.to_csv(RESULT_CSV, index=False)

print("\n✔ Archivo generado correctamente:")
print(RESULT_CSV)
print(df_out.head())



# ==============================================================
# 8. GUARDAR RESULTADOS
# ==============================================================

df_out = pd.DataFrame(results_final)
df_out.to_csv(RESULT_CSV, index=False)

print("\n✔ Archivo generado correctamente:")
print(RESULT_CSV)
print(df_out.head())
