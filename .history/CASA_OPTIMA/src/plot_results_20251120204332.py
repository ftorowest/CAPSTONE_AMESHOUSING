"""
Plot Results Script
Lee el CSV de batch_optimization y genera gráficos:
1. Presupuesto vs ROI para las 10 casas (con etiquetas de gasto real)
2. ROI promedio por presupuesto
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
INPUT_CSV = "models/batch_optimization_results.csv"
OUTPUT_DIR = "results"

print("=" * 70)
print("GENERACIÓN DE GRÁFICOS - BATCH OPTIMIZATION")
print("=" * 70)

# Cargar datos
print(f"\n[1/3] Cargando datos desde {INPUT_CSV}...")
try:
    df = pd.read_csv(INPUT_CSV)
    print(f"✓ {len(df)} registros cargados")
except FileNotFoundError:
    print(f"✗ Error: No se encontró {INPUT_CSV}")
    print("   Ejecuta primero 'python src/batch_optimization.py'")
    exit(1)

# Validar columnas necesarias
required_cols = ["house_idx", "budget", "roi", "spent"]
if not all(col in df.columns for col in required_cols):
    print("✗ Error: CSV no tiene las columnas requeridas")
    exit(1)

# Limpiar datos (eliminar valores NaN en ROI)
df_clean = df.dropna(subset=["roi", "spent"])
print(f"✓ Datos limpios: {len(df_clean)} registros válidos")

# ========== GRÁFICO 1: Presupuesto vs ROI (10 casas) ==========
print("\n[2/3] Generando gráfico: Presupuesto vs ROI...")

plt.figure(figsize=(14, 8))

# Configurar paleta de colores para las 10 casas
houses = df_clean["house_idx"].unique()
colors = sns.color_palette("tab10", n_colors=len(houses))

for idx, (house, color) in enumerate(zip(houses, colors)):
    df_house = df_clean[df_clean["house_idx"] == house]
    
    # Línea conectando puntos
    plt.plot(df_house["budget"], df_house["roi"],
             marker="o", color=color, label=f"Casa {house}",
             linewidth=2, markersize=8, alpha=0.7)
    
    # Etiquetas con gasto real
    for _, row in df_house.iterrows():
        plt.annotate(
            f'${row["spent"]:,.0f}',
            xy=(row["budget"], row["roi"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=color, alpha=0.2)
        )

plt.xlabel("Presupuesto (USD)", fontsize=12, fontweight="bold")
plt.ylabel("ROI (Return on Investment)", fontsize=12, fontweight="bold")
plt.title("ROI vs Presupuesto para 10 Casas\n"
          "(Etiquetas muestran gasto real)",
          fontsize=14, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
           frameon=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()

output_path_1 = f"{OUTPUT_DIR}/roi_vs_budget_10houses.png"
plt.savefig(output_path_1, dpi=300, bbox_inches="tight")
print(f"✓ Guardado: {output_path_1}")
plt.close()

# ========== GRÁFICO 2: ROI Promedio por Presupuesto ==========
print("\n[3/3] Generando gráfico: ROI Promedio por Presupuesto...")

# Calcular ROI promedio y desviación estándar por presupuesto
roi_stats = df_clean.groupby("budget")["roi"].agg([
    ("mean", "mean"),
    ("std", "std"),
    ("min", "min"),
    ("max", "max")
]).reset_index()

plt.figure(figsize=(12, 7))

# Línea principal con área de confianza
plt.plot(roi_stats["budget"], roi_stats["mean"],
         marker="o", color="#2E86AB", linewidth=3,
         markersize=10, label="ROI Promedio")

# Área de desviación estándar
plt.fill_between(
    roi_stats["budget"],
    roi_stats["mean"] - roi_stats["std"],
    roi_stats["mean"] + roi_stats["std"],
    alpha=0.3,
    color="#2E86AB",
    label="± 1 Desviación Estándar"
)

# Etiquetas de valores
for _, row in roi_stats.iterrows():
    plt.annotate(
        f'{row["mean"]:.2f}',
        xy=(row["budget"], row["mean"]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="white", edgecolor="#2E86AB", alpha=0.9)
    )

plt.xlabel("Presupuesto (USD)", fontsize=12, fontweight="bold")
plt.ylabel("ROI Promedio", fontsize=12, fontweight="bold")
plt.title("ROI Promedio por Presupuesto\n"
          "(Promedio de 10 casas optimizadas)",
          fontsize=14, fontweight="bold")
plt.legend(loc="best", frameon=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()

output_path_2 = f"{OUTPUT_DIR}/roi_promedio_por_presupuesto.png"
plt.savefig(output_path_2, dpi=300, bbox_inches="tight")
print(f"✓ Guardado: {output_path_2}")
plt.close()

# Mostrar estadísticas ROI
print("\n" + "=" * 70)
print("ESTADÍSTICAS RESUMEN - ROI")
print("=" * 70)
print(roi_stats.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
print("=" * 70)

# ========== GRÁFICO 3: Análisis de Cambios por Atributo ==========
print("\n[Bonus] Analizando cambios por atributo...")

import json

# Parsear columnas de changes y cost_breakdown
all_changes = []
all_costs = []

for _, row in df_clean.iterrows():
    try:
        changes = json.loads(row["changes"])
        costs = json.loads(row["cost_breakdown"])
        all_changes.append(changes)
        all_costs.append(costs)
    except (json.JSONDecodeError, KeyError):
        all_changes.append({})
        all_costs.append({})

# Crear DataFrame con todos los cambios
changes_df = pd.DataFrame(all_changes)
costs_df = pd.DataFrame(all_costs)

# Agregar metadata
changes_df["house_idx"] = df_clean["house_idx"].values
changes_df["budget"] = df_clean["budget"].values
costs_df["house_idx"] = df_clean["house_idx"].values
costs_df["budget"] = df_clean["budget"].values

# Calcular estadísticas por atributo
attribute_stats = []

for col in changes_df.columns:
    if col in ["house_idx", "budget", "Year_Remod_Add"]:
        continue
    
    # Número de veces que se modificó (cambio != 0)
    modifications = (changes_df[col].fillna(0).abs() > 0.001).sum()
    
    # Cambio promedio cuando se modifica
    changes_nonzero = changes_df[col][changes_df[col].fillna(0).abs() > 0.001]
    avg_change = changes_nonzero.mean() if len(changes_nonzero) > 0 else 0
    
    # Gasto promedio cuando se modifica
    if col in costs_df.columns:
        costs_nonzero = costs_df[col][costs_df[col].fillna(0).abs() > 0.001]
        avg_cost = costs_nonzero.mean() if len(costs_nonzero) > 0 else 0
        total_cost = costs_df[col].fillna(0).sum()
    else:
        avg_cost = 0
        total_cost = 0
    
    if modifications > 0:
        attribute_stats.append({
            "attribute": col,
            "modifications": modifications,
            "avg_change": avg_change,
            "avg_cost": avg_cost,
            "total_cost": total_cost,
            "pct_modified": (modifications / len(df_clean)) * 100
        })

stats_df = pd.DataFrame(attribute_stats)
stats_df = stats_df.sort_values("modifications", ascending=False)

# Función para generar colores basados en cuartil 75
def get_colors_by_median(values):
    q75_val = values.quantile(0.75)
    return ["#5DADE2" if v > q75_val else "#95A5A6" for v in values]

# Gráfico 3A: Frecuencia de modificación
print("\n[3A] Generando gráfico: Frecuencia de modificaciones...")

# Top 15 atributos más modificados
top_15 = stats_df.head(15)
colors_freq = get_colors_by_median(top_15["modifications"])

plt.figure(figsize=(10, 8))
plt.barh(top_15["attribute"], top_15["modifications"], color=colors_freq)
plt.xlabel("Número de veces modificado", fontsize=12, fontweight="bold")
plt.title("Top 15: Atributos más modificados",
          fontsize=14, fontweight="bold")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

output_path_3a = f"{OUTPUT_DIR}/atributos_frecuencia.png"
plt.savefig(output_path_3a, dpi=300, bbox_inches="tight")
print(f"✓ Guardado: {output_path_3a}")
plt.close()

# Gráfico 3B: Porcentaje de modificación
print("\n[3B] Generando gráfico: % de modificación...")
colors_pct = get_colors_by_median(top_15["pct_modified"])

plt.figure(figsize=(10, 8))
plt.barh(top_15["attribute"], top_15["pct_modified"], color=colors_pct)
plt.xlabel("% de casos modificados", fontsize=12, fontweight="bold")
plt.title("Top 15: % de modificación",
          fontsize=14, fontweight="bold")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

output_path_3b = f"{OUTPUT_DIR}/atributos_porcentaje.png"
plt.savefig(output_path_3b, dpi=300, bbox_inches="tight")
print(f"✓ Guardado: {output_path_3b}")
plt.close()

# Gráfico 3C: Gasto promedio por atributo
print("\n[3C] Generando gráfico: Gasto promedio...")
top_15_cost = stats_df.nlargest(15, "avg_cost")
colors_avg = get_colors_by_median(top_15_cost["avg_cost"])

plt.figure(figsize=(10, 8))
plt.barh(top_15_cost["attribute"], top_15_cost["avg_cost"], color=colors_avg)
plt.xlabel("Gasto promedio (USD)", fontsize=12, fontweight="bold")
plt.title("Top 15: Gasto promedio al modificar",
          fontsize=14, fontweight="bold")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

output_path_3c = f"{OUTPUT_DIR}/atributos_gasto_promedio.png"
plt.savefig(output_path_3c, dpi=300, bbox_inches="tight")
print(f"✓ Guardado: {output_path_3c}")
plt.close()

# Gráfico 3D: Gasto total acumulado
print("\n[3D] Generando gráfico: Gasto total acumulado...")
top_15_total = stats_df.nlargest(15, "total_cost")
colors_total = get_colors_by_median(top_15_total["total_cost"])

plt.figure(figsize=(10, 8))
plt.barh(top_15_total["attribute"], top_15_total["total_cost"],
         color=colors_total)
plt.xlabel("Gasto total acumulado (USD)", fontsize=12, fontweight="bold")
plt.title("Top 15: Inversión total por atributo",
          fontsize=14, fontweight="bold")
plt.grid(axis="x", alpha=0.3)

# Formatear el eje X para evitar notación científica
ax = plt.gca()
ax.ticklabel_format(style='plain', axis='x')

plt.tight_layout()

output_path_3d = f"{OUTPUT_DIR}/atributos_gasto_total.png"
plt.savefig(output_path_3d, dpi=300, bbox_inches="tight")
print(f"✓ Guardado: {output_path_3d}")
plt.close()

# Guardar estadísticas en CSV
stats_csv = f"{OUTPUT_DIR}/attribute_statistics.csv"
stats_df.to_csv(stats_csv, index=False)
print(f"✓ Estadísticas guardadas: {stats_csv}")

print("\n" + "=" * 70)
print("TOP 10 ATRIBUTOS MÁS MODIFICADOS")
print("=" * 70)
print(stats_df.head(10)[["attribute", "modifications", "pct_modified",
                          "avg_cost", "total_cost"]].to_string(index=False))
print("=" * 70)
print("✓ Todos los gráficos generados exitosamente")
print("="*70)