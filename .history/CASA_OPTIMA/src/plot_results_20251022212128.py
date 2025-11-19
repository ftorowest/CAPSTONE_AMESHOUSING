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
OUTPUT_DIR = "models"

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

# Mostrar estadísticas finales
print("\n" + "=" * 70)
print("ESTADÍSTICAS RESUMEN")
print("=" * 70)
print(roi_stats.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
print("=" * 70)
print("✓ Gráficos generados exitosamente")
print("="*70)