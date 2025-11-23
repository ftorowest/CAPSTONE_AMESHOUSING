#!/usr/bin/env python3
"""
VISUALIZACIÓN DE ANÁLISIS V2 - REMODELACIONES
Genera gráficos profesionales para los 5 análisis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuración
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
RESULTS_DIR = Path("sensitivity_results_v2")
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

print("="*80)
print("GENERANDO GRÁFICOS - ANÁLISIS V2")
print("="*80)

# ============================================================================
# GRÁFICO 1: ROI vs Precio Base (Multi-Presupuesto)
# ============================================================================
print("\n📊 Gráfico 1: ROI vs Precio Base (5 Presupuestos)...")

df1 = pd.read_csv(RESULTS_DIR / "analysis1_roi_vs_price_multibudget.csv")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1.1: ROI vs Precio Base (cada presupuesto = color diferente)
for i, budget in enumerate(sorted(df1['budget'].unique())):
    data = df1[df1['budget'] == budget]
    axes[0, 0].scatter(data['baseline_price']/1000, data['roi']*100,
                      label=f'${budget/1000:.0f}k', s=80, alpha=0.7,
                      color=COLORS[i % len(COLORS)])

axes[0, 0].set_xlabel('Precio Base ($1000s)', fontweight='bold')
axes[0, 0].set_ylabel('ROI (%)', fontweight='bold')
axes[0, 0].set_title('ROI vs Precio Base\n(Por Presupuesto)', fontweight='bold', fontsize=14)
axes[0, 0].legend(title='Presupuesto')
axes[0, 0].grid(True, alpha=0.3)

# 1.2: Ganancia vs Precio Base
for i, budget in enumerate(sorted(df1['budget'].unique())):
    data = df1[df1['budget'] == budget]
    axes[0, 1].scatter(data['baseline_price']/1000, data['profit']/1000,
                      label=f'${budget/1000:.0f}k', s=80, alpha=0.7,
                      color=COLORS[i % len(COLORS)])

axes[0, 1].set_xlabel('Precio Base ($1000s)', fontweight='bold')
axes[0, 1].set_ylabel('Ganancia ($1000s)', fontweight='bold')
axes[0, 1].set_title('Ganancia Neta vs Precio Base\n(Por Presupuesto)', fontweight='bold', fontsize=14)
axes[0, 1].legend(title='Presupuesto')
axes[0, 1].grid(True, alpha=0.3)

# 1.3: % Incremento vs Precio Base
for i, budget in enumerate(sorted(df1['budget'].unique())):
    data = df1[df1['budget'] == budget]
    axes[1, 0].scatter(data['baseline_price']/1000, data['price_increase_pct'],
                      label=f'${budget/1000:.0f}k', s=80, alpha=0.7,
                      color=COLORS[i % len(COLORS)])

axes[1, 0].set_xlabel('Precio Base ($1000s)', fontweight='bold')
axes[1, 0].set_ylabel('Incremento de Precio (%)', fontweight='bold')
axes[1, 0].set_title('% Incremento vs Precio Base\n(Por Presupuesto)', fontweight='bold', fontsize=14)
axes[1, 0].legend(title='Presupuesto')
axes[1, 0].grid(True, alpha=0.3)

# 1.4: Boxplot ROI por Presupuesto
budget_groups = [df1[df1['budget']==b]['roi']*100 for b in sorted(df1['budget'].unique())]
bp = axes[1, 1].boxplot(budget_groups, 
                        labels=[f'${b/1000:.0f}k' for b in sorted(df1['budget'].unique())],
                        patch_artist=True, showmeans=True)
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[1, 1].set_xlabel('Presupuesto', fontweight='bold')
axes[1, 1].set_ylabel('ROI (%)', fontweight='bold')
axes[1, 1].set_title('Distribución de ROI por Presupuesto', fontweight='bold', fontsize=14)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "viz1_roi_vs_price_multibudget.png", dpi=300, bbox_inches='tight')
print("   ✓ Guardado: viz1_roi_vs_price_multibudget.png")
plt.close()

# ============================================================================
# GRÁFICO 2: Sensibilidad Presupuesto (Multi-Casa)
# ============================================================================
print("\n📊 Gráfico 2: Sensibilidad Presupuesto (Múltiples Casas)...")

df2 = pd.read_csv(RESULTS_DIR / "analysis2_budget_sensitivity_multihouse.csv")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 2.1: ROI vs Presupuesto (cada casa = línea diferente)
for house_idx in df2['house_idx'].unique()[:10]:  # Primeras 10 casas
    data = df2[df2['house_idx'] == house_idx].sort_values('budget')
    axes[0, 0].plot(data['budget']/1000, data['roi']*100, 
                   marker='o', linewidth=2, alpha=0.7, label=f'Casa #{house_idx}')

axes[0, 0].set_xlabel('Presupuesto ($1000s)', fontweight='bold')
axes[0, 0].set_ylabel('ROI (%)', fontweight='bold')
axes[0, 0].set_title('ROI vs Presupuesto\n(Por Casa)', fontweight='bold', fontsize=14)
axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# 2.2: Ganancia vs Presupuesto
for house_idx in df2['house_idx'].unique()[:10]:
    data = df2[df2['house_idx'] == house_idx].sort_values('budget')
    axes[0, 1].plot(data['budget']/1000, data['profit']/1000, 
                   marker='s', linewidth=2, alpha=0.7, label=f'Casa #{house_idx}')

axes[0, 1].set_xlabel('Presupuesto ($1000s)', fontweight='bold')
axes[0, 1].set_ylabel('Ganancia ($1000s)', fontweight='bold')
axes[0, 1].set_title('Ganancia vs Presupuesto\n(Por Casa)', fontweight='bold', fontsize=14)
axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# 2.3: ROI Promedio por Presupuesto
avg_roi = df2.groupby('budget')['roi'].mean() * 100
std_roi = df2.groupby('budget')['roi'].std() * 100
budgets = sorted(df2['budget'].unique())

axes[1, 0].bar(range(len(budgets)), avg_roi, 
              color=COLORS[0], edgecolor='black', alpha=0.8)
axes[1, 0].errorbar(range(len(budgets)), avg_roi, yerr=std_roi, 
                   fmt='none', color='black', capsize=5, linewidth=2)
axes[1, 0].set_xticks(range(len(budgets)))
axes[1, 0].set_xticklabels([f'${b/1000:.0f}k' for b in budgets])
axes[1, 0].set_ylabel('ROI Promedio (%)', fontweight='bold')
axes[1, 0].set_xlabel('Presupuesto', fontweight='bold')
axes[1, 0].set_title('ROI Promedio ± Desviación Estándar\n(Todas las Casas)', 
                    fontweight='bold', fontsize=14)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 2.4: Heatmap ROI (Casa × Presupuesto)
pivot = df2.pivot_table(values='roi', index='house_idx', columns='budget', aggfunc='mean')
im = axes[1, 1].imshow(pivot.values * 100, cmap='RdYlGn', aspect='auto')
axes[1, 1].set_xticks(range(len(pivot.columns)))
axes[1, 1].set_xticklabels([f'${b/1000:.0f}k' for b in pivot.columns], rotation=45)
axes[1, 1].set_yticks(range(len(pivot.index)))
axes[1, 1].set_yticklabels([f'#{idx}' for idx in pivot.index], fontsize=8)
axes[1, 1].set_xlabel('Presupuesto', fontweight='bold')
axes[1, 1].set_ylabel('Casa ID', fontweight='bold')
axes[1, 1].set_title('Heatmap ROI (%)\n(Casa × Presupuesto)', fontweight='bold', fontsize=14)
plt.colorbar(im, ax=axes[1, 1], label='ROI (%)')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "viz2_budget_sensitivity_multihouse.png", dpi=300, bbox_inches='tight')
print("   ✓ Guardado: viz2_budget_sensitivity_multihouse.png")
plt.close()

# ============================================================================
# GRÁFICO 3: Sensibilidad PWL
# ============================================================================
print("\n📊 Gráfico 3: Sensibilidad PWL...")

df3 = pd.read_csv(RESULTS_DIR / "analysis3_pwl_sensitivity.csv")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 3.1: ROI vs PWL
axes[0].plot(df3['pwl_k'], df3['roi']*100, marker='o', linewidth=3, 
            markersize=10, color=COLORS[0])
axes[0].set_xlabel('Parámetro PWL (k)', fontweight='bold')
axes[0].set_ylabel('ROI (%)', fontweight='bold')
axes[0].set_title('ROI vs Parámetro PWL', fontweight='bold', fontsize=14)
axes[0].grid(True, alpha=0.3)

# 3.2: Ganancia vs PWL
axes[1].plot(df3['pwl_k'], df3['profit']/1000, marker='s', linewidth=3, 
            markersize=10, color=COLORS[1])
axes[1].set_xlabel('Parámetro PWL (k)', fontweight='bold')
axes[1].set_ylabel('Ganancia ($1000s)', fontweight='bold')
axes[1].set_title('Ganancia vs Parámetro PWL', fontweight='bold', fontsize=14)
axes[1].grid(True, alpha=0.3)

# 3.3: Precio Optimizado vs PWL
axes[2].plot(df3['pwl_k'], df3['price_before']/1000, 
            label='Precio Base', linestyle='--', linewidth=2, color='gray')
axes[2].plot(df3['pwl_k'], df3['optimized_price']/1000, 
            label='Precio Optimizado', marker='d', linewidth=3, 
            markersize=10, color=COLORS[2])
axes[2].set_xlabel('Parámetro PWL (k)', fontweight='bold')
axes[2].set_ylabel('Precio ($1000s)', fontweight='bold')
axes[2].set_title('Precio vs Parámetro PWL', fontweight='bold', fontsize=14)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "viz3_pwl_sensitivity.png", dpi=300, bbox_inches='tight')
print("   ✓ Guardado: viz3_pwl_sensitivity.png")
plt.close()

# ============================================================================
# GRÁFICO 4: Predicción vs Real
# ============================================================================
print("\n📊 Gráfico 4: Predicción vs Real...")

df4 = pd.read_csv(RESULTS_DIR / "analysis4_prediction_vs_real.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 4.1: Scatter Predicho vs Real (precio base)
axes[0, 0].scatter(df4['real_price_before']/1000, df4['predicted_price_before']/1000,
                  alpha=0.5, s=50, color=COLORS[0])
# Línea perfecta
max_price = max(df4['real_price_before'].max(), df4['predicted_price_before'].max()) / 1000
axes[0, 0].plot([0, max_price], [0, max_price], 'r--', linewidth=2, label='Predicción Perfecta')
axes[0, 0].set_xlabel('Precio Real ($1000s)', fontweight='bold')
axes[0, 0].set_ylabel('Precio Predicho ($1000s)', fontweight='bold')
axes[0, 0].set_title('Precio Predicho vs Real (Antes de Optimizar)', fontweight='bold', fontsize=14)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 4.2: Histograma de errores %
axes[0, 1].hist(df4['price_diff_pct_before'], bins=30, 
               color=COLORS[1], edgecolor='black', alpha=0.7)
axes[0, 1].axvline(df4['price_diff_pct_before'].mean(), color='red', 
                  linestyle='--', linewidth=2, label=f'Media: {df4["price_diff_pct_before"].mean():.2f}%')
axes[0, 1].set_xlabel('Error de Predicción (%)', fontweight='bold')
axes[0, 1].set_ylabel('Frecuencia', fontweight='bold')
axes[0, 1].set_title('Distribución de Errores de Predicción', fontweight='bold', fontsize=14)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 4.3: Error vs Precio Real
axes[1, 0].scatter(df4['real_price_before']/1000, df4['price_diff_before']/1000,
                  alpha=0.5, s=50, color=COLORS[2])
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Precio Real ($1000s)', fontweight='bold')
axes[1, 0].set_ylabel('Error Absoluto ($1000s)', fontweight='bold')
axes[1, 0].set_title('Error de Predicción vs Precio Real', fontweight='bold', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)

# 4.4: Estadísticas por modelo (si hay)
if 'model' in df4.columns and len(df4['model'].unique()) > 1:
    model_errors = df4.groupby('model')['price_diff_pct_before'].agg(['mean', 'std'])
    models = model_errors.index.tolist()
    
    x = np.arange(len(models))
    axes[1, 1].bar(x, model_errors['mean'], yerr=model_errors['std'],
                  color=COLORS[:len(models)], edgecolor='black', 
                  alpha=0.8, capsize=5)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].set_ylabel('Error Promedio (%)', fontweight='bold')
    axes[1, 1].set_title('Error de Predicción por Modelo', fontweight='bold', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
else:
    axes[1, 1].text(0.5, 0.5, 'Estadísticas adicionales\nno disponibles', 
                   ha='center', va='center', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "viz4_prediction_vs_real.png", dpi=300, bbox_inches='tight')
print("   ✓ Guardado: viz4_prediction_vs_real.png")
plt.close()

# ============================================================================
# GRÁFICO 5: XGBoost vs Linear
# ============================================================================
print("\n📊 Gráfico 5: XGBoost vs Linear...")

df5 = pd.read_csv(RESULTS_DIR / "analysis5_xgb_vs_linear.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1: ROI Comparación
xgb_data = df5[df5['model'] == 'XGBoost']
linear_data = df5[df5['model'] == 'Linear']

x = np.arange(len(xgb_data))
width = 0.35

axes[0, 0].bar(x - width/2, xgb_data['roi']*100, width, 
              label='XGBoost', color=COLORS[0], edgecolor='black', alpha=0.8)
axes[0, 0].bar(x + width/2, linear_data['roi']*100, width, 
              label='Linear', color=COLORS[1], edgecolor='black', alpha=0.8)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels([f'{idx}' for idx in xgb_data['house_idx']], rotation=45, ha='right')
axes[0, 0].set_xlabel('ID Casa', fontweight='bold')
axes[0, 0].set_ylabel('ROI (%)', fontweight='bold')
axes[0, 0].set_title('Comparación ROI: XGBoost vs Linear', fontweight='bold', fontsize=14)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 5.2: Ganancia Comparación
axes[0, 1].bar(x - width/2, xgb_data['profit']/1000, width, 
              label='XGBoost', color=COLORS[0], edgecolor='black', alpha=0.8)
axes[0, 1].bar(x + width/2, linear_data['profit']/1000, width, 
              label='Linear', color=COLORS[1], edgecolor='black', alpha=0.8)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels([f'{idx}' for idx in xgb_data['house_idx']], rotation=45, ha='right')
axes[0, 1].set_xlabel('ID Casa', fontweight='bold')
axes[0, 1].set_ylabel('Ganancia ($1000s)', fontweight='bold')
axes[0, 1].set_title('Comparación Ganancia: XGBoost vs Linear', fontweight='bold', fontsize=14)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 5.3: Scatter XGBoost vs Linear (ROI)
axes[1, 0].scatter(xgb_data['roi']*100, linear_data['roi']*100, 
                  s=150, alpha=0.7, color=COLORS[2], edgecolors='black', linewidth=2)
max_roi = max(xgb_data['roi'].max(), linear_data['roi'].max()) * 100
axes[1, 0].plot([0, max_roi], [0, max_roi], 'r--', linewidth=2, label='ROI Igual')
axes[1, 0].set_xlabel('XGBoost ROI (%)', fontweight='bold')
axes[1, 0].set_ylabel('Linear ROI (%)', fontweight='bold')
axes[1, 0].set_title('XGBoost vs Linear (ROI)', fontweight='bold', fontsize=14)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5.4: Número de cambios (si disponible)
if 'n_changes' in df5.columns:
    axes[1, 1].bar(x - width/2, xgb_data['n_changes'], width, 
                  label='XGBoost', color=COLORS[0], edgecolor='black', alpha=0.8)
    axes[1, 1].bar(x + width/2, linear_data['n_changes'], width, 
                  label='Linear', color=COLORS[1], edgecolor='black', alpha=0.8)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'{idx}' for idx in xgb_data['house_idx']], rotation=45, ha='right')
    axes[1, 1].set_xlabel('ID Casa', fontweight='bold')
    axes[1, 1].set_ylabel('Número de Atributos Modificados', fontweight='bold')
    axes[1, 1].set_title('Cantidad de Cambios: XGBoost vs Linear', fontweight='bold', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
else:
    # Boxplot de comparación general
    bp = axes[1, 1].boxplot([xgb_data['roi']*100, linear_data['roi']*100],
                            labels=['XGBoost', 'Linear'],
                            patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], COLORS[:2]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 1].set_ylabel('ROI (%)', fontweight='bold')
    axes[1, 1].set_title('Distribución ROI: XGBoost vs Linear', fontweight='bold', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "viz5_xgb_vs_linear.png", dpi=300, bbox_inches='tight')
print("   ✓ Guardado: viz5_xgb_vs_linear.png")
plt.close()

# ============================================================================
# GRÁFICO 6: Patrones de Modificación por Modelo (SIMPLIFICADO)
# ============================================================================
print("\n📊 Gráfico 6: Patrones de Modificación por Modelo...")

# Cargar datos de patrones
df_patterns = pd.read_csv(RESULTS_DIR / "analysis5_improvement_patterns.csv")

# Filtrar solo atributos que tienen cambios significativos (frecuencia > 0)
df_patterns_active = df_patterns[df_patterns['frecuencia'] > 0].copy()

# Ordenar por diferencia acumulada absoluta
df_patterns_active = df_patterns_active.sort_values('abs_diff_acumulada', ascending=False)

# Tomar top 10 atributos más modificados (más claro y conciso)
top_n = min(10, len(df_patterns_active))
df_top = df_patterns_active.head(top_n).copy()

# Figura con 2 subplots (solo gráficos de la izquierda)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 6.1: Heatmap de cambios promedio por modelo
attributes = df_top['attribute'].values
xgb_values = df_top['xgb_promedio'].values
linear_values = df_top['linear_promedio'].values

# Crear DataFrame para heatmap
heatmap_data = pd.DataFrame({
    'XGBoost': xgb_values,
    'Linear': linear_values
}, index=attributes)

sns.heatmap(heatmap_data.T, annot=True, fmt='.1f', cmap='RdYlGn', 
            center=0, cbar_kws={'label': 'Cambio Promedio'}, ax=axes[0],
            linewidths=0.5, linecolor='gray', annot_kws={'size': 11})
axes[0].set_title('Cambios Promedio por Modelo\n(Top 10 Atributos Más Significativos)', 
                  fontweight='bold', fontsize=15)
axes[0].set_xlabel('Atributo', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Modelo', fontweight='bold', fontsize=12)
axes[0].tick_params(axis='x', rotation=45, labelsize=10)
axes[0].tick_params(axis='y', labelsize=12)

# 6.2: Diferencia acumulada (XGBoost - Linear)
x = np.arange(len(df_top))
colors_diff = ['#2E86AB' if val > 0 else '#A23B72' for val in df_top['diff_acumulada']]
axes[1].barh(x, df_top['diff_acumulada'], color=colors_diff, edgecolor='black', alpha=0.85, linewidth=1.5)
axes[1].set_yticks(x)
axes[1].set_yticklabels(df_top['attribute'], fontsize=11)
axes[1].set_xlabel('Diferencia Acumulada (XGBoost - Linear)', fontweight='bold', fontsize=12)
axes[1].set_title('Diferencia Sistemática entre Modelos\n(Azul = XGBoost > Linear | Morado = Linear > XGBoost)', 
                  fontweight='bold', fontsize=15)
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=2.5)
axes[1].grid(True, alpha=0.3, axis='x')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "viz6_model_modification_patterns.png", dpi=300, bbox_inches='tight')
print("   ✓ Guardado: viz6_model_modification_patterns.png")
plt.close()

print("\n" + "="*80)
print("✅ TODOS LOS GRÁFICOS GENERADOS")
print("="*80)
print(f"\n📁 Archivos en: {RESULTS_DIR.absolute()}/")
print("\n   1. viz1_roi_vs_price_multibudget.png")
print("   2. viz2_budget_sensitivity_multihouse.png")
print("   3. viz3_pwl_sensitivity.png")
print("   4. viz4_prediction_vs_real.png")
print("   5. viz5_xgb_vs_linear.png")
print("   6. viz6_model_modification_patterns.png (Patrones de Modificación - Top 10)")
print("="*80)

