# ANÁLISIS DE SENSIBILIDAD V2 - REMODELACIONES

**Fecha:**### **5. XGBoost vs Linear** ⭐ ACTUALIZADO
- **Objetivo:** Contrastar remodelaciones sugeridas por ambos modelos **e identificar PATRONES ACUMULADOS en las diferencias (no casos extremos)**
- **Configuración:**
  - 20 casas representativas
  - Presupuesto fijo: $150k
  - Modelos: XGBoost y Linear
  - Total: 40 optimizaciones (20×2)
- **Gráficos:**
  - Comparación ROI (barras lado a lado)
  - Comparación Ganancia (barras lado a lado)
  - Scatter XGBoost vs Linear ROI
  - Número de atributos modificados
- **NUEVO:** CSVs de patrones acumulados
  - **`analysis5_improvement_patterns.csv`** (⭐ PRINCIPAL):
    - **Patrones generales** por atributo (no casos extremos)
    - Columnas: `attribute`, `frecuencia`, `xgb_promedio`, `linear_promedio`, `diff_acumulada`, `diff_promedio`
    - Ordenado por **diferencia acumulada absoluta** (patrones más relevantes primero)
    - **Ejemplo:** "En las 20 casas, XGBoost aumenta First_Flr_SF +300 pies² más que Linear (acumulado: +6000 pies²)"
  - `analysis5_improvement_differences_detailed.csv`:
    - Todas las diferencias casa por casa (detalle completo)
- **Insight esperado:** ¿Qué atributos priorizan sistemáticamente diferente cada modelo? (ej: XGBoost prefiere ampliar pies² vs Linear prefiere mejorar calidad)mbre, 2025  
**Objetivo:** Análisis exhaustivo de optimización de casas (remodelaciones únicamente)

---

## 📋 DESCRIPCIÓN DE LOS 5 ANÁLISIS

### **1. ROI vs Precio Base (Multi-Presupuesto)**
- **Objetivo:** Ver cómo el ROI, ganancia e incremento % varían según precio base y presupuesto
- **Configuración:**
  - 30 casas de diferentes rangos de precio
  - 5 presupuestos: $50k, $100k, $150k, $200k, $250k
  - Total: 150 optimizaciones
- **Gráficos:**
  - ROI vs Precio Base (scatter por presupuesto)
  - Ganancia vs Precio Base (scatter por presupuesto)
  - % Incremento vs Precio Base (scatter por presupuesto)
  - Boxplot ROI por Presupuesto

### **2. Sensibilidad Presupuesto (Multi-Casa)** ⭐ ACTUALIZADO
- **Objetivo:** Entender cómo diferentes casas responden a variaciones de presupuesto, **incluyendo presupuestos altos ($500k) para validar rendimientos decrecientes**
- **Configuración:**
  - 10 casas representativas
  - **6 presupuestos: $50k, $100k, $150k, $200k, $250k, $500k** ← **NUEVO: +$500k**
  - Total: **60 optimizaciones** (+10 vs versión anterior)
- **Gráficos:**
  - ROI vs Presupuesto (líneas por casa)
  - Ganancia vs Presupuesto (líneas por casa)
  - ROI Promedio ± Desviación Estándar
  - Heatmap ROI (Casa × Presupuesto)
- **Insight esperado:** ¿A partir de qué presupuesto el ROI cae significativamente?

### **3. Sensibilidad PWL**
- **Objetivo:** Validar robustez del parámetro de aproximación lineal
- **Configuración:**
  - 1 casa representativa
  - 7 valores PWL: [10, 15, 20, 25, 30, 40, 50]
  - Presupuesto fijo: $125k
  - Total: 7 optimizaciones
- **Gráficos:**
  - ROI vs PWL
  - Ganancia vs PWL
  - Precio Optimizado vs PWL

### **4. Predicción vs Real (Tracking Automático)**
- **Objetivo:** Evaluar precisión del modelo predictor comparando precio real vs predicho
- **Configuración:**
  - Recopilación automática de TODAS las optimizaciones (Análisis 1, 2, 5)
  - Registra: precio real, precio predicho antes, precio predicho después
  - Total: ~200+ registros
- **Gráficos:**
  - Scatter Predicho vs Real (con línea perfecta)
  - Histograma de errores de predicción %
  - Error vs Precio Real
  - Error por modelo (si aplica)

### **5. Comparación XGBoost vs Linear**
- **Objetivo:** Contrastar remodelaciones sugeridas por ambos modelos
- **Configuración:**
  - 5 casas representativas
  - Presupuesto fijo: $150k
  - Modelos: XGBoost y Linear
  - Total: 10 optimizaciones (5×2)
- **Gráficos:**
  - Comparación ROI (barras lado a lado)
  - Comparación Ganancia (barras lado a lado)
  - Scatter XGBoost vs Linear ROI
  - Número de atributos modificados

---

## 🚀 EJECUCIÓN

### **Paso 1: Sincronización con versión nueva**

```bash
# Respaldar trabajo actual
mkdir ~/backup_analysis_$(date +%Y%m%d)
cp -r CASA_OPTIMA/sensitivity_results ~/backup_analysis_$(date +%Y%m%d)/
cp CASA_OPTIMA/sensitivity_analysis_v2.py ~/backup_analysis_$(date +%Y%m%d)/
cp CASA_OPTIMA/visualize_analysis_v2.py ~/backup_analysis_$(date +%Y%m%d)/

# Limpiar cambios locales
cd /path/to/CAPSTONE_AMESHOUSING
git restore CASA_OPTIMA/src/train_model.py
git clean -fd CASA_OPTIMA/src/__pycache__/
git restore CASA_OPTIMA/models/*.pkl

# Pull versión nueva
git pull origin main

# Restaurar scripts V2 (deberían ser compatibles)
cp ~/backup_analysis_*/sensitivity_analysis_v2.py CASA_OPTIMA/
cp ~/backup_analysis_*/visualize_analysis_v2.py CASA_OPTIMA/
```

### **Paso 2: Reentrenar modelos con código nuevo**

```bash
cd CASA_OPTIMA
source ../venv/bin/activate

# Borrar modelos antiguos
rm -f models/*.pkl

# Entrenar con código NUEVO
python3 src/train_model.py
```

### **Paso 3: Ejecutar análisis de sensibilidad**

```bash
# Ejecutar análisis (puede tardar varias horas)
python3 sensitivity_analysis_v2.py 2>&1 | tee sensitivity_log_v2.txt

# En background si prefieres:
# nohup python3 sensitivity_analysis_v2.py > sensitivity_log_v2.txt 2>&1 &
```

### **Paso 4: Generar gráficos**

```bash
# Después de que termine el análisis
python3 visualize_analysis_v2.py
```

---

## 📊 RESULTADOS ESPERADOS

### **Archivos CSV generados:**
```
sensitivity_results_v2/
├── analysis1_roi_vs_price_multibudget.csv      (~150 filas)
├── analysis2_budget_sensitivity_multihouse.csv  (~50 filas)
├── analysis3_pwl_sensitivity.csv                (~7 filas)
├── analysis4_prediction_vs_real.csv             (~200+ filas)
├── analysis5_xgb_vs_linear.csv                  (~10 filas)
└── summary_report.txt                           (resumen ejecutivo)
```

### **Gráficos PNG generados:**
```
sensitivity_results_v2/
├── viz1_roi_vs_price_multibudget.png      (4 sub-gráficos)
├── viz2_budget_sensitivity_multihouse.png  (4 sub-gráficos)
├── viz3_pwl_sensitivity.png                (3 sub-gráficos)
├── viz4_prediction_vs_real.png             (4 sub-gráficos)
└── viz5_xgb_vs_linear.png                  (4 sub-gráficos)
```

---

## ⏱️ TIEMPO ESTIMADO ⭐ ACTUALIZADO

- **Análisis 1 (ROI Multi-Presupuesto):** ~2-3 horas (150 optimizaciones)
- **Análisis 2 (Budget Multi-Casa):** ~1.5-2 horas (**60 optimizaciones** - incluye $500k)
- **Análisis 3 (PWL):** ~15-20 minutos (7 optimizaciones)
- **Análisis 4 (Tracking):** Automático (sin tiempo adicional)
- **Análisis 5 (XGBoost vs Linear):** ~1-1.5 horas (**40 optimizaciones** - 20 casas × 2 modelos)

**TOTAL: 257 optimizaciones**  
**TIEMPO ESTIMADO:** 5-7 horas (dependiendo de hardware y complejidad del modelo)

---

## 🎯 NOTAS IMPORTANTES

### **Enfoque en Remodelaciones:**
- El código nuevo incluye lógica para construcción desde cero Y remodelaciones
- Estos análisis se enfocan **únicamente en remodelaciones**
- Si el código nuevo cambia la interfaz de `optimize_house()`, ajustar en línea 62-80

### **Compatibilidad:**
- Scripts diseñados para ser robustos ante cambios menores
- Si hay cambios en estructura de `result` dict, revisar líneas 82-95 y 62-80
- El tracking automático (Análisis 4) se integra en cada llamada a `run_optimization()`

### **Ajustes Posibles:**
```python
# Si quieres cambiar número de casas:
N_HOUSES_PRICE_ANALYSIS = 30  # Línea 33 (Análisis 1)
N_HOUSES_BUDGET_ANALYSIS = 10  # Línea 34 (Análisis 2)
N_HOUSES_COMPARISON = 5        # Línea 35 (Análisis 5)

# Si quieres cambiar presupuestos:
BUDGET_LEVELS = [50000, 100000, 150000, 200000, 250000]  # Línea 29

# Si quieres cambiar valores PWL:
PWL_VALUES = [10, 15, 20, 25, 30, 40, 50]  # Línea 30
```

---

## 📧 TROUBLESHOOTING

### **Error: "optimize_house() missing required argument"**
- El código nuevo cambió la firma de `optimize_house()`
- Revisar `src/optimization.py` y ajustar llamada en `sensitivity_analysis_v2.py` línea 62-80

### **Error: "KeyError: 'profit'"**
- El dict de resultado cambió estructura
- Revisar qué keys devuelve ahora `optimize_house()` y ajustar línea 82-95

### **Modelos no se cargan:**
- Asegurarse de haber reentrenado con código nuevo: `python3 src/train_model.py`
- Verificar que existan `models/xgb_optuna_model.pkl` y `models/linear_model.pkl`

### **Análisis muy lento:**
- Reducir `N_HOUSES_*` en líneas 33-35
- Reducir `BUDGET_LEVELS` a 3 valores: `[50000, 150000, 250000]`
- Ejecutar en background: `nohup python3 sensitivity_analysis_v2.py &`

---

## ✅ CHECKLIST PRE-EJECUCIÓN

- [ ] Git pull completado
- [ ] Modelos reentrenados con código nuevo
- [ ] Scripts V2 copiados a CASA_OPTIMA/
- [ ] Directorio `sensitivity_results_v2/` creado (automático)
- [ ] Venv activado: `source ../venv/bin/activate`
- [ ] Gurobi license válida (verificar con `python3 -c "import gurobipy"`)

---

**¡Listo para ejecutar cuando el código nuevo esté disponible!** 🚀
