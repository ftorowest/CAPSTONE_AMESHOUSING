"""
optimization.py
----------------
Integra el modelo predictivo (Linear o XGB) con Gurobi para determinar
la "Casa Óptima" dentro de un presupuesto, maximizando la ganancia neta.

⚙️ Idea general:
A partir de una casa base y un modelo de tasación entrenado, se busca
la combinación de mejoras que:
    - No exceda el presupuesto disponible,
    - Aumente el valor estimado por el modelo predictivo,
    - Maximize la ganancia: (Precio nuevo - Precio original - Costo mejoras).

Uso típico:
    from src.optimization import optimize_house
    optimize_house(model, X, y_log, trained_feats, trained_stats)
"""

# ==========================================================
# 📦 Importación de librerías
# ==========================================================
import gurobipy as gp                    # Modelador y solver de optimización
from gurobi_ml import add_predictor_constr  # Vincula el modelo ML con Gurobi
from gurobipy import GRB
import numpy as np
import pandas as pd


# ==========================================================
# 🧩 Función principal de optimización
# ==========================================================
def optimize_house(
    model,
    X,
    y_log,
    trained_feats,
    trained_stats,
    baseline_idx=0,
    budget=200_000,
    pwl_k=25
):
    """
    Ejecuta la optimización de diseño de vivienda usando un modelo predictivo
    (Linear o XGB), maximizando la ganancia neta esperada.

    Parámetros
    ----------
    model : objeto entrenado (Pipeline con LinearRegression o XGBRegressor)
    X : pd.DataFrame
        Dataset original con las features de entrenamiento.
    y_log : pd.Series
        Precio de venta en log (SalePrice_Log).
    trained_feats : list
        Variables usadas para entrenar el modelo.
    trained_stats : pd.DataFrame
        Estadísticos de las features (q05, median, q95).
    baseline_idx : int
        Índice de la casa base que se optimizará.
    budget : float
        Presupuesto máximo para mejoras.
    pwl_k : int
        Número de segmentos para aproximar exp() (función no lineal).
    """

    print("\n=== 🧱 OPTIMIZACIÓN CASA ÓPTIMA (MODO PROFIT) ===")

    # ==========================================================
    # 1️⃣ Selección de la vivienda base
    # ==========================================================
    n = len(X)
    idx = baseline_idx if 0 <= baseline_idx < n else 0
    baseline = X.iloc[idx].astype(float)

    # Predicciones iniciales (en log y en valor real)
    pred_log = float(model.predict(baseline.to_frame().T)[0])
    price_pred = float(np.expm1(pred_log))        # Valor estimado (modelo)
    price_real = float(np.expm1(y_log.iloc[idx])) # Valor real en datos originales

    print(f"\n🏠 Casa {idx} seleccionada como baseline")
    print(f"Precio predicho: {price_pred:,.0f} | Precio real: {price_real:,.0f}")

    # ==========================================================
    # 2️⃣ Definición de costos base y “espacio para mejorar”
    # ==========================================================
    # Costo unitario aproximado de aumentar cada variable (en USD o equivalente)
    # ==========================================================
    # 💰 Costos unitarios estimados por variable (en USD aprox.)
    # ==========================================================
    default_costs = {
        "First_Flr_SF":        200,   # costo por pie² en el primer piso
        "Second_Flr_SF":       220,   # segundo piso es más caro estructuralmente
        "Year_Built":            0,   # costo por "año equivalente" de antigüedad (renovaciones estructurales)
        "Exter_Qual":          7000,  # mejorar calidad exterior
        "Total_Bsmt_SF":        80,   # costo por pie² adicional en sótano
        "Lot_Area":             0,   # costo por pie² de terreno
        "Overall_Cond":       12000,  # mejorar condición general
        "Garage_Cars":        17000,  # agregar espacio de estacionamiento
        "Kitchen_Qual":        8000,  # mejorar calidad de cocina
        "Fireplaces":          6000,  # agregar chimenea
        "Year_Remod_Add":     0,   # costo asociado a remodelación reciente
        "Sale_Condition_Normal": 0,   # categórica (no accionable directamente)
        "Longitude":              0,  # ubicación fija (no modificable)
        "BsmtFin_Type_1":      4000,  # mejorar tipo de acabado en sótano
        "Bsmt_Unf_SF":          80,   # terminar superficie no acabada
        "Full_Bath":          25000,  # agregar baño completo
        "Bsmt_Qual":           5000,  # mejorar calidad del sótano
        "Latitude":               0,  # ubicación fija (no modificable)
        "Bsmt_Exposure":       4000,  # agregar ventanas o acceso al exterior
        "TotRms_AbvGrd":      10000,  # costo por agregar una habitación
    }


    # ==========================================================
    # 📈 "Room to grow": máximos incrementos posibles por variable
    # ==========================================================
    room = {
        "First_Flr_SF":       400.0,  # pies² adicionales en primer piso
        "Second_Flr_SF":      400.0,  # pies² adicionales en segundo piso
        "Year_Built":           0,  
        "Exter_Qual":           1.0,  # subir un nivel de calidad (TA→Gd→Ex)
        "Total_Bsmt_SF":      300.0,
        "Lot_Area":                 0,  
        "Overall_Cond":         1.0,  # subir un nivel de condición general
        "Garage_Cars":          1.0,
        "Kitchen_Qual":         1.0,  # subir un nivel (TA→Gd→Ex)
        "Fireplaces":           1.0,
        "Year_Remod_Add":       0 ,  # remodelar o actualizar hasta 3 "años equivalentes"
        "Sale_Condition_Normal":0.0,  # no se modifica
        "Longitude":            0.0,  # ubicación fija
        "BsmtFin_Type_1":       1.0,  # subir un nivel de terminación
        "Bsmt_Unf_SF":        200.0,  # pies² que se pueden terminar
        "Full_Bath":            1.0,
        "Bsmt_Qual":            1.0,
        "Latitude":             0.0,
        "Bsmt_Exposure":        1.0,
        "TotRms_AbvGrd":        1.0,  # agregar una habitación adicional
    }

    q95 = trained_stats["q95"]  # Percentil 95 usado para evitar valores irreales

    # ==========================================================
    # 3️⃣ Construcción de límites y costos efectivos
    # ==========================================================
    bounds, costs = {}, {}

    for f in trained_feats:
        base = float(baseline.get(f, X[f].median()))
        ub_room = base + room.get(f, 0.0)     # límite según “room to grow”
        ub_q95  = float(q95.get(f, base))     # límite según distribución
        lb = base                             # no se reduce el valor base
        ub = max(lb, min(ub_room, ub_q95))    # se toma el mínimo de ambos
        bounds[f] = (lb, ub)
        costs[f]  = float(default_costs.get(f, 0.0))

    # ==========================================================
    # 4️⃣ Creación del modelo de optimización Gurobi
    # ==========================================================
    m = gp.Model("casa_optima_profit")

    # Variables que deben ser enteras (no continuas)
    int_like = {
        "Garage_Cars", "Full_Bath", "Fireplaces",
        "Overall_Qual", "KitchenQual_ord", "GarageFinish_ord"
    }

    x = {}  # variables de decisión (cada feature optimizable)

    # --- Creación dinámica de variables ---
    for c in trained_feats:
        lb, ub = bounds[c]
        if c in int_like:
            # Variables enteras: se redondean los límites
            lb_i, ub_i = int(np.floor(lb)), int(np.ceil(ub))
            if ub_i < lb_i:
                ub_i = lb_i
            x[c] = m.addVar(lb=lb_i, ub=ub_i, vtype=GRB.INTEGER, name=c)
        else:
            # Variables continuas (m², pies², etc.)
            x[c] = m.addVar(lb=float(lb), ub=float(ub), vtype=GRB.CONTINUOUS, name=c)

    # ==========================================================
    # 5️⃣ Restricciones básicas
    # ==========================================================

    # 💰 Restricción de presupuesto total
    cost_expr = gp.quicksum(costs[c] * (x[c] - float(baseline[c])) for c in trained_feats)
    m.addConstr(cost_expr <= float(budget), name="Budget")

    # 🧩 --- Espacio para restricciones adicionales ---
    # Ejemplos posibles:
    # m.addConstr(x["Full_Bath"] <= x["Bedroom_AbvGr"], name="Baths_limit")
    # m.addConstr(x["Garage_Cars"] <= 3, name="Garage_limit")
    # m.addConstr(x["Overall_Qual"] >= x["KitchenQual_ord"], name="Quality_relation")
    # -------------------------------------------------

    # ==========================================================
    # 6️⃣ Conexión con el modelo predictivo (Gurobi + ML)
    # ==========================================================
    x_df = pd.DataFrame([[x[c] for c in trained_feats]], columns=trained_feats)
    y_pred_log = m.addVar(name="y_pred_log")  # variable para el precio predicho (log)

    add_predictor_constr(
        gp_model=m,          # modelo de Gurobi
        predictor=model,     # modelo de ML (XGB o Linear)
        input_vars=x_df,     # variables de entrada (features)
        output_vars=y_pred_log  # salida (log-precio)
    )

    # ==========================================================
    # 7️⃣ Conversión del log-precio a precio real (PWL)
    # ==========================================================
    ymin, ymax = np.percentile(y_log, [1, 99])
    ymin, ymax = float(np.clip(ymin, -1e2, 1e2)), float(np.clip(ymax, -1e2, 1e2))
    if ymax <= ymin:
        ymin, ymax = 10.5, 13.5  # valores razonables por defecto

    xs = np.linspace(ymin, ymax, pwl_k).tolist()
    ys = [float(np.clip(np.expm1(v), -1e9, 1e9)) for v in xs]

    price = m.addVar(name="price")
    m.addGenConstrPWL(y_pred_log, price, xs, ys, name="log_to_price")

    # Precio antes de mejorar (baseline)
    baseline_vec = pd.DataFrame([baseline[trained_feats].to_dict()], columns=trained_feats)
    price_before = float(np.expm1(model.predict(baseline_vec))[0])

    # ==========================================================
    # 8️⃣ Función objetivo: maximizar ganancia neta
    # ==========================================================
    m.setObjective(price - cost_expr, GRB.MAXIMIZE)

    # ==========================================================
    # 9️⃣ Resolver el modelo
    # ==========================================================
    m.Params.OutputFlag = 1   # mostrar log de Gurobi
    m.optimize()

    # ==========================================================
    # 🔟 Reporte y resultados
    # ==========================================================
    if m.SolCount > 0:
        price_after = float(price.X)
        deltas = {c: x[c].X - float(baseline[c]) for c in trained_feats}
        spent  = float(cost_expr.getValue())
        profit = price_after - price_before - spent
        roi    = (profit / spent) if spent > 0 else float('nan')

        print("\n=== 💎 RESULTADOS: CASA ÓPTIMA (PROFIT) ===")
        print(f"Precio antes   : {price_before:,.0f}")
        print(f"Precio después : {price_after:,.0f}")
        print(f"Gasto total    : {spent:,.0f} (presupuesto {budget:,.0f})")
        print(f"Ganancia neta  : {profit:,.0f} | ROI: {roi:,.2f}\n")

        print("Cambios sugeridos:")
        for c in trained_feats:
            delta = deltas[c]
            if abs(delta) > 1e-6:
                print(f" - {c:20s}: {x[c].X:10.3f}  (Δ={delta:+.3f})")

        return {
            "price_before": price_before,
            "price_after": price_after,
            "spent": spent,
            "profit": profit,
            "roi": roi,
            "changes": deltas
        }

    else:
        print("❌ No se encontró solución factible.")
        return None
