"""
optimization.py
----------------
Integra el modelo predictivo (Linear o XGB) con Gurobi para
determinar la "Casa Óptima" dentro de un presupuesto.

Uso típico:
    from src.optimization import optimize_house
    optimize_house(model, X, y, baseline, trained_feats, trained_stats)
"""

import gurobipy as gp
from gurobi_ml import add_predictor_constr
from gurobipy import GRB
import numpy as np
import pandas as pd


def optimize_house(
    model,
    X,
    y_log,
    trained_feats,
    trained_stats,
    baseline_idx=0,
    budget=200_000,
    objective_mode="profit",
    roi_min=0.10,
    pwl_k=25
):
    """
    Ejecuta la optimización de diseño de vivienda usando un modelo predictivo
    previamente entrenado (Linear o XGB).

    Parámetros
    ----------
    model : modelo entrenado (Pipeline con LinearRegression o XGBRegressor)
    X : pd.DataFrame
        Dataset con las features originales.
    y_log : pd.Series
        Variable objetivo en log (Sale_Price_Log).
    trained_feats : list
        Nombres de las features usadas para entrenar el modelo.
    trained_stats : pd.DataFrame
        Estadísticos de las features (q05, median, q95).
    baseline_idx : int
        Índice de la casa base que se optimizará.
    budget : float
        Presupuesto máximo disponible para mejoras.
    objective_mode : str
        "price", "profit" o "roi".
    roi_min : float
        ROI mínimo (solo si objective_mode = "roi").
    pwl_k : int
        Número de segmentos para aproximar exp() con PWL.
    """

    print("\n=== 🧱 OPTIMIZACIÓN CASA ÓPTIMA ===")

    # 1️⃣ Baseline
    n = len(X)
    idx = baseline_idx if 0 <= baseline_idx < n else 0
    baseline = X.iloc[idx].astype(float)

    pred_log = float(model.predict(baseline.to_frame().T)[0])
    price_pred = float(np.expm1(pred_log))
    price_real = float(np.expm1(y_log.iloc[idx]))

    print(f"\n🏠 Casa {idx} seleccionada como baseline")
    print(f"Precio predicho: {price_pred:,.0f} | Precio real: {price_real:,.0f}")

    # 2️⃣ Costos base y “room to grow”
    default_costs = {
        "Gr_Liv_Area":     200,
        "Total_Bsmt_SF":    80,
        "Garage_Area":      60,
        "Garage_Cars":   17000,
        "Full_Bath":     25000,
        "Fireplaces":      6000,
        "Overall_Qual":   20000,
        "KitchenQual_ord": 8000,
        "GarageFinish_ord":4000,
    }

    room = {
        "Gr_Liv_Area":     400.0,
        "Total_Bsmt_SF":   300.0,
        "Garage_Area":     250.0,
        "Garage_Cars":       1.0,
        "Full_Bath":         1.0,
        "Fireplaces":        1.0,
        "Overall_Qual":      1.0,
        "KitchenQual_ord":   1.0,
        "GarageFinish_ord":  1.0,
    }

    q95 = trained_stats["q95"]

    # 3️⃣ Construcción de límites y costos
    bounds, costs = {}, {}
    for f in trained_feats:
        base = float(baseline.get(f, X[f].median()))
        ub_room = base + room.get(f, 0.0)
        ub_q95  = float(q95.get(f, base))
        lb = base
        ub = max(lb, min(ub_room, ub_q95))
        bounds[f] = (lb, ub)
        costs[f]  = float(default_costs.get(f, 0.0))

    # 4️⃣ Crear modelo de optimización
    m = gp.Model("casa_optima")
    int_like = {"Garage_Cars","Full_Bath","Fireplaces","Overall_Qual","KitchenQual_ord","GarageFinish_ord"}
    x = {}

    for c in trained_feats:
        lb, ub = bounds[c]
        if c in int_like:
            lb_i, ub_i = int(np.floor(lb)), int(np.ceil(ub))
            if ub_i < lb_i:
                ub_i = lb_i
            x[c] = m.addVar(lb=lb_i, ub=ub_i, vtype=GRB.INTEGER, name=c)
        else:
            x[c] = m.addVar(lb=float(lb), ub=float(ub), vtype=GRB.CONTINUOUS, name=c)

    cost_expr = gp.quicksum(costs[c] * (x[c] - float(baseline[c])) for c in trained_feats)
    m.addConstr(cost_expr <= float(budget), name="Budget")

    x_df = pd.DataFrame([[x[c] for c in trained_feats]], columns=trained_feats)
    y_pred_log = m.addVar(name="y_pred_log")

    add_predictor_constr(
        gp_model=m,
        predictor=model,
        input_vars=x_df,
        output_vars=y_pred_log
    )

    # 5️⃣ Función objetivo
    if objective_mode == "price":
        m.setObjective(y_pred_log, GRB.MAXIMIZE)
    else:
        ymin, ymax = np.percentile(y_log, [1, 99])
        ymin, ymax = float(np.clip(ymin, -1e2, 1e2)), float(np.clip(ymax, -1e2, 1e2))
        if ymax <= ymin:
            ymin, ymax = 10.5, 13.5

        xs = np.linspace(ymin, ymax, pwl_k).tolist()
        ys = [float(np.clip(np.expm1(v), -1e9, 1e9)) for v in xs]

        price = m.addVar(name="price")
        m.addGenConstrPWL(y_pred_log, price, xs, ys, name="log_to_price")

        baseline_vec = pd.DataFrame([baseline[trained_feats].to_dict()], columns=trained_feats)
        price_before = float(np.expm1(model.predict(baseline_vec))[0])

        if objective_mode == "profit":
            m.setObjective(price - cost_expr, GRB.MAXIMIZE)
        elif objective_mode == "roi":
            m.addConstr(price - price_before >= roi_min * cost_expr, name="ROImin")
            m.setObjective(price - cost_expr, GRB.MAXIMIZE)
        else:
            raise ValueError("objective_mode debe ser 'price', 'profit' o 'roi'.")

    # 6️⃣ Resolver
    m.Params.OutputFlag = 1
    m.optimize()

    # 7️⃣ Reporte de resultados
    if m.SolCount > 0:
        baseline_vec = pd.DataFrame([baseline[trained_feats].to_dict()], columns=trained_feats)
        price_before_rep = float(np.expm1(model.predict(baseline_vec))[0])
        price_after_rep  = float(np.expm1(y_pred_log.X)) if objective_mode == "price" else float(price.X)

        deltas = {c: x[c].X - float(baseline[c]) for c in trained_feats}
        spent  = float(cost_expr.getValue())
        profit = price_after_rep - price_before_rep - spent
        roi    = (profit / spent) if spent > 0 else float('nan')

        print("\n=== 💎 SOLUCIÓN CASA ÓPTIMA ===")
        print(f"Precio antes   : {price_before_rep:,.0f}")
        print(f"Precio después : {price_after_rep:,.0f}")
        print(f"Gasto total    : {spent:,.0f} (presupuesto {budget:,.0f})")
        print(f"Ganancia neta  : {profit:,.0f} | ROI: {roi:,.2f}\n")

        print("Cambios sugeridos:")
        for c in trained_feats:
            delta = deltas[c]
            if abs(delta) > 1e-6:
                print(f" - {c:20s}: {x[c].X:10.3f}  (Δ={delta:+.3f})")

        return {
            "price_before": price_before_rep,
            "price_after": price_after_rep,
            "spent": spent,
            "profit": profit,
            "roi": roi,
            "changes": deltas
        }

    else:
        print("❌ No se encontró solución factible.")
        return None
