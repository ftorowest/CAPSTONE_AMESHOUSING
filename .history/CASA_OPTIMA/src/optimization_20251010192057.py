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
    pwl_k=25
):
    """
    Parámetros
  
    model : objeto entrenado (XGBRegressor)
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

    print("\nOPTIMIZACIÓN CASA ÓPTIMA")

    # Selección de la vivienda base
    n = len(X)
    idx = baseline_idx if 0 <= baseline_idx < n else 0
    baseline = X.iloc[idx].astype(float)

    # Predicciones iniciales (en log y en valor real)
    pred_log = float(model.predict(baseline.to_frame().T)[0])
    price_pred = float(np.expm1(pred_log))        # Valor estimado (modelo)
    price_real = float(np.expm1(y_log.iloc[idx])) # Valor real en datos originales

    print(f"\nCasa {idx} seleccionada como baseline")
    print(f"Precio predicho: {price_pred:,.0f} | Precio real: {price_real:,.0f}")

    # Definición de costos base y “espacio para mejorar”
    # Costo unitario aproximado de aumentar cada variable (en USD)
    # Variables no accionables se le impone un costo 0
    default_costs = {
        "First_Flr_SF":        200,   # costo por pie² en el primer piso
        "Second_Flr_SF":       220,   # segundo piso es más caro estructuralmente
        "Year_Built":            0,   # costo por "año equivalente" de antigüedad 
        "Exter_Qual":          70,  # mejorar calidad exterior
        "Total_Bsmt_SF":        0,   # costo por pie² adicional en sótano
        "Lot_Area":             0,   # costo por pie² de terreno
        "Overall_Cond":       12000,  # mejorar condición general
        "Garage_Cars":        17000,  # agregar espacio de estacionamiento
        "Garage_Cond":         3000,  # mejorar condición del garage
        "Kitchen_Qual":        8000,  # mejorar calidad de cocina (por nivel)
        "Kitchen_AbvGr":        45000,  # construir cocina nueva
        "Fireplaces":          6000,  # agregar chimenea
        "Year_Remod_Add":          0,   # costo asociado a remodelación reciente
        "Sale_Condition_Normal": 0,   # categórica (no accionable directamente)
        "Longitude":              0,  # ubicación fija 
        "Latitude":               0,  # ubicación fija 
        "Full_Bath":          25000,  # agregar baño completo
        "Half_Bath":         15000,  # agregar medio baño (sin ducha)
        "Bsmt_Qual":           5000,  # mejorar calidad del sótano (por nivel)
        "Bsmt_Exposure":       4000,  # agregar ventanas o acceso al exterior del sotano
        "TotRms_AbvGrd":      10000,  # costo por agregar una habitación
        "House_Style_One_Story": 0,  # categórica (no accionable directamente)
        "Heating_QC":          4000,  # mejorar calidad del sistema de calefacción (por nivel)
        "Pool_Area":           8000,  # agregar piscina
    }

    # "Room to grow": máximos incrementos posibles por variable
    M_grande = 1e6  
    room = {
        "First_Flr_SF":       400,  # pies² adicionales en primer piso
        "Second_Flr_SF":      400,  # pies² adicionales en segundo piso
        "Year_Built":           0,  
        "Exter_Qual":         5,  # subir un nivel de calidad (TA→Gd→Ex)
        "Total_Bsmt_SF":      300,
        "Lot_Area":                 0,  
        "Overall_Cond":         M_grande,  # subir un nivel de condición general
        "Garage_Cars":          M_grande,
        "Garage_Cond":          M_grande,  # mejorar condición del garage
        "Kitchen_Qual":         M_grande,  # subir un nivel (TA→Gd→Ex)
        "Kitchen_AbvGr":        M_grande,  # categórica (no accionable directamente)
        "Fireplaces":           M_grande,
        "Year_Remod_Add":       M_grande ,  # remodelar o actualizar hasta 3 "años equivalentes"
        "Sale_Condition_Normal":0,  # no se modifica
        "Longitude":            0,  # ubicación fija
        "Longitude":            0,  # ubicación fija
        "Full_Bath":            M_grande,
        "Half_Bath":         M_grande,  # agregar medio baño
        "Bsmt_Qual":            M_grande,
        "Bsmt_Exposure":        M_grande,
        "Sale_Condition_Normal": 0,  # no se modifica
        "Latitude":             0,
        "TotRms_AbvGrd":        M_grande,  # agregar una habitación adicional
        "House_Style_One_Story": 0,  # no se modifica
        "Heating_QC":          M_grande,  # mejorar calidad del sistema de calefacción
        "Pool_Area":            M_grande,  # no se modifica
    }


    # Construcción de límites y costos 
    bounds= {} 
    costs = {}
    #Estas variables no se modifican, no tienen bounds
    ignore_max = {"Year_Built", "Year_Remod_Add", "Longitude", "Latitude"}
    # Máximo valor que puede tomar es el maximo de la base de datos
    maximo = trained_stats["max"]


    for f in trained_feats:
        base = float(baseline.get(f, X[f].median()))

        # Upper Bound 
        ub_room = base + room.get(f, 0)
        ub_max = float(maximo.get(f, base))

        # Lower Bound
        lb = base

        if f in ignore_max:
            # Usa solo room (o base si room=0)
            ub = max(lb, ub_room)
        else:
            # Usa el menor entre room y max
            ub = max(lb, min(ub_room, ub_max))

        #Añade los límites y costos a los diccionarios
        bounds[f] = (lb, ub)
        costs[f] = float(default_costs.get(f,0))

    # Creación del modelo de optimización Gurobi
    m = gp.Model("casa_optima_profit")

    # Variables que deben ser enteras
    int_like = {"Exter_Qual", "Overall_Cond", "Garage_Cars", "Kitchen_Qual", 
    "Kitchen_AbvGr", "Full_Bath", "Half_Bath", "Fireplaces", "Garage_Cond", "BsmntQual"
    , "Bsmt_Exposure"}

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

    # Restricción de presupuesto total
    cost_expr = gp.quicksum(costs[c] * (x[c] - float(baseline[c])) for c in trained_feats)
    m.addConstr(cost_expr <= float(budget), name="Budget")
    m.addConstr(cost_expr >= 0, name="NonNegativeCost")  # no gastar "negativo"

    espacio_por_auto = 260 # pies² por auto adicional
    M_sqr_feet = 1e6  # gran número para restricciones tipo "if"

    # 🧩 --- Espacio para restricciones adicionales ---
    # Ejemplos posibles:
    # m.addConstr(x["Full_Bath"] <= x["Bedroom_AbvGr"], name="Baths_limit")
    # m.addConstr(x["Garage_Cars"] <= 3, name="Garage_limit")
    # m.addConstr(x["Overall_Qual"] >= x["KitchenQual_ord"], name="Quality_relation")

    # primer piso mas garage no puede superar el area del lote
    m.addConstr(x["First_Flr_SF"] + x["Garage_Cars"] * espacio_por_auto + x["Open_Porch_SF"] + x["Wood_Deck_SF"] + x["Pool_Area"] <= x["Lot_Area"], name="LotArea_limit")

    #segundo piso no puede superar el primer piso
    m.addConstr(x["Second_Flr_SF"] <= x["First_Flr_SF"] , name="SecondFloor_limit")

    # si la casa es de un solo piso, el segundo piso debe ser 0
    m.addConstr(x["Second_Flr_SF"] <= M_sqr_feet * (1 - baseline["House_Style_One_Story"]), name="HouseStyle_1Story_limit")

    # El garage es mas chico que el primer piso
    m.addConstr(x["Garage_Cars"] * espacio_por_auto <= x["First_Flr_SF"], name="Garage_size_limit")

    # El tamaño del sótano no puede superar el primer piso
    m.addConstr(x["Total_Bsmt_SF"] <= x["First_Flr_SF"], name="Basement_size_limit")

    # El numero de baños completos no puede superar el número de habitaciones
    m.addConstr(x["Full_Bath"] + x["Half_Bath"] <= x["TotRms_AbvGrd"] + 1 , name="Baths_limit")

    #no pueden haber mas baños completos que habitaciones
    m.addConstr(x["Full_Bath"] <= x["TotRms_AbvGrd"] , name="FullBath_limit")

    #El numero de baños half bath no puede ser mayor a baños completos
    m.addConstr(x["Half_Bath"] <= x["Full_Bath"] , name="HalfBath_limit")

    # El numero de fireplaces no puede superar el número de habitaciones
    m.addConstr(x["Fireplaces"] <= x["Full_Bath"] + x["Half_Bath"], name="Fireplaces_limit")

    # EL año de remodelación es igual a el año actual
    m.addConstr(x["Year_Remod_Add"] == 2025 , name="Remodeling_year_limit")





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
        print("-" * 70)
        print(f"{'Variable':25s} {'Δ Valor':>10s} {'Costo unitario':>15s} {'Costo total':>15s}")
        print("-" * 70)

        cost_breakdown = {}
        for c in trained_feats:
            delta = deltas[c]
            unit_cost = costs.get(c, 0.0)
            total_cost = delta * unit_cost
            if abs(delta) > 1e-6 and unit_cost > 0:
                cost_breakdown[c] = total_cost
                print(f"{c:25s} {delta:+10.3f} {unit_cost:15,.0f} {total_cost:15,.0f}")

        print("-" * 70)
        print(f"{'TOTAL':25s} {'':>10s} {'':>15s} {sum(cost_breakdown.values()):15,.0f}\n")

        return {
            "price_before": price_before,
            "price_after": price_after,
            "spent": spent,
            "profit": profit,
            "roi": roi,
            "changes": deltas,
            "cost_breakdown": cost_breakdown  # <- 🔹 diccionario con desglose por mejora
        }

    else:
        print("❌ No se encontró solución factible.")
        return None

