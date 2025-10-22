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
    pwl_k=25,
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

    baseline = {
    "First_Flr_SF": 0,
    "Second_Flr_SF": 0,
    "Year_Built": 2025,
    "Exter_Qual": 0,
    "Total_Bsmt_SF": 0,
    "Lot_Area": 12000,
    "Garage_Cars": 0,
    "Kitchen_Qual": 0,
    "Fireplaces": 0,
    "Year_Remod_Add": 2025,
    "Sale_Condition_Normal": 0,
    "Longitude": -93.62,
    "Full_Bath": 0,
    "Bsmt_Qual": 0,
    "Latitude": 42.05,
    "Bsmt_Exposure": 0,
    "TotRms_AbvGrd": 0,
    "Half_Bath": 0,
    "Heating_QC": 0,
    "Garage_Finish": 0,
    "Garage_Cond": 0,
    "Wood_Deck_SF": 0,
    "Open_Porch_SF": 0,
    "Bsmt_Full_Bath": 0,
    "House_Style_One_Story": 0,
    "Sale_Type_New": 0,
    "Bedroom_AbvGr": 0,
    "Garage_Qual": 0,
    "Kitchen_AbvGr": 0,
    "Pool_Area": 0,
    "Overall_Cond": 0
}
    baseline = pd.Series(baseline)

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
        "First_Flr_SF":        151,   # costo por pie² en el primer piso
        "Second_Flr_SF":       203,   # segundo piso es más caro estructuralmente
        "Year_Built":            0,   # costo por "año equivalente" de antigüedad 
        "Exter_Qual":          70,  # mejorar calidad exterior
        "Total_Bsmt_SF":        100,   # costo por pie² adicional en sótano
        "Lot_Area":             0,   # costo por pie² de terreno
        "Garage_Cars":        19463,  # agregar espacio de estacionamiento
        "Garage_Cond":         3000,  # mejorar condición del garage 
        "Garage_Finish":       3000,
        "Garage_Qual":         6000,
        "Kitchen_Qual":        8000,  # mejorar calidad de cocina (por nivel)
        "Kitchen_AbvGr":        45000,  # construir cocina nueva
        "Fireplaces":          6942,  # agregar chimenea
        "Year_Remod_Add":          0,   # costo asociado a remodelación reciente
        "Sale_Condition_Normal": 0,   # categórica (no accionable directamente)
        "Sale_Type_New":         0,
        "Longitude":              0,  # ubicación fija 
        "Latitude":               0,  # ubicación fija 
        "Full_Bath":          10386,  # agregar baño completo
        "Half_Bath":         6150,  # agregar medio baño (sin ducha)
        "Bsmt_Qual":           5000,  # mejorar calidad del sótano (por nivel)
        "Bsmt_Exposure":       4000,  # agregar ventanas o acceso al exterior del sotano
        "TotRms_AbvGrd":      10000,  # costo por agregar una habitación
        "Bedroom_AbvGr":       0,
        "House_Style_One_Story": 0,  # categórica (no accionable directamente)
        "Heating_QC":          4000,  # mejorar calidad del sistema de calefacción (por nivel)
        "Pool_Area":           8000,  # agregar piscina
        "Bsmt_Full_Bath":      6000,
        "Open_Porch_SF":       200,
        "Wood_Deck_SF":           200
    }
    
   

    # "Room to grow": máximos incrementos posibles por variable
    M_grande = 1e6  
    room = {
        "First_Flr_SF":       M_grande,  # pies² adicionales en primer piso
        "Second_Flr_SF":      M_grande,  # pies² adicionales en segundo piso
        "Year_Built":           0,  
        "Exter_Qual":         M_grande,  # subir un nivel de calidad (TA→Gd→Ex)
        "Total_Bsmt_SF":      M_grande,
        "Lot_Area":                 0,  
        "Garage_Cars":          M_grande,
        "Garage_Cond":          M_grande,  # mejorar condición del garage
        "Kitchen_Qual":         M_grande,  # subir un nivel (TA→Gd→Ex)
        "Kitchen_AbvGr":        M_grande,  # categórica (no accionable directamente)
        "Fireplaces":           M_grande,
        "Year_Remod_Add":       M_grande,  # remodelar o actualizar hasta 3 "años equivalentes"
        "Sale_Condition_Normal": 0,  # no se modifica
        "Longitude":            0,  # ubicación fija
        "Latitude":            0,  # ubicación fija
        "Full_Bath":            M_grande,
        "Half_Bath":         M_grande,  # agregar medio baño
        "Bsmt_Qual":            M_grande,
        "Bsmt_Exposure":        M_grande,
        "Sale_Type_New": 0,  # no se modifica
        "TotRms_AbvGrd":        M_grande,  # agregar una habitación adicional
        "House_Style_One_Story": 0,  # no se modifica
        "Heating_QC":          M_grande,  # mejorar calidad del sistema de calefacción
        "Pool_Area":            M_grande, 
        "Garage_Finish":        M_grande,
        "Wood_Deck_SF":         M_grande,
        "Open_Porch_SF":        M_grande,
        "Bsmt_Full_Bath":       M_grande,
        "Bedroom_AbvGr":        M_grande,
        "Garage_Qual":          M_grande
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
    int_like = {"Exter_Qual", "Overall_Cond", "Garage_Cars", "Garage_Cond","Kitchen_Qual", 
    "Kitchen_AbvGr", "Full_Bath", "Half_Bath", "Fireplaces", "BsmntQual",
    "Bsmt_Exposure","TotRms_AbvGrd", "Heating_QC" }

    # Variables de decisión 
    x = {}

    # Creación dinámica de variables
    for c in trained_feats:
        lb, ub = bounds[c]
        if c in int_like:
            # Variables enteras: se redondean los límites
            lb_i, ub_i = int(np.floor(lb)), int(np.ceil(ub))
            if ub_i < lb_i:
                ub_i = lb_i
            x[c] = m.addVar(lb=lb_i, ub=ub_i, vtype=GRB.INTEGER, name=c)
        else:
            # Variables continuas (m², pies²)
            x[c] = m.addVar(lb=float(lb), ub=float(ub), vtype=GRB.CONTINUOUS, name=c)
    

    #Parametros
    espacio_por_auto = 260 # pies² por auto adicional
    M_sqr_feet = 1e6  # gran número para restricciones tipo "if"
    cocina_promedio = 300 #BUSCAR INFO
    baño_promedio = 150 #BUSCAR INFO
    habitacion_promedio = 200 #BUSCAR INFO

    # Restricciones básicas

    # 1. Restricción de presupuesto total
    cost = gp.quicksum(costs[c] * (x[c] - float(baseline[c])) for c in trained_feats)
    m.addConstr(cost <= float(budget), name= "Budget")
    m.addConstr(cost >= 0, name= "Non_negative_cost")  # no gastar "negativo"

    espacio_por_auto = 260 # pies² por auto adicional
    M_sqr_feet = 1e6  # gran número para restricciones tipo "if"

    # 2. Primer piso mas garage no puede superar el area del lote
    m.addConstr(x["First_Flr_SF"] + x["Garage_Cars"] * espacio_por_auto + x["Open_Porch_SF"] + x["Wood_Deck_SF"] + x["Pool_Area"] <= x["Lot_Area"], name="LotArea_limit")

    # 3. Segundo piso no puede superar el primer piso
    m.addConstr(x["Second_Flr_SF"] <= x["First_Flr_SF"] , name="SecondFloor_limit")

    # 4. Si la casa es de un solo piso, el segundo piso debe ser 0
    m.addConstr(x["Second_Flr_SF"] <= M_sqr_feet * (1 - baseline["House_Style_One_Story"]), name="HouseStyle_1Story_limit")

    # 5. El garage es mas chico que el primer piso
    m.addConstr(x["Garage_Cars"] * espacio_por_auto <= x["First_Flr_SF"], name="Garage_size_limit")

    # 6. El tamaño del sótano no puede superar el primer piso
    m.addConstr(x["Total_Bsmt_SF"] <= x["First_Flr_SF"], name="Basement_size_limit")

    # 7. El numero de baños completos no puede superar el número de habitaciones
    m.addConstr(x["Full_Bath"] + x["Half_Bath"] <= x["TotRms_AbvGrd"] + 1 , name="Baths_limit")

    # 8. No pueden haber mas baños completos que habitaciones
    m.addConstr(x["Full_Bath"] <= x["TotRms_AbvGrd"] , name="FullBath_limit")

    # 9. El numero de baños half bath no puede ser mayor a baños completos
    m.addConstr(x["Half_Bath"] <= x["Full_Bath"] , name="HalfBath_limit")

    # 10. El numero de fireplaces no puede superar el número de habitaciones
    m.addConstr(x["Fireplaces"] <= x["Full_Bath"] + x["Half_Bath"], name="Fireplaces_limit")

    # 11. EL año de remodelación es igual a el año actual
    m.addConstr(x["Year_Remod_Add"] == 2025 , name="Remodeling_year_limit")

    # 12. Las ampliaciones en SF deben ser significativas (No de 1 pie²)
    min_delta = {"First_Flr_SF": 40, "Second_Flr_SF": 40, "Total_Bsmt_SF": 20, "Pool_Area": 20}

    ampliaciones = {}
    for v, min_d in min_delta.items():
        # Crear variable binaria: 1 si hay ampliación, 0 si no se modifica
        ampliaciones[v] = m.addVar(vtype=GRB.BINARY, name=f"A_{v}")

        # Si se amplía, debe aumentar al menos min_delta
        m.addConstr( x[v] - baseline[v] >= min_d * ampliaciones[v], name=f"{v}_min_delta")

        # Si no se amplía, el cambio debe ser 0
        m.addConstr(x[v] - baseline[v] <= M_grande * ampliaciones[v], name=f"A_{v}_activation")

    # 13. Debe haber al menos una cocina
    m.addConstr( 1 <= x["Kitchen_AbvGr"] , name="kitchen_min")

    # 14. Debe haber al menos una habitacion
    m.addConstr( 1 <= x["TotRms_AbvGrd"] , name="rooms_min")

    # 15. Debe haber al menos un baño
    m.addConstr( 1 <= x["Full_Bath"] , name="bath_min")

    m.addConstr( x["First_Flr_SF"] + x["Second_Flr_SF"] + x["Total_Bsmt_SF"] >= x["Full_Bath"] * baño_promedio + x["Kitchen_AbvGr"] * cocina_promedio + x["TotRms_AbvGrd"] * habitacion_promedio + 100 , name="sf_min")

    m.addConstr( x["Kitchen_AbvGr"] <= M_grande * x["Kitchen_Qual"], name = "calidad_cocina")

    m.addConstr( x["Total_Bsmt_SF"] <= M_grande * x["Bsmt_Exposure"], name = "bsmt_exposure")

    m.addConstr( x["Total_Bsmt_SF"] <= M_grande * x["Bsmt_Qual"], name = "bsmt_quality")
    
    m.addConstr( x["Total_Bsmt_SF"] >= baño_promedio * x["Bsmt_Full_Bath"], name = "bsmt_bath")

    # ==========================================================
# 🚗 Coherencia del garaje: si hay autos, deben existir calidades (>0)
# y si no hay garage (0 autos), todas deben ser 0
# ==========================================================

# Variable binaria de existencia de garage
G = m.addVar(vtype=GRB.BINARY, name="Garage_Binary")

# Activación: si hay garage (al menos 1 auto), G = 1
# Si no hay garage (0 autos), G = 0
m.addConstr(x["Garage_Cars"] * espacio_por_auto <= M_grande * G, name="g_activation")
m.addConstr(x["Garage_Cars"] >= G, name="g_min_if_exists")

# Atributos asociados al garage
garage_related = ["Garage_Finish", "Garage_Cond", "Garage_Qual"]

# Si existe garage (G=1), las calidades deben ser >0
# Si no hay garage (G=0), las calidades deben ser =0
for v in garage_related:
    # Si no hay garage → atributo = 0
    m.addConstr(x[v] <= M_grande * G, name=f"{v}_upper_if_garage")
    # Si hay garage → atributo >= 1
    m.addConstr(x[v] >= G, name=f"{v}_lower_if_garage")





    # Conexión con el modelo predictivo (Gurobi + ML)

    x_df = pd.DataFrame([[x[c] for c in trained_feats]], columns=trained_feats)
    y_pred_log = m.addVar(name="y_pred_log")  # variable para el precio predicho (log)

    add_predictor_constr(
        gp_model=m,          # modelo de Gurobi
        predictor=model,     # modelo de ML (XGB o Linear)
        input_vars=x_df,     # variables de entrada (features)
        output_vars=y_pred_log  # salida (log-precio)
    )

    # Conversión del log-precio a precio real (PWL)
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

    # Función objetivo: maximizar ganancia neta
    m.setObjective(price - cost, GRB.MAXIMIZE)

    # Resolver el modelo
    m.Params.OutputFlag = 1  
    m.optimize()

    # Resultados
    if m.SolCount > 0:
        price_after = float(price.X)
        deltas = {c: x[c].X - float(baseline[c]) for c in trained_feats}
        spent  = float(cost.getValue())
        profit = price_after - price_before - spent
        roi    = (profit / spent) if spent > 0 else float('nan')

        print("\n RESULTADOS: CASA ÓPTIMA")
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

        print("\nAtributos finales de la casa optimizada:")
        print("-" * 70)

        # Crear diccionario con valores finales y baseline
        final_values = {c: x[c].X for c in trained_feats}
        baseline_values = {c: float(baseline[c]) for c in trained_feats}

        # Crear DataFrame comparativo
        df_final = pd.DataFrame({
            "Inicial": baseline_values,
            "Optimizado": final_values,
            "Δ Cambio": {c: final_values[c] - baseline_values[c] for c in trained_feats}
        })

        # Mostrar todo el DataFrame
        print(df_final.to_string(float_format=lambda x: f"{x:,.2f}"))
        print("-" * 70)

        # Guardar DataFrame para retorno o análisis posterior
        df_final_house = pd.DataFrame([final_values])
        

        return {
            "price_before": price_before,
            "price_after": price_after,
            "spent": spent,
            "profit": profit,
            "roi": roi,
            "changes": deltas,
            "cost_breakdown": cost_breakdown,
            "final_house": df_final_house
        }

        
    else:
        print("No se encontró solución factible.")
        return None

