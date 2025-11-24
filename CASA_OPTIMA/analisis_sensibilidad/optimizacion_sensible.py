import gurobipy as gp                    
from gurobi_ml import add_predictor_constr  
from gurobipy import GRB
import numpy as np
import sys

import pandas as pd
# try:
#     # When src is imported as a package (from src.optimization import ...)
#     from .check_feasible_houses import check_house_feasibility
# except Exception:
#     # Fallback for running the module directly (python src/optimization.py)
#     from check_feasible_houses import check_house_feasibility

def optimize_house(
    model,
    X,
    y_log,
    trained_feats,
    trained_stats,
    baseline_idx = 0,
    budget = None,
    pwl_k = 25,
    zero = False,
    LON = None,
    LAT = None,
    Lot_Area = None,
    # ========== PARÁMETROS DE SENSIBILIDAD ==========
    cost_multiplier = 1.0,
    custom_costs = None,
    market_factor = 1.0,
    lot_coverage_limit = 0.65,
    second_floor_ratio = 1.2,
    basement_ratio = 1.2,
    garage_ratio = 1.0,
    max_changes = None,
    enabled_variables = None
):
    """
    Optimización de Casa con Análisis de Sensibilidad Completo
  
    PARÁMETROS BÁSICOS:
    -------------------
    model : XGBRegressor
        Modelo entrenado de precio de vivienda
    X : pd.DataFrame
        Dataset con features de entrenamiento
    y_log : pd.Series
        Precio de venta en log (SalePrice_Log)
    trained_feats : list
        Variables usadas para entrenar
    trained_stats : pd.DataFrame
        Estadísticos de features (min, max, median, q05, q95)
    baseline_idx : int, default=0
        Índice de la casa base a optimizar
    budget : float
        Presupuesto máximo disponible
    pwl_k : int, default=25
        Segmentos para aproximar exp()
    zero : bool, default=False
        Si True, construye casa desde cero
    LON, LAT, Lot_Area : float
        Coordenadas y área (solo si zero=True)
    
    PARÁMETROS DE SENSIBILIDAD:
    ---------------------------
    cost_multiplier : float, default=1.0
        Multiplicador GLOBAL de costos
        Ejemplos:
            1.0  = costos normales
            1.15 = inflación 15%
            0.85 = descuento 15%
        
    custom_costs : dict, default=None
        Costos PERSONALIZADOS por variable
        Si se especifica, SOBRESCRIBE cost_multiplier
        Ejemplo:
            {'First_Flr_SF': 200, 'Kitchen_Qual': 25000}
        Permite variar UN SOLO costo sin afectar los demás
    
    market_factor : float, default=1.0
        Factor de ajuste del mercado
        Ejemplos:
            0.85 = recesión (-15%)
            1.20 = boom (+20%)
            1.35 = burbuja (+35%)
    
    lot_coverage_limit : float, default=0.65
        % máximo de ocupación del lote
        Ejemplo: 0.70 permite cubrir hasta 70%
    
    second_floor_ratio : float, default=1.2
        Tamaño máximo 2do piso vs 1er piso
        Ejemplo: 1.5 permite 2do piso hasta 1.5x
    
    basement_ratio : float, default=1.2
        Tamaño máximo sótano vs 1er piso
    
    garage_ratio : float, default=1.0
        Tamaño máximo garage vs 1er piso
    
    max_changes : int, default=None
        Número máximo de variables a modificar
        Si None, no hay límite
        Ejemplos:
            3  = solo modificar 3 atributos
            5  = solo modificar 5 atributos
    
    enabled_variables : list, default=None
        Lista específica de variables permitidas
        Si se especifica, SOLO estas pueden cambiar
        Ejemplo:
            ['First_Flr_SF', 'Kitchen_Qual', 'Garage_Area']
    
    RETORNA:
    --------
    dict con:
        - price_before, price_after, spent, profit, roi
        - changes: dict con cambios por variable
        - cost_breakdown: dict con costos por variable
        - final_house: DataFrame con casa optimizada
        - num_changes: cantidad de variables modificadas
        - sensitivity_params: parámetros usados
    """

    print("\nOPTIMIZACIÓN CASA ÓPTIMA")

    
    # Selección de la vivienda base
    n = len(X)

    if zero == True:
        baseline = { "First_Flr_SF": 0,
                "Second_Flr_SF": 0,
                "Year_Built": 2025,
                "Exter_Qual": 0,
                "Total_Bsmt_SF": 0,
                "Lot_Area": Lot_Area if Lot_Area is not None else X["Lot_Area"].median(),
                "Garage_Area": 0,
                "Kitchen_Qual": 0,
                "Fireplaces": 0,
                "Year_Remod_Add": 2025,
                "Sale_Condition_Normal": 1,   # Normal
                "Longitude": LON if LON is not None else X["Longitude"].median(),             
                "Full_Bath": 0,
                "Bsmt_Qual": 0,
                "Latitude": LAT if LAT is not None else X["Latitude"].median(),              
                "Bsmt_Exposure": 0,
                "TotRms_AbvGrd": 0,
                "Half_Bath": 0,
                "Heating_QC": 0,
                "Garage_Finish": 0,
                "Garage_Cond": 0,
                "Wood_Deck_SF": 0,
                "Open_Porch_SF": 0,
                "Bsmt_Full_Bath": 0,
                "House_Style_One_Story": 1,
                "Sale_Type_New": 1,
                "Bedroom_AbvGr": 0,
                "Garage_Qual": 0,
                "Kitchen_AbvGr": 0,
                "Overall_Cond": 10,
                "Central_Air_Y": 0}
        
        baseline = pd.DataFrame([baseline])
        baseline = baseline.astype(float)
        pred_log = float(model.predict(baseline)[0])



    else:
        idx = baseline_idx if 0 <= baseline_idx < n else 0
        baseline = X.iloc[idx].astype(float)
        pred_log = float(model.predict(baseline.to_frame().T)[0])

    
    # Verificar factibilidad de la casa baseline
    #is_feasible, violations = check_house_feasibility(baseline)
    #if not is_feasible:
        #violated_constraints = [k for k, v in violations.items() if v]
        #print(f"\nERROR: La casa {idx} NO es factible para optimización")
        #print(f"Restricciones violadas: {', '.join(violated_constraints)}")
        #print("Seleccione una casa factible para continuar.")
        #return None, None, None, None

    # Predicciones iniciales (en log y en valor real)
    price_pred = float(np.expm1(pred_log))        # Valor estimado (modelo)

    if zero == False:
        price_real = float(np.expm1(y_log.iloc[idx])) # Valor real en datos originales

        print(f"\nCasa {idx} seleccionada como baseline")
        print(f"Precio predicho: {price_pred:,.0f} | Precio real: {price_real:,.0f}")

    else:
        print(f"\nConstruyendo casa desde cero en las coordenadas ({float(baseline['Longitude']):.6f}, {float(baseline['Latitude']):.6f})")
        print(f"Presupuesto disponible: ${budget:,.0f}")

    #Parametros
    cocina_promedio = 161 
    baño_promedio = 45 # 70% de 65
    habitacion_promedio = 120 # 70% de 172
    M_grande = 1e6 

    # ========== CONFIGURACIÓN DE COSTOS ==========
    # Costos base por unidad de mejora (USD)
    # Variables no accionables tienen costo 0
    default_costs = {
        "First_Flr_SF":        151,   # costo por pie² en el primer piso
        "Second_Flr_SF":       203,   # segundo piso es más caro estructuralmente
        "Year_Built":            0,   # costo por "año equivalente" de antigüedad 
        "Exter_Qual":          450,  # mejorar calidad exterior
        "Total_Bsmt_SF":        100,   # costo por pie² adicional en sótano
        "Lot_Area":             0,   # costo por pie² de terreno
        "Garage_Area":        70,  # costo por pie² ampliar garage
        "Garage_Cond":         20,  # mejorar condición del garage  (pie²)
        "Garage_Finish":       20,  # mejorar terminaciones del garage (pie²)
        "Garage_Qual":         25,  #mejorar calidad del garage (pie²)
        "Kitchen_Qual":        126 * cocina_promedio ,  # mejorar calidad de cocina (por nivel)
        "Kitchen_AbvGr":        60000,  # construir cocina nueva
        "Fireplaces":          6942,  # agregar chimenea
        "Year_Remod_Add":          0,   # no accionable (directamente)
        "Sale_Condition_Normal": 0,   # no accionable
        "Sale_Type_New":         0,      # no accionable
        "Longitude":              0,  # ubicación fija 
        "Latitude":               0,  # ubicación fija 
        "Full_Bath":          10386,  # agregar baño completo
        "Half_Bath":         6150,  # agregar medio baño (sin ducha)
        "Bsmt_Qual":           20,  # mejorar calidad del sótano (por nivel) (pie²)
        "Bsmt_Exposure":       4000,  # agregar ventanas o acceso al exterior del sotano
        "TotRms_AbvGrd":      40000,  # costo por agregar una habitación
        "Bedroom_AbvGr":       46000,   # costo por agregar un dormitorio
        "House_Style_One_Story": 0,  # categórica (no accionable directamente)
        "Heating_QC":          2500,  # mejorar calidad del sistema de calefacción (por nivel)
        "Bsmt_Full_Bath":      18500,   # costo por agregar un baño en el sotano
        "Open_Porch_SF":       111,     #costo por agregar pie² del pórtico
        "Wood_Deck_SF":           10,   #costo por agregar pie² de la terraza
        "Overall_Cond":         0, #no accionable DIRECTAMENTE
        "Central_Air_Y":       3900  # instalar aire acondicionado central
    }
    
    # APLICAR COSTOS SEGÚN PARÁMETROS DE SENSIBILIDAD
    if custom_costs is not None:
        # Opción 1: Multiplicadores personalizados por variable
        # custom_costs contiene factores multiplicadores, no costos absolutos
        costs_to_use = default_costs.copy()
        for var, factor in custom_costs.items():
            if var in costs_to_use:
                costs_to_use[var] = default_costs[var] * factor
        print(f"\n📌 Multiplicadores aplicados: {custom_costs}")
    else:
        # Opción 2: Aplicar multiplicador global
        costs_to_use = {k: v * cost_multiplier for k, v in default_costs.items()}
        if cost_multiplier != 1.0:
            print(f"\n📌 Multiplicador de costos global: {cost_multiplier:.2%}")
    
    # "Room to grow": máximos incrementos posibles por variable 
    room = {
        "First_Flr_SF":       M_grande,  # pies² adicionales en primer piso
        "Second_Flr_SF":      M_grande,  # pies² adicionales en segundo piso
        "Year_Built":           0,  
        "Exter_Qual":         M_grande,  # subir un nivel de calidad (TA→Gd→Ex)
        "Total_Bsmt_SF":      M_grande,
        "Lot_Area":                 0,  
        "Garage_Area":          M_grande,
        "Garage_Cond":          M_grande,  # mejorar condición del garage
        "Kitchen_Qual":         M_grande,  # subir un nivel (TA→Gd→Ex)
        "Kitchen_AbvGr":        M_grande,  # categórica (no accionable directamente)
        "Fireplaces":           M_grande,  # agregar chimenea   
        "Year_Remod_Add":       M_grande,  # remodelar o actualizar hasta 3 "años equivalentes"
        "Sale_Condition_Normal": 0,  # no se modifica
        "Longitude":            0,  # ubicación fija
        "Latitude":            0,  # ubicación fija
        "Full_Bath":            M_grande,
        "Half_Bath":         M_grande,  # agregar medio baño
        "Bsmt_Qual":            M_grande,
        "Bsmt_Exposure":        0,   # no se modifica
        "Sale_Type_New":        0,  # no se modifica
        "TotRms_AbvGrd":        M_grande,  # agregar una habitación adicional
        "House_Style_One_Story": M_grande,  
        "Heating_QC":          M_grande,  # mejorar calidad del sistema de calefacción
        "Garage_Finish":        M_grande,
        "Wood_Deck_SF":         M_grande,
        "Open_Porch_SF":        M_grande,
        "Bsmt_Full_Bath":       M_grande,
        "Bedroom_AbvGr":        M_grande,
        "Garage_Qual":          M_grande,
        "Overall_Cond":         M_grande,
        "Central_Air_Y":        1
    }

    # Construcción de límites y costos 
    bounds= {} 
    costs = {}
    #Estas variables no se modifican, no tienen bounds
    ignore_max = {"Year_Built", "Year_Remod_Add", "Longitude", "Latitude", "Lot_Area", 
                  "Sale_Condition_Normal", "Sale_Type_New", "Bsmt_Exposure"}
    # Máximo valor que puede tomar es el maximo de la base de datos
    maximo = trained_stats["max"]

    for f in trained_feats:
        base = float(baseline.get(f, X[f].median()))

        # CASA NUEVA: calidad FIJA de cocina en nivel 4 (Buena)
        if zero and f == "Kitchen_Qual":
            lb = ub = 4
            bounds[f] = (lb, ub)
            costs[f] = 0 # no se paga por calidad en construcción nueva
            continue

        #  Basement/Garage calidad FIJA en nivel 4 (Buena) si existen, 0 si no existen
        if zero and f in ["Bsmt_Qual", "Garage_Cond", "Garage_Qual"]:
            lb = 0
            ub = 4      
            bounds[f] = (lb, ub)
            costs[f] = 0  # no se paga por calidad en construcción nueva
            continue

        # FILTRAR VARIABLES SI SE ESPECIFICA enabled_variables
        if enabled_variables is not None and f not in enabled_variables:
            # Variable bloqueada: no puede cambiar
            lb = ub = base
            bounds[f] = (lb, ub)
            costs[f] = 0
            continue

        # Si es la variable binaria que indica si la casa es de 1 piso,
        # permitir que pase de 1 → 0 si se construye segundo piso.
        if f == "House_Style_One_Story":
            lb = 0
            ub = 1

        else:
            # Lógica normal: las variables NO pueden disminuir
            lb = base
            ub_room = base + room.get(f, 0)
            ub_max = float(maximo.get(f, base))

            if f in ignore_max:
                ub = max(lb, ub_room)
            else:
                ub = max(lb, min(ub_room, ub_max))

        bounds[f] = (lb, ub)
        costs[f] = float(costs_to_use.get(f, 0))


    # Creación del modelo de optimización Gurobi
    m = gp.Model("casa_optima_profit")

    # Variables que deben ser enteras
    int_like = {"Exter_Qual", "Overall_Cond", "Garage_Cond","Kitchen_Qual", 
    "Kitchen_AbvGr", "Full_Bath", "Half_Bath", "Fireplaces", "BsmntQual", "Bsmt_Exposure",
    "TotRms_AbvGrd", "Heating_QC", "Year_Built", "Year_Remod_Add", "Bsmt_Qual", 
    "House_Style_One_Story","Garage_Finish", "Bsmt_Full_Bath", "Bedroom_AbvGr", 
    "Garage_Qual", "Overall_Cond", "Sale_Condition_Normal", "Sale_Type_New", "Central_Air_Y"}

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
    

    
    # ========== RESTRICCIONES BÁSICAS (AJUSTABLES) ==========

    # 2. Primer piso + garage no puede superar % del área del lote (Restricciones Legales)
    m.addConstr(x["First_Flr_SF"] + x["Garage_Area"] + x["Open_Porch_SF"]
                 + x["Wood_Deck_SF"] <= x["Lot_Area"] * lot_coverage_limit, 
                 name="LotArea_limit")

    # 3. Segundo piso no puede superar X veces el primer piso (ajustable)
    m.addConstr(x["Second_Flr_SF"] <= second_floor_ratio * x["First_Flr_SF"], 
                name="SecondFloor_limit")

    # 4. Si la casa es de un solo piso, el segundo piso debe ser 0
    m.addConstr(x["Second_Flr_SF"] <= M_grande * (1 - x["House_Style_One_Story"]),
                 name="one_story_no_second_floor")

    # 5. El garage es mas chico que el primer piso (ajustable)
    m.addConstr(x["Garage_Area"] <= x["First_Flr_SF"] * garage_ratio, 
                name="Garage_size_limit")

    # 6. El tamaño del sótano no puede superar el primer piso (ajustable)
    m.addConstr(x["Total_Bsmt_SF"] <= x["First_Flr_SF"] * basement_ratio, 
                name="Basement_size_limit")

    # 7. El numero de baños no puede superar el número de habitaciones
    m.addConstr(x["Full_Bath"] + x["Half_Bath"] <= x["TotRms_AbvGrd"] + 1 , name="Baths_limit")

    # 8. No pueden haber mas baños completos que dormitorios
    m.addConstr(x["Full_Bath"] <= x["Bedroom_AbvGr"] , name="FullBath_limit")

    # 9. El numero de baños half bath no puede ser mayor a baños completos
    m.addConstr(x["Half_Bath"] <= x["Full_Bath"] , name="HalfBath_limit")

    # 10. El numero de fireplaces no puede ser mayor a 2
    m.addConstr(x["Fireplaces"] <=2 , name="Fireplaces_limit")

    # 11. EL año de remodelación es igual a el año actual
    m.addConstr(x["Year_Remod_Add"] == 2025 , name="Remodeling_year_limit")

    # 12. Las ampliaciones en SF deben ser significativas (No de 1 pie²)
    min_delta = {"First_Flr_SF": 100, "Second_Flr_SF": 100 , "Total_Bsmt_SF": 100}

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

    # 16. Los SF construidos deben ser suficientes para que quepan los atributos seleccionados
    m.addConstr( x["First_Flr_SF"] + x["Second_Flr_SF"] >= 
                x["Full_Bath"] * baño_promedio + x["Bsmt_Full_Bath"] * baño_promedio + 
                x["Kitchen_AbvGr"] * cocina_promedio + x["TotRms_AbvGrd"] * habitacion_promedio + 100 , 
                name="sf_min")

    # 17. Las cocinas deben tener una calidad asociada distinta de 0
    m.addConstr( x["Kitchen_AbvGr"] <= M_grande * x["Kitchen_Qual"], name = "calidad_cocina")

    # 18. Basement debe tener sus atributos asociados a su existencia

    # Variable binaria: B = 1 si hay basement, 0 si no
    B = m.addVar(vtype=GRB.BINARY, name="Basement_Binary")

    # Área mínima real para considerar basement existente
    eps_b = 100
    m.addConstr(x["Total_Bsmt_SF"] >= eps_b * B, name="basement_min_area_if_exists")
    m.addConstr(x["Total_Bsmt_SF"] <= M_grande * B, name="basement_zero_if_not_exists")

    # Detectar si la casa originalmente tenía basement
    had_basement = 1 if float(baseline["Total_Bsmt_SF"]) > 0 else 0

    # Variable que indica si se está construyendo basement nuevo
    B_new = m.addVar(vtype=GRB.BINARY, name="Basement_New")
    m.addConstr(B_new >= B - had_basement, name="Basement_new_flag")

    if zero:
        # Casa nueva → calidad fija
        m.addConstr(x["Bsmt_Qual"] == 4 * B, name="BsmtQual_zero_new")
        m.addConstr(x["Bsmt_Exposure"] == 1 * B, name="BsmtExposure_zero_new")
    else:
        # Si basement es nuevo → calidad fija en 4 y no se cobra
        m.addConstr(
            x["Bsmt_Qual"] == 4 * B_new + baseline["Bsmt_Qual"] * (1 - B_new),
            name="BsmtQual_if_new"
        )

        m.addConstr(
            x["Bsmt_Exposure"] == 1 * B_new + baseline["Bsmt_Exposure"] * (1 - B_new),
            name="BsmtExposure_if_new"
        )

        # Si existía basement antes, permitir mejorar calidad (sí cobra)
        if had_basement == 1:
            m.addConstr(x["Bsmt_Qual"] >= baseline["Bsmt_Qual"], name="BsmtQual_can_improve")
            m.addConstr(x["Bsmt_Exposure"] >= baseline["Bsmt_Exposure"], name="BsmtExposure_can_improve")
        
        # Si NO hay basement (B = 0), los atributos deben ser exactamente 0
        m.addConstr(x["Bsmt_Qual"] == 0 * (1 - B) + x["Bsmt_Qual"] * B, name="BsmtQual_zero_if_no_b")



    # 19. Baño de basement debe caber en este
    m.addConstr(x["Total_Bsmt_SF"] * 0.75 >= baño_promedio * x["Bsmt_Full_Bath"], name="bsmt_bath_min_size")


    # 20. Garage debe tener sus atributos asociados a su existencia

    # Variable binaria: G = 1 si hay garage, 0 si no
    G = m.addVar(vtype=GRB.BINARY, name="Garage_Binary")

    # Área mínima realista para considerar un garage existente
    eps_g = 100  
    m.addConstr(x["Garage_Area"] >= eps_g * G, name="garage_min_area_if_exists")
    m.addConstr(x["Garage_Area"] <= M_grande * G, name="garage_zero_if_not_exists")

    # Detectar si la casa originalmente tenía garage
    had_garage = 1 if float(baseline["Garage_Area"]) > 0 else 0

    # Variable que indica si se está construyendo GARAGE NUEVO
    G_new = m.addVar(vtype=GRB.BINARY, name="Garage_New")
    m.addConstr(G_new >= G - had_garage, name="Garage_new_flag")

    if zero:
        # CASA NUEVA: calidades fijas en 4
        m.addConstr(x["Garage_Cond"]   == 4 * G, name="GarageCond_zero_new")
        m.addConstr(x["Garage_Qual"]   == 4 * G, name="GarageQual_zero_new")
        m.addConstr(x["Garage_Finish"] == 4 * G, name="GarageFinish_zero_new")

    else:

        # Si el garage es NUEVO → calidad fija en 4
        m.addConstr(
            x["Garage_Cond"] == 4 * G_new + baseline["Garage_Cond"] * (1 - G_new),
            name="GarageCond_if_new"
        )

        m.addConstr(
            x["Garage_Qual"] == 4 * G_new + baseline["Garage_Qual"] * (1 - G_new),
            name="GarageQual_if_new"
        )

        m.addConstr(
            x["Garage_Finish"] == 4 * G_new + baseline["Garage_Finish"] * (1 - G_new),
            name="GarageFinish_if_new"
        )

        # Si existía garage antes → permitir mejorar calidad
        if had_garage == 1:
            for v in ["Garage_Cond", "Garage_Qual", "Garage_Finish"]:
                m.addConstr(
                    x[v] >= baseline[v],
                    name=f"{v}_can_improve"
            )

        # Si NO hay garage (G = 0) → atributos deben ser 0
        m.addConstr(x["Garage_Cond"]   <= M_grande * G, name="GarageCond_zero_if_no_garage")
        m.addConstr(x["Garage_Qual"]   <= M_grande * G, name="GarageQual_zero_if_no_garage")
        m.addConstr(x["Garage_Finish"] <= M_grande * G, name="GarageFinish_zero_if_no_garage")

    # 21. La casa debe tener almenos 1 dormitorio
    m.addConstr( 1 <= x["Bedroom_AbvGr"] , name="min_bedroom")

    # 22. Overall_Cond sube si se mejora la casa
    # Baseline quality score
    if zero == False:
        baseline_quality_avg = (
            baseline["Exter_Qual"] +
            baseline["Kitchen_Qual"] +
            baseline["Bsmt_Qual"] +
            baseline["Heating_QC"] +
            baseline["Garage_Cond"] +
            baseline["Garage_Qual"]) / 6
        # Variable auxiliar
        Quality_Score = m.addVar(lb=baseline_quality_avg, ub=9, name="Quality_Score")
        # Relación del score con las calidades optimizadas
        m.addConstr(
            Quality_Score == (
                x["Exter_Qual"] +
                x["Kitchen_Qual"] +
                x["Bsmt_Qual"] +
                x["Heating_QC"] +
                x["Garage_Cond"] +
                x["Garage_Qual"]
            ) / 6 )
        # Parámetro de sensibilidad 
        alpha = 0.5
        # No disminuir condición
        m.addConstr(x["Overall_Cond"] >= baseline["Overall_Cond"])
        # Ajuste lineal: Overall sube si la calidad promedio sube
        m.addConstr( x["Overall_Cond"] <= baseline["Overall_Cond"] + alpha * (Quality_Score - baseline_quality_avg))


    # ========== RESTRICCIÓN DE CARDINALIDAD (MÁXIMO NÚMERO DE CAMBIOS) ==========
    if max_changes is not None and max_changes > 0:
        print(f"\n📌 Restricción: máximo {max_changes} variables pueden cambiar")
        
        # Crear variables binarias para detectar cambios
        change_indicators = {}
        eps_change = 1e-4  # tolerancia para detectar cambio
        
        for c in trained_feats:
            base_val = float(baseline[c])
            lb, ub = bounds[c]
            
            # Solo crear indicador si la variable PUEDE cambiar
            if ub > lb + eps_change:
                change_indicators[c] = m.addVar(vtype=GRB.BINARY, name=f"changed_{c}")
                
                # Si cambió hacia arriba, el indicador debe ser 1
                m.addConstr(x[c] - base_val <= M_grande * change_indicators[c], 
                           name=f"change_up_{c}")
                
                # También detectar cambios hacia abajo (aunque normalmente no disminuyen)
                m.addConstr(base_val - x[c] <= M_grande * change_indicators[c], 
                           name=f"change_down_{c}")
        
        # Limitar número total de cambios
        if len(change_indicators) > 0:
            m.addConstr(gp.quicksum(change_indicators.values()) <= max_changes, 
                       name="max_changes_limit")
    
    # ========== CÁLCULO DE COSTOS ==========
    # Calcular Costos estándar (excepto Garage y Basement que tienen lógica especial)
    cost_standard = gp.quicksum( costs_to_use[c] * (x[c] - float(baseline[c])) for c in trained_feats
    if c not in ["Garage_Qual", "Garage_Cond", "Garage_Finish", "Bsmt_Qual"])

    # Calcular Costos Garage (dependen del área actual)
    area_garage = float(baseline["Garage_Area"])

    c_g_qual   = costs_to_use["Garage_Qual"]
    c_g_cond   = costs_to_use["Garage_Cond"]
    c_g_finish = costs_to_use["Garage_Finish"]

    cost_g_qual   = c_g_qual   * area_garage * (x["Garage_Qual"]   - baseline["Garage_Qual"])   * (1 - G_new)
    cost_g_cond   = c_g_cond   * area_garage * (x["Garage_Cond"]   - baseline["Garage_Cond"])   * (1 - G_new)
    cost_g_finish = c_g_finish * area_garage * (x["Garage_Finish"] - baseline["Garage_Finish"]) * (1 - G_new)

    # Calcular Costos Basement
    c_b_qual = costs_to_use["Bsmt_Qual"]
    cost_b_qual = c_b_qual * float(baseline["Total_Bsmt_SF"]) * (x["Bsmt_Qual"] - float(baseline["Bsmt_Qual"])) * (1 - B_new)

    # Restricción de presupuesto total
    cost_expr = (cost_standard + cost_g_qual + cost_g_cond + cost_g_finish + cost_b_qual)

    m.addConstr(cost_expr <= float(budget), name="Budget")
    m.addConstr(cost_expr >= 0, name="Non_negative_cost")

    # Conexión con el modelo predictivo (Gurobi + ML)

    x_df = pd.DataFrame([[x[c] for c in trained_feats]], columns=trained_feats)
    y_pred_log = m.addVar(name="y_pred_log")  # variable para el precio predicho (log)

    add_predictor_constr(
        gp_model=m,          # modelo de Gurobi
        predictor=model,     # modelo de ML (XGB)
        input_vars=x_df,     # variables de entrada (features)
        output_vars=y_pred_log  # salida (log-precio)
    )

    # Conversión del log-precio a precio real (PWL)
    ymin, ymax = np.percentile(y_log, [1, 99])
    ymin, ymax = float(np.clip(ymin, -1e2, 1e2)), float(np.clip(ymax, -1e2, 1e2))
    if ymax <= ymin:
        ymin, ymax = 10.5, 13.5  

    xs = np.linspace(ymin, ymax, pwl_k).tolist()
    ys = [float(np.clip(np.expm1(v), -1e9, 1e9)) for v in xs]

    price_base = m.addVar(name="price_base")
    m.addGenConstrPWL(y_pred_log, price_base, xs, ys, name="log_to_price")
    
    # ========== APLICAR FACTOR DE MERCADO ==========
    price = m.addVar(name="price")
    m.addConstr(price == price_base * market_factor, name="market_adjustment")
    
    if market_factor != 1.0:
        print(f"📌 Factor de mercado aplicado: {market_factor:.2%}")

    # Precio antes de mejorar (baseline)
    if zero == True:
        price_before = 0
    
    else:
        baseline_vec = baseline[trained_feats].astype(float)
        pred_before = model.predict(baseline_vec.values.reshape(1, -1))
        price_before = float(np.expm1(pred_before)[0]) * market_factor  # Aplicar factor también al baseline

    # Función objetivo: maximizar ganancia neta
    m.setObjective(price - price_before - cost_expr, GRB.MAXIMIZE)

    # Resolver el modelo
    m.Params.OutputFlag = 1  
    m.optimize()

    # Resultados
    if m.SolCount > 0:
        price_after = float(price.X)
        deltas = {c: x[c].X - float(baseline[c]) for c in trained_feats}
        spent  = float(cost_expr.getValue())
        profit = price_after - price_before - spent
        roi    = (profit / spent) if spent > 0 else float('nan')
        
        # Contar cambios significativos
        num_changes = sum(1 for v in deltas.values() if abs(v) > 1e-6)

        print("\n" + "="*70)
        print(" RESULTADOS: CASA ÓPTIMA")
        print("="*70)
        print(f"Precio antes   : ${price_before:,.0f}")
        print(f"Precio después : ${price_after:,.0f}")
        print(f"Gasto total    : ${spent:,.0f} / ${budget:,.0f}")
        print(f"Ganancia neta  : ${profit:,.0f}")
        print(f"ROI            : {roi:.2%}")
        print(f"Variables modificadas: {num_changes}")
        
        # Mostrar parámetros de sensibilidad si fueron modificados
        if cost_multiplier != 1.0 or custom_costs or market_factor != 1.0 or max_changes:
            print("\n📊 Parámetros de sensibilidad:")
            if cost_multiplier != 1.0:
                print(f"   • Multiplicador costos: {cost_multiplier:.0%}")
            if custom_costs:
                print(f"   • Costos personalizados: {len(custom_costs)} variables")
            if market_factor != 1.0:
                print(f"   • Factor de mercado: {market_factor:.0%}")
            if max_changes:
                print(f"   • Máximo cambios: {max_changes}")
        print()

        print("Cambios sugeridos:")
        print("-" * 70)
        print("-" * 70)

        cost_breakdown = {}
        for c in trained_feats:
            if c in ["Garage_Qual", "Garage_Cond", "Garage_Finish", "Bsmt_Qual"]:
                continue  

            delta = deltas[c]
            unit_cost = costs.get(c, 0.0)
            total_cost = delta * unit_cost

            if abs(delta) > 1e-6: #and unit_cost > 0
                cost_breakdown[c] = total_cost
                print(f"{c:25s} {delta:+10.3f} {unit_cost:15,.0f} {total_cost:15,.0f}")

        area_garage = float(baseline["Garage_Area"])

        c_garage_qual = costs_to_use["Garage_Qual"]
        c_garage_cond = costs_to_use["Garage_Cond"]
        c_garage_finish = costs_to_use["Garage_Finish"]

        delta_garage_qual = deltas["Garage_Qual"]
        delta_garage_cond = deltas["Garage_Cond"]
        delta_garage_finish = deltas["Garage_Finish"]

        total_garage_qual = c_garage_qual * area_garage * delta_garage_qual
        total_garage_cond = c_garage_cond * area_garage * delta_garage_cond
        total_garage_finish = c_garage_finish * area_garage * delta_garage_finish

        if abs(delta_garage_qual) > 1e-6:
            cost_breakdown["Garage_Qual"] = total_garage_qual
            print(f"{'Garage_Qual':25s} {delta_garage_qual:+10.3f} {c_garage_qual*area_garage:15,.0f} {total_garage_qual:15,.0f}")
        if abs(delta_garage_cond) > 1e-6:
            cost_breakdown["Garage_Cond"] = total_garage_cond
            print(f"{'Garage_Cond':25s} {delta_garage_cond:+10.3f} {c_garage_cond*area_garage:15,.0f} {total_garage_cond:15,.0f}")
        if abs(delta_garage_finish) > 1e-6:
            cost_breakdown["Garage_Finish"] = total_garage_finish
            print(f"{'Garage_Finish':25s} {delta_garage_finish:+10.3f} {c_garage_finish*area_garage:15,.0f} {total_garage_finish:15,.0f}")

        area_basement = float(baseline["Total_Bsmt_SF"])
        c_basement_qual = costs_to_use["Bsmt_Qual"]
        delta_basement_qual = deltas["Bsmt_Qual"]
        total_basement_qual = c_basement_qual * area_basement * delta_basement_qual

        if abs(delta_basement_qual) > 1e-6:
            cost_breakdown["Bsmt_Qual"] = total_basement_qual
            print(f"{'Bsmt_Qual':25s} {delta_basement_qual:+10.3f} {c_basement_qual*area_basement:15,.0f} {total_basement_qual:15,.0f}")

        print("-" * 70)
        total_cost_sum = sum(cost_breakdown.values())
        print(f"{'TOTAL':25s} {'':>10s} {'':>15s} {total_cost_sum:15,.0f}\n")

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
            "final_house": df_final_house.to_dict(orient="records"),
            "num_changes": num_changes,
            "sensitivity_params": {
                "cost_multiplier": cost_multiplier,
                "market_factor": market_factor,
                "lot_coverage_limit": lot_coverage_limit,
                "second_floor_ratio": second_floor_ratio,
                "basement_ratio": basement_ratio,
                "garage_ratio": garage_ratio,
                "max_changes": max_changes,
                "custom_costs": list(custom_costs.keys()) if custom_costs else None,
                "enabled_variables": enabled_variables
            }
        }

        
    else:
        result_info = {
            "status": "infeasible",
            "violated_constraints": [],
            "active_constraints": [],
            "involved_variables": [],
            "message": "No se encontró solución factible."
        }

        try:
            # Calcular IIS (conjunto mínimo de restricciones conflictivas)
            m.computeIIS()
            result_info["violated_constraints"] = [
                c.ConstrName for c in m.getConstrs() if c.IISConstr
            ]
            result_info["involved_variables"] = [
                v.VarName for v in m.getVars() if v.IISLB or v.IISUB
            ]
        except Exception as e:
            result_info["message"] += f" (Error al detectar restricciones conflictivas: {str(e)})"

        # Restricciones con slack ≈ 0 (activas o casi violadas)
        try:
            result_info["active_constraints"] = [
                {"name": c.ConstrName, "slack": c.Slack}
                for c in m.getConstrs() if abs(c.Slack) < 1e-6
            ]
        except Exception:
            pass

        return result_info


