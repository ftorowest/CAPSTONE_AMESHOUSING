"""
Análisis de Casas Factibles en el Dataset
==========================================
Este script verifica qué casas del dataset Ames Housing cumplen con TODAS las
restricciones del modelo de optimización, identificando cuáles son factibles
como punto de partida para optimización.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Parámetros constantes (deben coincidir con optimization.py)
ESPACIO_POR_AUTO = 260  # pies² por auto adicional
COCINA_PROMEDIO = 161   # pies² por cocina
BAÑO_PROMEDIO = 45      # pies² por baño (70% de 65)
HABITACION_PROMEDIO = 120  # pies² por habitación (70% de 172)
M_SQFT = 1e6  # gran número para restricciones


def check_house_feasibility(house):
    """
    Verifica si una casa cumple con todas las restricciones del modelo de optimización.

    Parameters:
    house : pd.Series
        Serie con las características de la casa a verificar

    Returns:
    tuple: (is_feasible, violations_dict)
        is_feasible: bool indicando si la casa es factible
        violations_dict: dict con las restricciones violadas y sus valores
    """
    violations = {}

    # RESTRICCIÓN 2: Primer piso + garage + porches <= 80% del lote
    area_construida = (house['First_Flr_SF'] + 
                       house.get('Garage_Area', 0) + 
                       house.get('Open_Porch_SF', 0) + 
                       house.get('Wood_Deck_SF', 0) + 
                       house.get('Pool_Area', 0))
    violations['lot_area'] = area_construida > house['Lot_Area'] * 0.8    # RESTRICCIÓN 3: Segundo piso <= Primer piso
    violations['second_floor'] = house['Second_Flr_SF'] > 1.2 * house['First_Flr_SF']

    # RESTRICCIÓN 4: Si es de un piso, segundo piso = 0
    if 'House_Style_One_Story' in house.index:
        violations['one_story'] = (house['House_Style_One_Story'] == 1) & (house['Second_Flr_SF'] > 0)
    else:
        violations['one_story'] = False

    # RESTRICCIÓN 5: Garage <= Primer piso
    violations['garage_size'] = house.get('Garage_Area', 0) > house['First_Flr_SF']

    # RESTRICCIÓN 6: Basement <= Primer piso
    violations['basement_size'] = house['Total_Bsmt_SF'] > house['First_Flr_SF'] * 1.2

    # RESTRICCIÓN 7: Baños totales <= Habitaciones + 1
    violations['bath_rooms'] = (house['Full_Bath'] + house['Half_Bath']) > (house['TotRms_AbvGrd'] + 1)

    # RESTRICCIÓN 8: Baños completos <= Dormitorios
    violations['fullbath_bedroom'] = house['Full_Bath'] > house['Bedroom_AbvGr']

    # RESTRICCIÓN 9: Medios baños <= Baños completos
    violations['halfbath_fullbath'] = house['Half_Bath'] > house['Full_Bath']

    # RESTRICCIÓN 10: Chimeneas <= Baños totales
    violations['fireplaces'] = house['Fireplaces'] > (house['Full_Bath'] + house['Half_Bath'])

    # RESTRICCIÓN 13: Cocina >= 1
    violations['kitchen_min'] = house['Kitchen_AbvGr'] < 1

    # RESTRICCIÓN 14: Habitaciones >= 1
    violations['rooms_min'] = house['TotRms_AbvGrd'] < 1

    # RESTRICCIÓN 15: Baños completos >= 1
    violations['bath_min'] = house['Full_Bath'] < 1

    # RESTRICCIÓN 16: SF suficientes para atributos
    area_disponible = (house['First_Flr_SF'] + house['Second_Flr_SF'])
    area_necesaria = (house['Full_Bath'] * BAÑO_PROMEDIO +
                      house.get('Bsmt_Full_Bath', 0) * BAÑO_PROMEDIO +
                      house['Kitchen_AbvGr'] * COCINA_PROMEDIO +
                      house['TotRms_AbvGrd'] * HABITACION_PROMEDIO + 100)
    violations['sf_min'] = area_disponible < area_necesaria

    # RESTRICCIÓN 22: Dormitorios >= 1
    violations['bedroom_min'] = house['Bedroom_AbvGr'] < 1

    # Verificar si hay violaciones
    n_violations = sum(violations.values())
    is_feasible = n_violations == 0

    return is_feasible, violations