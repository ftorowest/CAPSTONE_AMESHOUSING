import pandas as pd
import numpy as np


#Carga el dataset y devuelve X, y listos para entrenamiento.
#Aplica limpieza de columnas según importancia (ya definida manualmente).

def load_and_prepare(path_csv: str):
    df = pd.read_csv(path_csv)
    df.columns = df.columns.str.strip()

    # Verificar variable objetivo
    assert "Sale_Price_Log" in df.columns, "Falta la columna Sale_Price_Log en el dataset."

    # Variable objetivo
    y = df["Sale_Price_Log"].copy()


    features = ["First_Flr_SF", "Second_Flr_SF","Year_Built","Exter_Qual","Total_Bsmt_SF",
    "Lot_Area", "Garage_Area", "Kitchen_Qual", "Fireplaces", "Year_Remod_Add","Sale_Condition_Normal",
    "Longitude", "Full_Bath","Bsmt_Qual", "Latitude", "Bsmt_Exposure","TotRms_AbvGrd", "Half_Bath",
    "Heating_QC", "Garage_Finish", "Garage_Cond", "Wood_Deck_SF", "Open_Porch_SF","Bsmt_Full_Bath",
    "House_Style_One_Story", "Sale_Type_New", "Bedroom_AbvGr", "Garage_Qual","Kitchen_AbvGr",
    "Overall_Cond"]   
    
    X = df[features].copy()

    # Rellenamos valores faltantes
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    print(f" Dataset procesado: {X.shape[0]} filas, {X.shape[1]} features.")
    return X, y 

