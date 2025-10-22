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

    # Variables a eliminar 
    drop_cols = [
        "Sale_Price", "Sale_Price_Log", "Overall_Qual",
        # Variables irrelevantes (importancia ≈ 0)
        "Neighborhood_Sawyer_West","Exterior_1st_VinylSd","Mas_Vnr_Type_BrkFace",
        "MS_SubClass_Two_Story_1945_and_Older","Condition_1_PosN","Neighborhood_Northridge",
        "Exterior_2nd_Stucco","Exterior_2nd_MetalSd","Sale_Condition_Alloca","Enclosed_Porch",
        "Neighborhood_Mitchell","Mas_Vnr_Type_None","Bsmt_Cond","Land_Contour_Low",
        "Sale_Condition_AdjLand","Land_Contour_HLS","Lot_Frontage","Electrical_SBrkr",
        "Exterior_2nd_HdBoard","Bldg_Type_TwoFmCon","Roof_Matl_CompShg",
        "Exterior_2nd_Plywood","MS_Zoning_Residential_High_Density","Exterior_1st_Plywood",
        "Condition_1_PosA","Lot_Shape_Slightly_Irregular","Condition_1_RRAe",
        "MS_SubClass_Two_and_Half_Story_All_Ages","Neighborhood_Stone_Brook",
        "Mas_Vnr_Type_Stone","Neighborhood_Green_Hills","Lot_Config_CulDSac",
        "MS_SubClass_Duplex_All_Styles_and_Ages","Garage_Type_CarPort","Heating_GasW",
        "Sale_Type_ConLD","Sale_Type_Oth","Roof_Style_Gable","Garage_Type_Basment",
        "House_Style_Two_Story","Lot_Config_Inside","Lot_Shape_Irregular",
        "Sale_Type_ConLw","Neighborhood_Gilbert","MS_SubClass_Two_Family_conversion_All_Styles_and_Ages",
        "Exterior_1st_Stucco","Low_Qual_Fin_SF","Exterior_2nd_Wd Sdng","Exterior_1st_MetalSd",
        "MS_SubClass_Split_Foyer","Roof_Style_Gambrel","Heating_Grav","Bsmt_Half_Bath",
        "Neighborhood_Timberland","Neighborhood_Northridge_Heights","Garage_Type_BuiltIn",
        "Exterior_1st_CemntBd","Neighborhood_Bloomington_Heights","MS_SubClass_Two_Story_PUD_1946_and_Newer",
        "Foundation_Stone","Neighborhood_Meadow_Village","Heating_Wall","House_Style_Two_and_Half_Unf",
        "Neighborhood_Veenker","Foundation_Slab","Exterior_2nd_BrkFace","Lot_Config_FR3",
        "Misc_Val","Three_season_porch","Electrical_FuseF","Neighborhood_College_Creek",
        "Lot_Shape_Moderately_Irregular","Exterior_1st_HdBoard","Bldg_Type_TwnhsE",
        "Exterior_2nd_Wd Shng","MS_SubClass_Split_or_Multilevel","Exterior_2nd_ImStucc",
        "Condition_1_RRNn","Exterior_1st_WdShing","House_Style_SFoyer","Roof_Matl_WdShake",
        "MS_Zoning_A_agr","MS_Zoning_I_all","MS_SubClass_PUD_Multilevel_Split_Level_Foyer",
        "MS_SubClass_One_and_Half_Story_Unfinished_All_Ages","MS_SubClass_One_Story_with_Finished_Attic_All_Ages",
        "MS_SubClass_One_and_Half_Story_PUD_All_Ages","Neighborhood_Hayden_Lake","Neighborhood_Greens",
        "Neighborhood_Blueste","Neighborhood_Northpark_Villa","Neighborhood_Briardale",
        "Neighborhood_Landmark","Bldg_Type_Twnhs","House_Style_Two_and_Half_Fin","Condition_1_RRAn",
        "Exterior_1st_ImStucc","Exterior_1st_CBlock","Exterior_1st_AsphShn","Exterior_1st_BrkComm",
        "Roof_Matl_WdShngl","Roof_Style_Shed","Roof_Matl_Roll","Roof_Matl_Metal","Roof_Style_Mansard",
        "Roof_Matl_Membran","Condition_1_RRNe","Bldg_Type_Duplex","House_Style_One_and_Half_Unf",
        "Exterior_1st_Stone","Exterior_2nd_AsphShn","Exterior_2nd_Brk Cmn","Mas_Vnr_Type_CBlock",
        "Exterior_2nd_Stone","Exterior_2nd_Other","Exterior_2nd_PreCast","Exterior_1st_PreCast",
        "Exterior_2nd_CBlock","Foundation_Wood","Heating_OthW","Electrical_Mix","Electrical_FuseP",
        "Electrical_Unknown","Sale_Type_Con","Sale_Type_CWD","Sale_Type_ConLI",
        "Garage_Type_More_Than_Two_Types", "Sale_Type_VWD", "Garage_Type_No_Garage"
    ]


    features = [
    "First_Flr_SF",
    "Second_Flr_SF",
    "Year_Built",
    "Exter_Qual",
    "Total_Bsmt_SF",
    "Lot_Area",
    "Kitchen_Qual",
    "Overall_Cond",
    "Lot_Area",
    
    "Fireplaces",
    "Sale_Condition_Normal",
    "Longitude",
    "Bsmt_Qual",
    "Year_Remod_Add",
    "Full_Bath",
    "TotRms_AbvGrd",
    "Garage_Finish",
    "Bsmt_Exposure",
    "Garage_Cond",
    "Latitude",
    "House_Style_One_Story",
    "Half_Bath",
    "Heating_QC",
    "Open_Porch_SF",
    "Bsmt_Full_Bath",
    "Sale_Type_New",
    "Wood_Deck_SF",
    "Kitchen_AbvGr",
    "Pool_Area",
    "Garage_Qual",
    "Bsmt_Full_Bath",
    "Bedroom_AbvGr"
]   
    X = df[features].copy()
    # Rellenamos valores faltantes
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    print(f" Dataset procesado: {X.shape[0]} filas, {X.shape[1]} features.")
    return X, y

