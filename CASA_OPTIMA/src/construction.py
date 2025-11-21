import pandas as pd
import numpy as np
from optimization import optimize_house
from joblib import load
from preprocessing import load_and_prepare


def construction_simulation(model, X, y_log, trained_feats, trained_stats):
    # Valores a testear
    budgets     = [  250000, 300000, 400000, 500000]
    lot_areas   = [5000, 10000, 15000, 20000]
    longitudes  = [-93.599996, -93.603702, -93.619754]
    latitudes   = [41.9887374, 42.027555, 42.05027]

    results = []

    print("\nEJECUTANDO MULTI–SIMULACIÓN...\n")

    for B in budgets:
        for LA in lot_areas:
            for lon in longitudes:
                for lat in latitudes:

                    print(f"\n>>> Probando: Budget={B}, Lot={LA}, Lon={lon}, Lat={lat}")

                    try:
                        res = optimize_house(
                            model=model,
                            X=X,
                            y_log=y_log,
                            trained_feats=trained_feats,
                            trained_stats=trained_stats,
                            budget=B,
                            zero=True,
                            LON=lon,
                            LAT=lat,
                            Lot_Area=LA
                        )
                        
                        if res is not None:
                            results.append({
                                "budget": B,
                                "lot_area": LA,
                                "longitude": lon,
                                "latitude": lat,
                                "price_before": res["price_before"],
                                "price_after": res["price_after"],
                                "spent": res["spent"],
                                "profit": res["profit"],
                                "roi": res["roi"]
                            })

                    except Exception as e:
                        print(f"ERROR en simulación: {e}")
                        continue

    df = pd.DataFrame(results)
    print("\nSIMULACIONES COMPLETADAS.")
    print(df)

    # Guardar CSV para analizar
    df.to_csv("results/construction_results.csv", index=False)
    print("\nResultados guardados en construction_results.csv")

    return df


if __name__ == "__main__":

    print("\nCargando modelos y dataset...")

    # IMPORTAR 

    from joblib import load
    DATA_PATH = "data/ames_dum.csv"
    MODEL_PATH = "models/xgb_optuna_model.pkl"
    OUTPUT_CSV = "results/batch_optimization_results.csv"
    X, y = load_and_prepare(DATA_PATH)
    model = load(MODEL_PATH)
    trained_feats = X.columns.tolist()
    trained_stats = pd.DataFrame({
        "q05": X.quantile(0.05),
        "median": X.median(),
        "q95": X.quantile(0.95),
        "max": X.max()
    })

    construction_simulation(model = model, X = X, y_log = y, trained_feats=trained_feats,
                trained_stats=trained_stats,)
