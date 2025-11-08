# api.py
import os
import json
import math
import time
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from joblib import load
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse




from src.preprocessing import load_and_prepare
from src.optimization import optimize_house


# -------- Config & rutas base --------
DATA_PATH = "data/ames_dum.csv"
SAVE_DIR  = "models"

app = FastAPI(title="Casa Óptima API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Artefactos en memoria (se cargan al iniciar) --------
X, y = load_and_prepare(DATA_PATH)
y_log: Optional[pd.Series] = None
TRAINED_STATS: Optional[pd.DataFrame] = None
MODEL = None

def _ensure_models():
    global X, y_log, TRAINED_STATS, MODEL, LINEAR_MODEL

    print("[API] Cargando y preparando dataset…")
    X, y_log = load_and_prepare(DATA_PATH)

    # 1️⃣ XGBoost
    xgb_path = os.path.join(SAVE_DIR, "xgb_optuna_model.pkl")
    linear_path = os.path.join(SAVE_DIR, "linear_model.pkl")
    os.makedirs(SAVE_DIR, exist_ok=True)

    if os.path.exists(xgb_path):
        print("[API] Modelo XGBoost encontrado.")
        MODEL = load(xgb_path)
    else:
        raise HTTPException(500, "Modelo XGBoost no encontrado.")

    if os.path.exists(linear_path):
        print("[API] Modelo Lineal encontrado.")
        LINEAR_MODEL = load(linear_path)
    else:
        print("[WARN] Modelo lineal no encontrado. Solo se usará XGBoost.")

    TRAINED_STATS = pd.DataFrame({
        "q05": X.quantile(0.05),
        "median": X.median(),
        "q95": X.quantile(0.95),
        "max": X.max()
    })
    print(f"[API] Listo. Filas={len(X)}")

# Carga artefactos al iniciar el proceso
_ensure_models()

# -------- Utilidades JSON --------
def to_py(obj: Any):
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, pd.Series):
        return {k: to_py(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records"))
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    return obj

# -------- Esquemas pydantic --------
class OptimizeRequest(BaseModel):
    model: str = "xgboost" # "xgboost" o "linear"
    baseline_idx: int = 0
    budget: float = 200000
    pwl_k: int = 25
    baseline_prueba: Optional[Dict[str, float]] = None

class OptimizeResponse(BaseModel):
    price_before: float
    price_after: float
    spent: float
    profit: float
    roi: Optional[float]
    changes: Dict[str, float]
    cost_breakdown: Dict[str, float]
    final_house: List[Dict[str, float]]

# -------- Rutas API --------
@app.post("/optimize")
def optimize(req: OptimizeRequest):
    """Optimiza una casa del dataset usando el modelo XGBoost."""
    if X is None or y_log is None:
        raise HTTPException(500, "Datos no cargados")

    t0 = time.time()
    try:
        # 🔹 Elegir modelo según el valor recibido desde el frontend
        if req.model.lower() == "xgboost":
            model_obj = MODEL
            print("[API] ✅ Usando modelo XGBoost (Optuna).")
        elif req.model.lower() == "linear":
            if 'LINEAR_MODEL' not in globals() or LINEAR_MODEL is None:
                raise HTTPException(500, "❌ Modelo lineal no cargado. Asegúrate de tener models/linear_model.pkl")
            model_obj = LINEAR_MODEL
            print("[API] ✅ Usando modelo Lineal.")
        else:
            raise HTTPException(400, f"❌ Modelo no reconocido: {req.model}. Usa 'xgboost' o 'linear'.")

        # Para lineal, usar un valor seguro por si acaso
        pwl_value = req.pwl_k if req.model.lower() == "xgboost" else 25

        # 🔹 Llamar a optimize_house con el modelo correcto
        result = optimize_house(
            model=model_obj,
            X=X,
            y_log=y_log,
            trained_feats=X.columns.tolist(),
            trained_stats=TRAINED_STATS,
            baseline_idx=req.baseline_idx,
            baseline_prueba=req.baseline_prueba,
            budget=req.budget,
            pwl_k=pwl_value
        )

        # ---------------------------
        # 🔹 CASO 1: Sin solución (None)
        # ---------------------------
        if result is None:
            return JSONResponse(
                content={
                    "status": "infeasible",
                    "message": "No se encontró una solución factible.",
                    "violated_constraints": [],
                    "active_constraints": []
                },
                status_code=200
            )

        # ---------------------------
        # 🔹 CASO 2: Resultado especial con restricciones violadas
        # ---------------------------
        if isinstance(result, dict) and result.get("status") == "infeasible":
            return JSONResponse(content=result, status_code=200)

        # ---------------------------
        # 🔹 CASO 3: Solución factible (normal)
        # ---------------------------
        out = {
            "status": "ok",
            "price_before": float(result["price_before"]),
            "price_after": float(result["price_after"]),
            "spent": float(result["spent"]),
            "profit": float(result["profit"]),
            "roi": (
                None
                if result["roi"] is None or math.isnan(result["roi"])
                else float(result["roi"])
            ),
            "changes": to_py(result["changes"]),
            "cost_breakdown": to_py(result["cost_breakdown"]),
            "final_house": to_py(result["final_house"]),
        }

        return JSONResponse(content=out, status_code=200)

    except Exception as e:
        # 🔹 Si ocurre otro error no esperado, lo informamos como 500
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

    finally:
        dt = time.time() - t0
        print(f"[INFO] /optimize tardó {dt:.2f}s")




if __name__ == "__main__":
    # Allow running this file directly with `py api.py` or `python api.py`.
    # Ensure project root is on sys.path so `import src...` works when run as script.
    import sys
    import pathlib
    root = pathlib.Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Start Uvicorn server (use port 8001 to avoid conflicts with other servers)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
