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
    """Carga/entrena y deja todo listo para servir."""
    global X, y_log, TRAINED_STATS, MODEL

    # 1) Datos
    print("[API] Cargando y preparando dataset…")
    X, y_log = load_and_prepare(DATA_PATH)

    # 2) Modelo XGBoost
    xgb_path = os.path.join(SAVE_DIR, "xgb_optuna_model.pkl")
    os.makedirs(SAVE_DIR, exist_ok=True)

    if os.path.exists(xgb_path):
        print("[API] Modelo encontrado. Cargando…")
        MODEL = load(xgb_path)
    else:
        raise HTTPException(500, "Modelo no encontrado. Entrena primero con train_model.py")

    # 3) Metadatos para optimización
    TRAINED_STATS = pd.DataFrame({
        "q05":   X.quantile(0.05),
        "median":X.median(),
        "q95":   X.quantile(0.95),
        "max":   X.max()
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

# -------- Endpoints --------
@app.post("/optimize")
def optimize(req: OptimizeRequest):
    """Optimiza una casa del dataset usando el modelo XGBoost."""
    if X is None or y_log is None:
        raise HTTPException(500, "Datos no cargados")

    t0 = time.time()
    try:
        result = optimize_house(
            model=MODEL,
            X=X,
            y_log=y_log,
            trained_feats=X.columns.tolist(),
            trained_stats=TRAINED_STATS,
            baseline_idx=req.baseline_idx,
            baseline_prueba=req.baseline_prueba,
            budget=req.budget,
            pwl_k=req.pwl_k,
        )
        if not result:
            raise HTTPException(422, "Optimización no factible")

        # Extraer desde el diccionario que retorna optimize_house
        price_before   = result.get("price_before")
        price_after    = result.get("price_after")
        spent          = result.get("spent")
        profit         = result.get("profit")
        roi            = result.get("roi")
        changes        = result.get("changes")
        cost_breakdown = result.get("cost_breakdown")
        final_house    = result.get("final_house")

        # Convertir ROI a número o None
        try:
            roi_value = float(roi) if roi is not None else None
            if roi_value is not None and math.isnan(roi_value):
                roi_value = None
        except (ValueError, TypeError):
            roi_value = None

        out = {
            "price_before": float(price_before),
            "price_after": float(price_after),
            "spent": float(spent),
            "profit": float(profit),
            "roi": roi_value,
            "changes": to_py(changes),
            "cost_breakdown": to_py(cost_breakdown),
            "final_house": to_py(final_house),
        }
        return out

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
