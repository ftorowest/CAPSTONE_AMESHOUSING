Casa Óptima — API (backend)
=============================

Descripción
-----------
API ligera para optimizar una vivienda (incluye la predicción interna) usando un modelo XGBoost ya entrenado.

Notas clave
----------
- Esta API NO entrena modelos. Debes colocar el modelo preentrenado en `models/xgb_optuna_model.pkl`.
- La optimización usa el módulo `src/optimization.py` y Gurobi. Gurobi debe estar instalado y licenciado localmente.
- El frontend enviará las características de la casa en JSON. La API ejecuta la predicción internamente y devuelve la optimización sugerida.

Instalación
-----------
1. Crear un entorno (opcional pero recomendado):

```powershell
python -m venv .venv;
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Asegúrate de tener una licencia válida de Gurobi y que `gurobipy` esté disponible en el entorno.

Archivos importantes
-------------------
- `backend.py` — código de la API (FastAPI).
- `models/xgb_optuna_model.pkl` — modelo XGBoost pre-entrenado requerido.
- `data/ames_dum.csv` — dataset usado para conocer orden de features y estadísticos (medianas, cuantiles).
- `src/optimization.py` — lógica de optimización (usa Gurobi + gurobi_ml).

Rutas (endpoints)
------------------
- GET /  
  Información básica y rutas disponibles.

- GET /models  
  Informa si existe `models/xgb_optuna_model.pkl`.

- POST /optimize  
  Ejecuta la predicción interna y optimiza la casa enviada bajo un presupuesto.
  Body (JSON):
  {
    "features": { "First_Flr_SF": 856, "Bedroom_AbvGr": 3, ... },
    "budget": 125000
  }

  Respuesta (JSON): resultado de `optimize_house` (precio antes/después, gasto, cambios sugeridos, ROI, final_house...)

Ejemplos (PowerShell / curl)
----------------------------
- Optimizar (PowerShell + curl):

```powershell
$body = '{"features": {"First_Flr_SF":856, "Bedroom_AbvGr":3}, "budget":125000}'
curl -Method POST -Uri http://127.0.0.1:8000/optimize -Body $body -ContentType 'application/json'
```

Ejecución
---------
Para ejecutar localmente:

```powershell
python backend.py
```

La API quedará disponible en `http://127.0.0.1:8000`. Puedes abrir `http://127.0.0.1:8000/docs` para ver la documentación automática (OpenAPI) y probar los endpoints.

Limitaciones y recomendaciones
------------------------------
- Gurobi y `gurobi_ml` son necesarios para `optimize`.
- El endpoint `/optimize` ejecuta un solver de optimización y puede tardar varios segundos o minutos.
- Para producción, considera:
  - Ejecutar la optimización como tarea en segundo plano y exponer un endpoint para consultar el estado/resultado.
  - Agregar autenticación y límites de uso.
  - Validar más estrictamente las features de entrada (rangos, tipos).

Siguientes pasos posibles
------------------------
- Añadir endpoint `/health` y métricas.
- Implementar background tasks para optimización (FastAPI BackgroundTasks o Celery).
- Añadir tests unitarios para `/optimize`.
- Mejorar el esquema de validación (Pydantic) para las features permitiendo valores por defecto y rangos.

Contacto
--------
Repositorio: CAPSTONE_AMESHOUSING

