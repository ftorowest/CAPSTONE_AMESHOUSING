# CAPSTONE: Optimización de Renovaciones en Ames Housing

Este proyecto utiliza técnicas de ciencia de datos y optimización para recomendar mejoras en viviendas del dataset Ames Housing, maximizando el valor de reventa bajo restricciones de presupuesto.

## Descripción General
- **Preprocesamiento:** Limpieza de datos, ajuste por inflación usando el índice CPI de FRED, y transformación de variables categóricas ordinales.
- **Modelado:** Entrenamiento de modelos de regresión (Lineal y Gradient Boosting) para predecir el precio ajustado de las viviendas.
- **Evaluación:** Validación cruzada y métricas de desempeño (RMSE, MAE, R²). Interpretabilidad de coeficientes e importancias.
- **Optimización:** Formulación de un modelo de optimización con Gurobi para seleccionar mejoras óptimas bajo un presupuesto, maximizando precio, ganancia o ROI.

## Estructura de Archivos
- `capstone.ipynb`: Notebook principal con todo el flujo de trabajo.
- `AmesHousing(in).csv`: Base de datos de viviendas.
- `README.md`: Este archivo.

## Requisitos
- Python 3.8+
- Paquetes: numpy, pandas, scikit-learn, matplotlib, joblib, gurobipy, gurobi-machinelearning, pandas-datareader

Instalación recomendada (en notebook):
```python
%pip install --upgrade pip
%pip install numpy pandas scikit-learn matplotlib joblib gurobipy gurobi-machinelearning pandas-datareader
```

## Ejecución
1. Ejecuta todas las celdas del notebook `capstone.ipynb` en orden.
2. Ajusta parámetros como el presupuesto (`BUDGET`), el modelo (`MODEL_NAME`), y el objetivo (`OBJECTIVE_MODE`) según tu análisis.
3. Revisa los reportes de métricas y la solución óptima propuesta por Gurobi.

## Créditos
- Datos: [Ames Housing Dataset](https://www.openml.org/d/42165)
- Inspirado en desafíos de optimización y ciencia de datos de la Universidad Católica de Chile.

---
*Desarrollado para el curso de Investigación Operativa, 2025.*
