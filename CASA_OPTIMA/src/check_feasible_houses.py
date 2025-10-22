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


def load_data(filepath='../data/ames_dum.csv'):
    """Carga el dataset de Ames Housing"""
    print("=" * 80)
    print("ANÁLISIS DE CASAS FACTIBLES DEL DATASET")
    print("=" * 80)
    
    df = pd.read_csv(filepath)
    print(f"\n✓ Dataset cargado: {len(df)} casas")
    return df


def check_all_constraints(df):
    """
    Verifica TODAS las restricciones del modelo para cada casa
    
    Returns:
        DataFrame con columnas adicionales indicando cumplimiento
    """
    print("\n" + "=" * 80)
    print("VERIFICANDO RESTRICCIONES DEL MODELO")
    print("=" * 80)
    
    df = df.copy()
    violations = pd.DataFrame(index=df.index)
    
    # RESTRICCIÓN 2: Primer piso + garage + porches <= 80% del lote
    print("\n[1/13] Área construida <= 80% lote...")
    area_construida = (df['First_Flr_SF'] + 
                       df['Garage_Cars'] * ESPACIO_POR_AUTO + 
                       df.get('Open_Porch_SF', 0) + 
                       df.get('Wood_Deck_SF', 0) + 
                       df.get('Pool_Area', 0))
    violations['lot_area'] = area_construida > df['Lot_Area'] * 0.8
    
    # RESTRICCIÓN 3: Segundo piso <= Primer piso
    print("[2/13] Segundo piso <= Primer piso...")
    violations['second_floor'] = df['Second_Flr_SF'] > 1.2* df['First_Flr_SF']
    
    # RESTRICCIÓN 4: Si es de un piso, segundo piso = 0
    print("[3/13] Casa un piso → segundo piso = 0...")
    if 'House_Style_One_Story' in df.columns:
        violations['one_story'] = (df['House_Style_One_Story'] == 1) & (df['Second_Flr_SF'] > 0)
    else:
        violations['one_story'] = False
    
    # RESTRICCIÓN 5: Garage <= Primer piso
    print("[4/14] Área garage <= Primer piso...")
    violations['garage_size'] = df.get('Garage_Area', 0) > df['First_Flr_SF']
    
    # RESTRICCIÓN 6: Basement <= Primer piso
    print("[5/13] Basement <= Primer piso...")
    violations['basement_size'] = df['Total_Bsmt_SF'] > df['First_Flr_SF']*1.2
    
    # RESTRICCIÓN 7: Baños totales <= Habitaciones + 1
    print("[6/13] Baños totales <= Habitaciones + 1...")
    violations['bath_rooms'] = (df['Full_Bath'] + df['Half_Bath']) > (df['TotRms_AbvGrd'] + 1)
    
    # RESTRICCIÓN 8: Baños completos <= Dormitorios
    print("[7/13] Baños completos <= Dormitorios...")
    violations['fullbath_bedroom'] = df['Full_Bath'] > df['Bedroom_AbvGr']
    
    # RESTRICCIÓN 9: Medios baños <= Baños completos
    print("[8/13] Medios baños <= Baños completos...")
    violations['halfbath_fullbath'] = df['Half_Bath'] > df['Full_Bath']
    
    # RESTRICCIÓN 10: Chimeneas <= Baños totales
    print("[9/13] Chimeneas <= Baños totales...")
    violations['fireplaces'] = df['Fireplaces'] > (df['Full_Bath'] + df['Half_Bath'])
    
    # RESTRICCIÓN 13: Cocina >= 1
    print("[10/13] Al menos 1 cocina...")
    violations['kitchen_min'] = df['Kitchen_AbvGr'] < 1
    
    # RESTRICCIÓN 14: Habitaciones >= 1
    print("[11/13] Al menos 1 habitación...")
    violations['rooms_min'] = df['TotRms_AbvGrd'] < 1
    
    # RESTRICCIÓN 15: Baños completos >= 1
    print("[12/13] Al menos 1 baño completo...")
    violations['bath_min'] = df['Full_Bath'] < 1
    
    # RESTRICCIÓN 16: SF suficientes para atributos
    print("[13/13] Área suficiente para atributos...")
    area_disponible = (df['First_Flr_SF'] + 
                       df['Second_Flr_SF'])
    area_necesaria = (df['Full_Bath'] * BAÑO_PROMEDIO + 
                      df.get('Bsmt_Full_Bath', 0) * BAÑO_PROMEDIO +
                      df['Kitchen_AbvGr'] * COCINA_PROMEDIO + 
                      df['TotRms_AbvGrd'] * HABITACION_PROMEDIO + 100)
    violations['sf_min'] = area_disponible < area_necesaria
    
    # RESTRICCIÓN 22: Dormitorios >= 1
    print("[14/13] Al menos 1 dormitorio...")
    violations['bedroom_min'] = df['Bedroom_AbvGr'] < 1
    
    # Calcular total de violaciones
    df['n_violations'] = violations.sum(axis=1)
    df['is_feasible'] = df['n_violations'] == 0
    
    # Agregar columnas de violaciones
    for col in violations.columns:
        df[f'violates_{col}'] = violations[col]
    
    return df, violations


def print_summary(df):
    """Imprime resumen de factibilidad"""
    print("\n" + "=" * 80)
    print("RESUMEN DE FACTIBILIDAD")
    print("=" * 80)
    
    n_feasible = df['is_feasible'].sum()
    pct_feasible = (n_feasible / len(df)) * 100
    
    print(f"\n✓ Casas FACTIBLES:   {n_feasible:>6,} ({pct_feasible:.2f}%)")
    print(f"✗ Casas INFACTIBLES: {len(df) - n_feasible:>6,} ({100 - pct_feasible:.2f}%)")
    
    # Distribución de violaciones
    print("\n" + "-" * 80)
    print("DISTRIBUCIÓN DE VIOLACIONES")
    print("-" * 80)
    
    violation_dist = df['n_violations'].value_counts().sort_index()
    for n_viols, count in violation_dist.items():
        pct = (count / len(df)) * 100
        print(f"  {n_viols:2d} violación(es): {count:>5,} casas ({pct:>5.2f}%)")


def analyze_violations(df):
    """Analiza qué restricciones se violan más"""
    print("\n" + "=" * 80)
    print("RESTRICCIONES MÁS VIOLADAS")
    print("=" * 80)
    
    violation_cols = [col for col in df.columns if col.startswith('violates_')]
    
    violation_counts = {}
    for col in violation_cols:
        restriction_name = col.replace('violates_', '')
        count = df[col].sum()
        if count > 0:
            violation_counts[restriction_name] = count
    
    # Ordenar por cantidad de violaciones
    sorted_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Restricción':<30} {'Violaciones':>12} {'% Casas':>10}")
    print("-" * 80)
    
    for restriction, count in sorted_violations:
        pct = (count / len(df)) * 100
        print(f"{restriction:<30} {count:>12,} {pct:>9.1f}%")


def show_feasible_examples(df, n=10):
    """Muestra ejemplos de casas factibles"""
    print("\n" + "=" * 80)
    print(f"EJEMPLOS DE CASAS FACTIBLES (primeras {n})")
    print("=" * 80)
    
    feasible = df[df['is_feasible']].head(n)
    
    if len(feasible) == 0:
        print("\n❌ No hay casas factibles en el dataset")
        return
    
    important_cols = ['First_Flr_SF', 'Second_Flr_SF', 'Total_Bsmt_SF', 
                      'Garage_Cars', 'Full_Bath', 'Half_Bath', 'Bedroom_AbvGr',
                      'TotRms_AbvGrd', 'Kitchen_AbvGr', 'Fireplaces', 'Lot_Area']
    
    print(f"\nCasas factibles encontradas: {len(df[df['is_feasible']]):,}")
    print("\nCaracterísticas de las primeras casas factibles:")
    print("-" * 80)
    
    for idx, row in feasible[important_cols].iterrows():
        print(f"\nCasa #{idx}:")
        for col in important_cols:
            print(f"  {col:20s}: {row[col]:>8,.0f}")


def show_infeasible_examples(df, n=5):
    """Muestra ejemplos de casas infactibles con sus violaciones"""
    print("\n" + "=" * 80)
    print(f"EJEMPLOS DE CASAS INFACTIBLES (primeras {n})")
    print("=" * 80)
    
    infeasible = df[~df['is_feasible']].head(n)
    
    if len(infeasible) == 0:
        print("\n✓ Todas las casas son factibles")
        return
    
    violation_cols = [col for col in df.columns if col.startswith('violates_')]
    
    for idx, row in infeasible.iterrows():
        violations_list = [col.replace('violates_', '') 
                          for col in violation_cols if row[col]]
        
        print(f"\nCasa #{idx} - {len(violations_list)} violación(es):")
        for v in violations_list:
            print(f"  ✗ {v}")


def save_results(df, output_file='../models/feasible_houses.csv'):
    """Guarda los resultados en un CSV"""
    print("\n" + "=" * 80)
    print("GUARDANDO RESULTADOS")
    print("=" * 80)
    
    # Seleccionar columnas relevantes
    output_cols = ['is_feasible', 'n_violations'] + \
                  [col for col in df.columns if col.startswith('violates_')]
    
    # Guardar todas las columnas
    df.to_csv(output_file, index=False)
    print(f"\n✓ Resultados guardados en: {output_file}")
    
    # Guardar solo casas factibles
    feasible_file = output_file.replace('.csv', '_only_feasible.csv')
    df[df['is_feasible']].to_csv(feasible_file, index=False)
    print(f"✓ Casas factibles guardadas en: {feasible_file}")
    
    # Guardar resumen
    summary_file = output_file.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("RESUMEN DE FACTIBILIDAD\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total casas: {len(df):,}\n")
        f.write(f"Casas factibles: {df['is_feasible'].sum():,} ({df['is_feasible'].mean()*100:.2f}%)\n")
        f.write(f"Casas infactibles: {(~df['is_feasible']).sum():,} ({(~df['is_feasible']).mean()*100:.2f}%)\n")
    
    print(f"✓ Resumen guardado en: {summary_file}")


def main():
    """Función principal"""
    # 1. Cargar datos
    df = load_data()
    
    # 2. Verificar restricciones
    df, violations = check_all_constraints(df)
    
    # 3. Imprimir resumen
    print_summary(df)
    
    # 4. Analizar violaciones
    analyze_violations(df)
    
    # 5. Mostrar ejemplos factibles
    show_feasible_examples(df, n=10)
    
    # 6. Mostrar ejemplos infactibles
    show_infeasible_examples(df, n=5)
    
    # 7. Guardar resultados
    save_results(df)
    
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
