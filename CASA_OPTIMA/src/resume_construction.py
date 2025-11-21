import pandas as pd
import os

INPUT = "results/construcciones_optimas.csv"

OUTPUT_MAIN = "results/construcciones_optimas_resumen.txt"

df = pd.read_csv(INPUT)

# Ordenar
df = df.sort_values(["latitude", "longitude", "budget"])

# Crear carpeta si no existe
os.makedirs("results", exist_ok=True)

# ============================================================================================
# 1. ARCHIVO PRINCIPAL: resumen por ubicación y presupuesto (SOLO CAMBIOS)
# ============================================================================================

with open(OUTPUT_MAIN, "w", encoding="utf-8") as f:

    f.write("RESUMEN DE CONSTRUCCIONES OPTIMAS (SOLO CAMBIOS)\n")
    f.write("="*80 + "\n\n")

    # Agrupar por ubicación
    for (lat, lon), sub in df.groupby(["latitude", "longitude"]):

        f.write(f"UBICACIÓN: ({lat:.6f}, {lon:.6f})\n")
        f.write("-"*80 + "\n\n")

        # Por cada presupuesto
        for budget, row in sub.groupby("budget"):

            r = row.iloc[0]

            f.write(f"--- PRESUPUESTO: ${budget:,.0f} ---\n")

            # --- INDICADORES ECONÓMICOS ---
            f.write("\n[RESULTADOS ECONÓMICOS]\n")
            f.write("-"*40 + "\n")
            f.write(f"  Precio antes       : {r['price_before']:,.0f}\n")
            f.write(f"  Precio después     : {r['price_after']:,.0f}\n")
            f.write(f"  Gasto total        : {r['spent']:,.0f}\n")
            f.write(f"  Ganancia (Profit)  : {r['profit']:,.0f}\n")
            f.write(f"  ROI                : {r['roi']:.4f}\n")
            f.write(f"  Lot Area           : {r['lot_area']:,.0f}\n")

            # --- CAMBIOS REALES ---
            f.write("\n[CAMBIOS REALIZADOS EN LA CASA]\n")
            f.write("-"*40 + "\n")

            cols_ignore = {
                "latitude","longitude","budget","lot_area",
                "price_before","price_after","spent",
                "profit","roi","tag"
            }

            cambios = []

            for col in row.columns:
                if col in cols_ignore:
                    continue

                valor = r[col]

                if isinstance(valor, (int, float)) and abs(valor) > 1e-9:
                    cambios.append((col, valor))

            if len(cambios) == 0:
                f.write("  * No hubo cambios\n")
            else:
                for nombre, valor in cambios:
                    f.write(f"  {nombre:25s}: {valor}\n")

            f.write("\n" + "-"*80 + "\n\n")

print(f"\n✔ Archivo principal generado en: {OUTPUT_MAIN}")


# ============================================================================================
# 2. ARCHIVOS POR PRESUPUESTO: un TXT por cada presupuesto
# ============================================================================================

presupuestos = sorted(df["budget"].unique())

for B in presupuestos:

    OUTPUT_B = f"results/resumen_presupuesto_{int(B)}.txt"

    with open(OUTPUT_B, "w", encoding="utf-8") as f:

        f.write(f"RESUMEN PARA PRESUPUESTO ${B:,.0f}\n")
        f.write("="*80 + "\n\n")

        dfB = df[df["budget"] == B]

        if dfB.empty:
            f.write("⚠ No hay construcciones para este presupuesto.\n")
            continue

        # Ordenar por ubicación
        dfB = dfB.sort_values(["latitude", "longitude"])

        for _, r in dfB.iterrows():

            lat = r["latitude"]
            lon = r["longitude"]

            f.write(f"UBICACIÓN: ({lat:.6f}, {lon:.6f})\n")
            f.write("-"*80 + "\n")

            # ECONÓMICOS
            f.write(f"  Precio antes       : {r['price_before']:,.0f}\n")
            f.write(f"  Precio después     : {r['price_after']:,.0f}\n")
            f.write(f"  Gastado            : {r['spent']:,.0f}\n")
            f.write(f"  Profit             : {r['profit']:,.0f}\n")
            f.write(f"  ROI                : {r['roi']:.4f}\n")
            f.write(f"  Lot Area           : {r['lot_area']:,.0f}\n")

            f.write("\n  CAMBIOS REALIZADOS:\n")
            f.write("  " + "-"*70 + "\n")

            cols_ignore = {
                "latitude","longitude","budget","lot_area",
                "price_before","price_after","spent",
                "profit","roi","tag"
            }

            cambios = []

            for col in dfB.columns:
                if col in cols_ignore:
                    continue

                val = r[col]
                if isinstance(val, (int, float)) and abs(val) > 1e-9:
                    cambios.append((col, val))

            if not cambios:
                f.write("   * Sin cambios\n")
            else:
                for nombre, valor in cambios:
                    f.write(f"   {nombre:25s}: {valor}\n")

            f.write("\n")

    print(f"✔ Archivo generado: {OUTPUT_B}")

print("\n🎉 Todos los archivos listos.\n")

