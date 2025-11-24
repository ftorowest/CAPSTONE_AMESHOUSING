// ============================================
//  CONSTRUCCIÓN DESDE CERO - LÓGICA
// ============================================

// Mostrar/ocultar precisión según modelo
const modelSelector = document.getElementById("modelSelector");
const precisionLabel = document.querySelector(".precision-label");

modelSelector.addEventListener("change", () => {
  precisionLabel.style.display = modelSelector.value === "xgboost" ? "block" : "none";
});

// Actualizar label de precisión
const pwlRange = document.getElementById("pwl_k");
const pwlLabel = document.getElementById("pwl_label");
pwlRange.addEventListener("input", () => {
  pwlLabel.textContent = pwlRange.value;
});

// ============================================
//  EVENTO PRINCIPAL: OPTIMIZAR CONSTRUCCIÓN
// ============================================

document.getElementById("optimizeForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const budget = parseFloat(document.getElementById("budget").value);
  const model = document.getElementById("modelSelector").value;
  const pwl_k = parseInt(document.getElementById("pwl_k").value);
  const LON = parseFloat(document.getElementById("LON").value);
  const LAT = parseFloat(document.getElementById("LAT").value);
  const Lot_Area = parseFloat(document.getElementById("Lot_Area").value);
  const land_price = parseFloat(document.getElementById("land_price").value);

  const payload = {
    model,
    baseline_idx: 0,
    budget,
    pwl_k,
    zero: true,
    LON,
    LAT,
    Lot_Area,
    land_price
  };

  const output = document.getElementById("output");
  const summary = document.getElementById("summary");
  const tables = document.getElementById("tables");
  const costTable = document.querySelector("#costTable tbody");
  const finalHouseTable = document.querySelector("#finalHouseTable tbody");

  output.textContent = "⏳ Optimizando construcción desde cero...";
  output.classList.remove("hidden");
  summary.classList.add("hidden");
  tables.classList.add("hidden");

  try {
    const res = await fetch("/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      output.textContent = `❌ Error ${res.status}:\n${data.detail || JSON.stringify(data)}`;
      return;
    }

    if (data.status === "infeasible") {
      output.textContent = "⚠️ No se encontró solución factible con los parámetros proporcionados.";
      console.warn("Restricciones violadas:", data.violated_constraints);
      return;
    }

    if (data.status === "error") {
      output.textContent = `❌ Error: ${data.message}`;
      return;
    }

    // Actualizar resumen
    document.getElementById("price_after").textContent = `$${data.price_after.toLocaleString()}`;
    document.getElementById("spent").textContent = `$${data.spent.toLocaleString()}`;
    document.getElementById("profit").textContent = `$${data.profit.toLocaleString()}`;
    document.getElementById("roi").textContent = data.roi ? `${data.roi.toFixed(2)}x` : "N/A";

    summary.classList.remove("hidden");
    tables.classList.remove("hidden");
    output.classList.add("hidden");

    // Costos
    costTable.innerHTML = "";
    for (const [k, v] of Object.entries(data.cost_breakdown)) {
      if (v !== 0) {
        costTable.innerHTML += `<tr><td>${k}</td><td>$${v.toLocaleString()}</td></tr>`;
      }
    }

    // Especificaciones finales
    finalHouseTable.innerHTML = "";
    if (Array.isArray(data.final_house) && data.final_house.length > 0) {
      for (const [k, v] of Object.entries(data.final_house[0])) {
        const displayValue = typeof v === 'number' ? v.toFixed(2) : v;
        finalHouseTable.innerHTML += `<tr><td>${k}</td><td>${displayValue}</td></tr>`;
      }
    }

    // Marcar en mapa
    const iframe = document.getElementById("mapIframe");
    const latOpt = parseFloat(data.final_house[0].Latitude);
    const lonOpt = parseFloat(data.final_house[0].Longitude);

    if (!isNaN(latOpt) && !isNaN(lonOpt)) {
      setTimeout(() => {
        if (iframe && iframe.contentWindow && iframe.contentWindow.addUserMarker) {
          iframe.contentWindow.addUserMarker(latOpt, lonOpt, "🏗️ Casa optimizada");
          console.log("✅ Marcador agregado en:", latOpt, lonOpt);
        } else {
          console.warn("⚠️ addUserMarker no disponible en el mapa");
        }
      }, 800);
    }

  } catch (err) {
    output.textContent = "⚠️ Error de conexión: " + err.message;
    console.error(err);
  }
});
