// ============================================
//  REMODELACIÓN - LÓGICA
// ============================================

// Mostrar valores de sliders dinámicamente
const sliderIDs = [
  "Exter_Qual", "Kitchen_Qual", "Bsmt_Qual", "Bsmt_Exposure",
  "Overall_Cond", "Heating_QC", "Garage_Cond", "Garage_Finish", "Garage_Qual"
];

sliderIDs.forEach((id) => {
  const input = document.getElementById(id);
  const valueLabel = document.getElementById(`v_${id}`);
  if (input && valueLabel) {
    const updateLabel = () => (valueLabel.textContent = input.value);
    input.addEventListener("input", updateLabel);
    input.addEventListener("change", updateLabel);
    updateLabel();
  }
});

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

// === Gráfico de coordenadas ===
function renderLocationChart(finalHouse, baselineHouse) {
  const ctx = document.getElementById("locationChart")?.getContext("2d");
  if (!ctx) return;

  const latBase = baselineHouse.Latitude;
  const lonBase = baselineHouse.Longitude;
  const latOpt = finalHouse.Latitude;
  const lonOpt = finalHouse.Longitude;

  if (window.locationChart) window.locationChart.destroy();

  window.locationChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Casa original",
          data: [{ x: lonBase, y: latBase }],
          backgroundColor: "rgba(0, 102, 255, 0.7)",
          pointRadius: 7,
        },
        {
          label: "Casa optimizada",
          data: [{ x: lonOpt, y: latOpt }],
          backgroundColor: "rgba(0, 204, 102, 0.8)",
          pointRadius: 7,
        },
      ],
    },
    options: {
      plugins: {
        legend: { position: "bottom" },
        tooltip: {
          callbacks: {
            label: (ctx) => `(${ctx.raw.y.toFixed(4)}, ${ctx.raw.x.toFixed(4)})`,
          },
        },
      },
      scales: {
        x: { title: { display: true, text: "Longitud" } },
        y: { title: { display: true, text: "Latitud" } },
      },
    },
  });
}

// ============================================
//  EVENTO PRINCIPAL: OPTIMIZAR REMODELACIÓN
// ============================================

document.getElementById("optimizeForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const budget = parseFloat(document.getElementById("budget").value);
  const model = document.getElementById("modelSelector").value;
  const pwl_k = parseInt(document.getElementById("pwl_k").value);

  // Recoger todas las 31 características
  const fields = [
    "First_Flr_SF", "Second_Flr_SF", "Year_Built", "Exter_Qual",
    "Total_Bsmt_SF", "Lot_Area", "Garage_Area", "Kitchen_Qual",
    "Fireplaces", "Year_Remod_Add", "Sale_Condition_Normal",
    "Longitude", "Full_Bath", "Bsmt_Qual", "Latitude", "Bsmt_Exposure",
    "TotRms_AbvGrd", "Half_Bath", "Heating_QC", "Garage_Finish",
    "Garage_Cond", "Wood_Deck_SF", "Open_Porch_SF", "Bsmt_Full_Bath",
    "House_Style_One_Story", "Sale_Type_New", "Bedroom_AbvGr",
    "Garage_Qual", "Kitchen_AbvGr", "Pool_Area", "Overall_Cond",
    "Central_Air_Y"
  ];

  const house_features = {};
  fields.forEach((f) => {
    const input = document.getElementById(f);
    if (!input) return;
    house_features[f] = input.type === "checkbox" 
      ? (input.checked ? 1 : 0) 
      : parseFloat(input.value);
  });

  const payload = {
    model,
    baseline_idx: 0,
    budget,
    pwl_k,
    baseline_prueba: house_features,
    zero: false
  };

  const output = document.getElementById("output");
  const summary = document.getElementById("summary");
  const tables = document.getElementById("tables");
  const changesTable = document.querySelector("#changesTable tbody");
  const costTable = document.querySelector("#costTable tbody");
  const finalHouseTable = document.querySelector("#finalHouseTable tbody");

  output.textContent = "⏳ Optimizando remodelación...";
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
      output.textContent = "⚠️ No se encontró solución factible.";
      console.warn("Restricciones violadas:", data.violated_constraints);
      return;
    }

    if (data.status === "error") {
      output.textContent = `❌ Error: ${data.message}`;
      return;
    }

    // Actualizar resumen
    document.getElementById("price_before").textContent = `$${data.price_before.toLocaleString()}`;
    document.getElementById("price_after").textContent = `$${data.price_after.toLocaleString()}`;
    document.getElementById("spent").textContent = `$${data.spent.toLocaleString()}`;
    document.getElementById("profit").textContent = `$${data.profit.toLocaleString()}`;
    document.getElementById("roi").textContent = data.roi ? `${data.roi.toFixed(2)}x` : "N/A";

    summary.classList.remove("hidden");
    tables.classList.remove("hidden");
    output.classList.add("hidden");

    // Cambios sugeridos
    changesTable.innerHTML = "";
    for (const [k, v] of Object.entries(data.changes)) {
      if (v !== 0) {
        const displayValue = typeof v === 'number' ? v.toFixed(2) : v;
        changesTable.innerHTML += `<tr><td>${k}</td><td>${displayValue}</td></tr>`;
      }
    }

    // Costos
    costTable.innerHTML = "";
    for (const [k, v] of Object.entries(data.cost_breakdown)) {
      if (v !== 0) {
        costTable.innerHTML += `<tr><td>${k}</td><td>$${v.toLocaleString()}</td></tr>`;
      }
    }

    // Atributos finales
    finalHouseTable.innerHTML = "";
    if (Array.isArray(data.final_house) && data.final_house.length > 0) {
      for (const [k, v] of Object.entries(data.final_house[0])) {
        const displayValue = typeof v === 'number' ? v.toFixed(2) : v;
        finalHouseTable.innerHTML += `<tr><td>${k}</td><td>${displayValue}</td></tr>`;
      }
    }

    // Dibujar gráfico de ubicación
    renderLocationChart(data.final_house[0], house_features);

    // Agregar marcador en el mapa
    const iframe = document.getElementById("mapIframe");
    const latOpt = parseFloat(data.final_house[0].Latitude);
    const lonOpt = parseFloat(data.final_house[0].Longitude);

    if (!isNaN(latOpt) && !isNaN(lonOpt)) {
      setTimeout(() => {
        if (iframe && iframe.contentWindow && iframe.contentWindow.addUserMarker) {
          iframe.contentWindow.addUserMarker(latOpt, lonOpt, "🏠 Casa optimizada");
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
