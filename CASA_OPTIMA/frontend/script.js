document.getElementById("optimizeForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const budget = parseFloat(document.getElementById("budget").value);
  const baseline_idx = parseInt(document.getElementById("baseline_idx").value);
  const pwl_k = parseInt(document.getElementById("pwl_k").value);

  const fields = [
    "First_Flr_SF","Second_Flr_SF","Year_Built","Exter_Qual","Total_Bsmt_SF",
    "Lot_Area","Garage_Area","Kitchen_Qual","Fireplaces","Year_Remod_Add",
    "Sale_Condition_Normal","Longitude","Full_Bath","Bsmt_Qual","Latitude",
    "Bsmt_Exposure","TotRms_AbvGrd","Half_Bath","Heating_QC","Garage_Finish",
    "Garage_Cond","Wood_Deck_SF","Open_Porch_SF","Bsmt_Full_Bath",
    "House_Style_One_Story","Sale_Type_New","Bedroom_AbvGr","Garage_Qual",
    "Kitchen_AbvGr","Pool_Area","Overall_Cond"
  ];

  const house_features = {};
  fields.forEach((f) => {
    const input = document.getElementById(f);
    house_features[f] = input ? parseFloat(input.value) : 0;
  });

  const payload = {
    baseline_idx,
    budget,
    pwl_k,
    baseline_prueba: house_features,
  };

  const output = document.getElementById("output");
  const summary = document.getElementById("summary");
  const tables = document.getElementById("tables");
  const changesTable = document.querySelector("#changesTable tbody");
  const costTable = document.querySelector("#costTable tbody");
  const finalHouseTable = document.querySelector("#finalHouseTable tbody");

  output.textContent = "⏳ Ejecutando optimización...";
  output.classList.remove("hidden");
  summary.classList.add("hidden");
  tables.classList.add("hidden");

  try {
    const res = await fetch("http://127.0.0.1:8001/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      output.textContent = `❌ Error ${res.status}:\n${data.detail || JSON.stringify(data)}`;
      return;
    }

    // Mostrar datos resumidos
    document.getElementById("price_before").textContent = `$${data.price_before.toLocaleString()}`;
    document.getElementById("price_after").textContent = `$${data.price_after.toLocaleString()}`;
    document.getElementById("spent").textContent = `$${data.spent.toLocaleString()}`;
    document.getElementById("profit").textContent = `$${data.profit.toLocaleString()}`;
    document.getElementById("roi").textContent = `${data.roi.toFixed(2)}x`;

    summary.classList.remove("hidden");
    tables.classList.remove("hidden");
    output.classList.add("hidden");

    // Tabla de cambios
    changesTable.innerHTML = "";
    for (const [k, v] of Object.entries(data.changes)) {
      if (v !== 0)
        changesTable.innerHTML += `<tr><td>${k}</td><td>${v}</td></tr>`;
    }

    // Tabla de costos
    costTable.innerHTML = "";
    for (const [k, v] of Object.entries(data.cost_breakdown)) {
      if (v !== 0)
        costTable.innerHTML += `<tr><td>${k}</td><td>$${v.toLocaleString()}</td></tr>`;
    }

    // 🏡 Tabla de la casa optimizada
    finalHouseTable.innerHTML = "";
    if (Array.isArray(data.final_house) && data.final_house.length > 0) {
      const finalHouse = data.final_house[0];
      for (const [k, v] of Object.entries(finalHouse)) {
        finalHouseTable.innerHTML += `<tr><td>${k}</td><td>${v}</td></tr>`;
      }
    }

  } catch (err) {
    output.textContent = "⚠️ Error: " + err;
  }
});
