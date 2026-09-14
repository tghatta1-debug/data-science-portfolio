fetch("data/summary.json").then(response => response.json()).then(data => {
  document.querySelector("#institution-count").textContent = data.institutions_analyzed;
  document.querySelector("#average-price").textContent = new Intl.NumberFormat("en-US", {style: "currency", currency: "USD", maximumFractionDigits: 0}).format(data.average_net_price);
  document.querySelector("#average-rate").textContent = `${data.average_graduation_rate}%`;
}).catch(() => {});
