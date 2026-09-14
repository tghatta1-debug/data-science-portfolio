import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

source = Path("data/college_scorecard_nc_public_four_year.json")
records = json.loads(source.read_text(encoding="utf-8"))["results"]
df = pd.DataFrame(records).rename(columns={
    "school.name": "institution",
    "latest.cost.avg_net_price.public": "net_price",
    "latest.completion.rate_suppressed.overall": "graduation_rate",
    "latest.student.size": "student_size",
})
df = df.dropna(subset=["net_price", "graduation_rate"]).copy()
df["graduation_rate"] = df["graduation_rate"] * 100
df = df.sort_values("graduation_rate", ascending=False)
Path("assets").mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(df["institution"], df["graduation_rate"], color="#16324f")
ax.invert_yaxis()
ax.set_xlabel("Six-year graduation rate (%)")
ax.set_title("Six-year graduation rates at North Carolina public four-year universities")
for bar, value in zip(bars, df["graduation_rate"]):
    ax.text(value + 0.7, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=8)
ax.set_xlim(0, min(100, df["graduation_rate"].max() + 10))
fig.tight_layout()
fig.savefig("assets/graduation-rates.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 6))
sizes = df["student_size"].fillna(df["student_size"].median()).clip(lower=100) / 16
ax.scatter(df["net_price"], df["graduation_rate"], s=sizes, color="#d46a3a", alpha=0.78, edgecolor="white", linewidth=0.8)
for _, row in df.iterrows():
    ax.annotate(row["institution"], (row["net_price"], row["graduation_rate"]), xytext=(5, 5), textcoords="offset points", fontsize=7)
ax.set_xlabel("Average annual net price for public institutions (USD)")
ax.set_ylabel("Six-year graduation rate (%)")
ax.set_title("Net price and graduation rate")
fig.tight_layout()
fig.savefig("assets/price-vs-graduation.png", dpi=180, bbox_inches="tight")
plt.close(fig)

summary = {
    "institutions_analyzed": int(len(df)),
    "average_net_price": round(float(df["net_price"].mean())),
    "average_graduation_rate": round(float(df["graduation_rate"].mean()), 1),
    "highest_graduation_rate": df.iloc[0]["institution"],
    "highest_graduation_rate_value": round(float(df.iloc[0]["graduation_rate"]), 1),
}
Path("data/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
