import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

TABLE = Path("spatial_shap_results/tables")
FIG = Path("spatial_shap_results/figures")

df = pd.read_csv(TABLE / "location_effect.csv")

values = df["location_effect_usd"].to_numpy() / 1000
vmax = np.max(np.abs(values))

fig, ax = plt.subplots(figsize=(8, 7))

scatter = ax.scatter(
    df["long"],
    df["lat"],
    c=values,
    cmap="coolwarm",
    norm=TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax
    ),
    s=20,
    alpha=0.9,
    linewidths=0
)

ax.set_xlabel("Longitude", fontsize=11)
ax.set_ylabel("Latitude", fontsize=11)

ax.set_title(
    "Overall Location Effect",
    fontsize=13
)

ax.set_aspect(
    1 / np.cos(np.deg2rad(df["lat"].mean()))
)

ax.grid(alpha=0.15)

cbar = fig.colorbar(
    scatter,
    ax=ax,
    shrink=0.85,
    pad=0.03
)

cbar.set_label(
    "Local minus global baseline ($1,000s)",
    fontsize=10
)

plt.tight_layout()

output = FIG / "Figure_Overall_Location_Effect_Publication.png"

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Location effect figure saved:", output)