import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm

# Input and output directories
TABLE = Path("spatial_shap_results/tables")
FIG = Path("spatial_shap_results/figures")
FIG.mkdir(parents=True, exist_ok=True)

features = ["grade", "sqft_living"]

columns = [
    ("standard_shap_usd", "Standard SHAP"),
    ("spatial_shap_usd", "KNN Spatial SHAP"),
    ("spatial_minus_standard_usd", "Spatial − Standard"),
]

fig, axes = plt.subplots(
    2, 3,
    figsize=(16, 9),
    sharex=True,
    sharey=True
)

for row, feature in enumerate(features):

    df = pd.read_csv(TABLE / f"{feature}_shap.csv")

    # Same color scale for Standard and Spatial SHAP
    shared_max = max(
        np.abs(df["standard_shap_usd"]).max(),
        np.abs(df["spatial_shap_usd"]).max()
    ) / 1000

    for col, (variable, title) in enumerate(columns):

        ax = axes[row, col]
        values = df[variable].to_numpy() / 1000

        # Difference map has its own symmetric scale
        if col < 2:
            vmax = shared_max
        else:
            vmax = max(np.abs(values).max(), 1e-9)

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
            s=14,
            alpha=0.9,
            linewidths=0
        )

        ax.set_title(
            f"({chr(97 + row * 3 + col)}) "
            f"{feature}: {title}",
            fontsize=12
        )

        ax.set_xlabel("Longitude")

        if col == 0:
            ax.set_ylabel("Latitude")

        ax.grid(alpha=0.15)

        # Correct longitude/latitude visual proportions
        ax.set_aspect(
            1 / np.cos(np.deg2rad(df["lat"].mean()))
        )

        colorbar = fig.colorbar(
            scatter,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

        colorbar.set_label(
            "SHAP value ($1,000s)",
            fontsize=9
        )

plt.tight_layout()

output = FIG / "Figure_SHAP_Comparison_6Panels.png"

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Publication figure saved:", output)