import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Directories
TABLE = Path("spatial_shap_results/tables")
FIG = Path("spatial_shap_results/figures")
FIG.mkdir(parents=True, exist_ok=True)

# Experimental settings
GRID_SIZE = 40

# Load regional importance results
knn = pd.read_csv(
    TABLE / "regional_g40_eval30_k20_mask10km.csv"
)

gaussian = pd.read_csv(
    TABLE /
    "gaussian_regional_g40_eval30_k20_mask10km_bw1_bg100.csv"
)

# Match grid cells
data = knn.merge(
    gaussian,
    on="grid_index",
    suffixes=("_knn", "_gaussian")
)

data = data.sort_values("grid_index")

assert len(data) == GRID_SIZE ** 2

# Coordinates
lon = data["long_knn"].to_numpy().reshape(
    GRID_SIZE, GRID_SIZE
)

lat = data["lat_knn"].to_numpy().reshape(
    GRID_SIZE, GRID_SIZE
)

features = ["grade", "sqft_living"]

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 10),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

for row, feature in enumerate(features):

    # KNN regional importance
    knn_values = data[
        f"{feature}_importance_usd_knn"
    ].to_numpy().reshape(GRID_SIZE, GRID_SIZE) / 1000

    # Gaussian regional importance
    gaussian_values = data[
        f"{feature}_importance_usd_gaussian"
    ].to_numpy().reshape(GRID_SIZE, GRID_SIZE) / 1000

    # Same color scale for both methods
    vmax = max(
        np.nanmax(knn_values),
        np.nanmax(gaussian_values)
    )

    maps = [
        (knn_values, "KNN Spatial SHAP"),
        (gaussian_values, "Gaussian Spatial SHAP")
    ]

    for col, (values, title) in enumerate(maps):

        ax = axes[row, col]

        im = ax.pcolormesh(
            lon,
            lat,
            np.ma.masked_invalid(values),
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            shading="auto"
        )

        letter = chr(97 + row * 2 + col)

        ax.set_title(
            f"({letter}) {feature}: {title}",
            fontsize=12
        )

        ax.set_xlabel("Longitude")

        if col == 0:
            ax.set_ylabel("Latitude")

        # Correct geographic proportions
        ax.set_aspect(
            1 / np.cos(np.deg2rad(np.nanmean(lat)))
        )

    # One shared colorbar for each feature
    cbar = fig.colorbar(
        im,
        ax=axes[row, :],
        shrink=0.85,
        pad=0.02
    )

    cbar.set_label(
        "Mean absolute Spatial SHAP ($1,000s)",
        fontsize=10
    )

output = FIG / "Figure_Regional_KNN_vs_Gaussian_4Panels.png"

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Regional comparison figure saved:", output)