
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

folder = Path(__file__).resolve().parent / "publication_result_1"

df = pd.read_csv(
    folder / "KingCounty_standard_vs_spatial_map_grade_600.csv"
)

lon = df["Longitude"].to_numpy()
lat = df["Latitude"].to_numpy()

# 150 x 150 continuous grid
grid_lon, grid_lat = np.meshgrid(
    np.linspace(lon.min(), lon.max(), 150),
    np.linspace(lat.min(), lat.max(), 150)
)

points = np.column_stack([lon, lat])

# Mask areas more than 8 km from the nearest house
scale = np.cos(np.deg2rad(lat.mean()))
tree = cKDTree(np.column_stack([lon * scale, lat]))
distance, _ = tree.query(
    np.column_stack([
        grid_lon.ravel() * scale,
        grid_lat.ravel()
    ])
)
mask = distance.reshape(grid_lon.shape) > 8 / 111

maps = [
    ("Standard SHAP", "Figure26_continuous_standard_SHAP_grade.png"),
    ("Spatial SHAP", "Figure27_continuous_spatial_SHAP_grade.png"),
    ("Spatial minus Standard", "Figure28_continuous_difference_SHAP_grade.png")
]

shared_limit = max(
    df["Standard SHAP"].abs().max(),
    df["Spatial SHAP"].abs().max()
) / 1000

for column, filename in maps:
    values = df[column].to_numpy() / 1000

    surface = griddata(
        points,
        values,
        (grid_lon, grid_lat),
        method="linear"
    )

    surface = np.ma.masked_invalid(surface)
    surface = np.ma.masked_where(mask, surface)

    limit = (
        shared_limit if column != "Spatial minus Standard"
        else df[column].abs().max() / 1000
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.pcolormesh(
        grid_lon, grid_lat, surface,
        cmap="RdBu_r",
        shading="auto",
        vmin=-limit,
        vmax=limit
    )

    ax.set_title(f"Continuous {column} Map for grade")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.colorbar(im, ax=ax, label="SHAP value ($1,000s)")
    fig.tight_layout()
    fig.savefig(folder / filename, dpi=200)
    plt.close(fig)

    print("Saved:", filename)

print("DONE: All 3 continuous maps generated.")
