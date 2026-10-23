
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

folder = Path(__file__).resolve().parent / "publication_result_1"
df = pd.read_csv(folder / "KingCounty_standard_vs_spatial_map_grade_600.csv")

lon = df["Longitude"].to_numpy()
lat = df["Latitude"].to_numpy()

# Grid and approximate distance correction
n = 200
gx, gy = np.meshgrid(
    np.linspace(lon.min(), lon.max(), n),
    np.linspace(lat.min(), lat.max(), n)
)

scale = np.cos(np.deg2rad(lat.mean()))
points = np.column_stack([lon * scale, lat])
grid_points = np.column_stack([gx.ravel() * scale, gy.ravel()])

tree = cKDTree(points)
dist, indices = tree.query(grid_points, k=12)

# IDW smoothing
weights = 1 / np.maximum(dist, 1e-8) ** 2
weights = weights / weights.sum(axis=1, keepdims=True)

# Do not extrapolate far beyond observed houses
mask = (dist[:, 0] * 111 > 8).reshape(gx.shape)

maps = [
    ("Standard SHAP", "Figure26_continuous_standard_SHAP_grade.png"),
    ("Spatial SHAP", "Figure27_continuous_spatial_SHAP_grade.png"),
    ("Spatial minus Standard", "Figure28_continuous_difference_SHAP_grade.png"),
]

shared_limit = np.percentile(
    np.abs(df[["Standard SHAP", "Spatial SHAP"]].to_numpy()), 98
) / 1000

for column, filename in maps:
    values = df[column].to_numpy() / 1000
    surface = np.sum(values[indices] * weights, axis=1).reshape(gx.shape)
    surface = np.ma.masked_where(mask, surface)

    limit = (
        shared_limit if column != "Spatial minus Standard"
        else np.percentile(np.abs(values), 98)
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.pcolormesh(
        gx, gy, surface,
        shading="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit
    )

    ax.scatter(lon, lat, s=2, c="black", alpha=0.12)

    ax.set_title(f"Continuous {column} Map for grade (IDW)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.colorbar(im, ax=ax, label="SHAP value ($1,000s)")
    fig.tight_layout()
    fig.savefig(folder / filename, dpi=200)
    plt.close(fig)

    print("Saved:", filename)

print("DONE")
