import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =====================================================
# Load California Housing Dataset
# =====================================================

housing = fetch_california_housing(as_frame=True)
df = housing.frame

# =====================================================
# Random Sampling
# =====================================================

sample = df.sample(n=2000, random_state=42)

# Save coordinates only for visualization
coords = sample[["Latitude", "Longitude"]].copy()

# Target
y = sample["MedHouseVal"]

# Remove target and coordinates from model input
X = sample.drop(columns=[
    "MedHouseVal",
    "Latitude",
    "Longitude"
])

print(X.columns)

# =====================================================
# Train/Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

coord_test = coords.loc[X_test.index]

# =====================================================
# Random Forest Model
# =====================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# Model Performance
# =====================================================

pred = model.predict(X_test)

r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)

print("R2:", r2)
print("RMSE:", rmse)
print("MAE:", mae)

# =====================================================
# SHAP
# =====================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

# =====================================================
# Figure 3
# Global SHAP Summary
# =====================================================

plt.figure(figsize=(8,6))

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()

plt.savefig(
    "Figure3_Global_SHAP_Summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# Global SHAP Importance
# =====================================================

importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({

    "Feature": X_test.columns,

    "Mean Absolute SHAP": importance

})

importance_df = importance_df.sort_values(
    by="Mean Absolute SHAP",
    ascending=False
)

print("\nGlobal SHAP Importance\n")

print(importance_df)

# =====================================================
# Figure 4
# Spatial SHAP - MedInc
# =====================================================

medinc_idx = X_test.columns.get_loc("MedInc")

plt.figure(figsize=(8,6))

plt.scatter(

    coord_test["Longitude"],

    coord_test["Latitude"],

    c=shap_values[:, medinc_idx],

    cmap="coolwarm",

    s=20

)

plt.colorbar(label="SHAP value")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("Spatial Distribution of SHAP Values for MedInc")

plt.tight_layout()

plt.savefig(
    "Figure4_MedInc_SHAP.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# Figure 5
# Spatial SHAP - HouseAge
# =====================================================

houseage_idx = X_test.columns.get_loc("HouseAge")

plt.figure(figsize=(8,6))

plt.scatter(

    coord_test["Longitude"],

    coord_test["Latitude"],

    c=shap_values[:, houseage_idx],

    cmap="coolwarm",

    s=20

)

plt.colorbar(label="SHAP value")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("Spatial Distribution of SHAP Values for HouseAge")

plt.tight_layout()

plt.savefig(
    "Figure5_HouseAge_SHAP.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# Figure 6
# Spatial SHAP - AveOccup
# =====================================================

occup_idx = X_test.columns.get_loc("AveOccup")

plt.figure(figsize=(8,6))

plt.scatter(

    coord_test["Longitude"],

    coord_test["Latitude"],

    c=shap_values[:, occup_idx],

    cmap="coolwarm",

    s=20

)

plt.colorbar(label="SHAP value")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("Spatial Distribution of SHAP Values for AveOccup")

plt.tight_layout()

plt.savefig(
    "Figure6_AveOccup_SHAP.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nAll figures generated successfully.")

shap.dependence_plot(
    ind="MedInc",
    shap_values=shap_values,
    features=X_test,
    interaction_index="HouseAge",
    show=False
)

plt.gcf().set_size_inches(8,6)
plt.tight_layout()
plt.savefig(
    "dependence_medinc_houseage.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()



import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial.distance import cdist

# ============================================================
# 1. Load data, train model (same as your original script)
# ============================================================

housing = fetch_california_housing(as_frame=True)
df = housing.frame

sample = df.sample(n=2000, random_state=42)
coords = sample[["Latitude", "Longitude"]].copy()
y = sample["MedHouseVal"]
X = sample.drop(columns=["MedHouseVal", "Latitude", "Longitude"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
coord_test = coords.loc[X_test.index]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Standard SHAP (global background)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("Model trained. Standard SHAP computed.")

# ============================================================
# 2. Small subset for spatial SHAP (fast on a laptop)
# ============================================================

N_SUBSET = 120   # bumped up from 60
K = 60           # bumped up from 30 (KNN neighbours)
BW = 0.5         # Gaussian bandwidth (degrees)
B = 60           # Gaussian background sample size

X_test_s = X_test.sample(N_SUBSET, random_state=42)
coord_test_s = coord_test.loc[X_test_s.index]
shap_values_s = shap_values[X_test.index.get_indexer(X_test_s.index)]

train_coords = coords.loc[X_train.index][["Latitude", "Longitude"]].values
test_coords_s = coord_test_s[["Latitude", "Longitude"]].values
dist_matrix_s = cdist(test_coords_s, train_coords)

n_features = X_test_s.shape[1]
knn_shap = np.zeros((N_SUBSET, n_features))
gauss_shap = np.zeros((N_SUBSET, n_features))
rng = np.random.default_rng(42)

for i in range(N_SUBSET):
    # KNN background
    neighbor_idx = np.argsort(dist_matrix_s[i])[:K]
    bg_knn = X_train.iloc[neighbor_idx]
    exp_knn = shap.TreeExplainer(model, data=bg_knn, feature_perturbation="interventional")
    knn_shap[i, :] = exp_knn.shap_values(X_test_s.iloc[[i]])[0]

    # Gaussian background
    w = np.exp(-(dist_matrix_s[i] ** 2) / (2 * BW ** 2))
    w = w / w.sum()
    sampled_idx = rng.choice(len(w), size=B, replace=True, p=w)
    bg_gauss = X_train.iloc[sampled_idx]
    exp_gauss = shap.TreeExplainer(model, data=bg_gauss, feature_perturbation="interventional")
    gauss_shap[i, :] = exp_gauss.shap_values(X_test_s.iloc[[i]])[0]

    if (i + 1) % 20 == 0:
        print(f"{i+1}/{N_SUBSET} done")

print("Spatial SHAP computed.")

# ============================================================
# 3. Plot: Standard vs KNN vs Gaussian for MedInc
# ============================================================

feature = "MedInc"
idx = list(X_test_s.columns).index(feature)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
panels = [
    (shap_values_s[:, idx], "Standard SHAP"),
    (knn_shap[:, idx], "KNN Spatial Shapley"),
    (gauss_shap[:, idx], "Gaussian Spatial Shapley"),
]
for ax, (values, title) in zip(axes, panels):
    sc = ax.scatter(coord_test_s["Longitude"], coord_test_s["Latitude"],
                     c=values, cmap="coolwarm", s=30)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(sc, ax=ax)

plt.suptitle(f"Comparison of SHAP Frameworks for {feature} (n={N_SUBSET})")
plt.tight_layout()
plt.savefig(f"Comparison_{feature}.png", dpi=300, bbox_inches="tight")
plt.show()

# Print value ranges so you can immediately check if spatial narrowed the range
print(f"\nStandard SHAP range: {shap_values_s[:, idx].min():.3f} to {shap_values_s[:, idx].max():.3f}")
print(f"KNN Spatial range:    {knn_shap[:, idx].min():.3f} to {knn_shap[:, idx].max():.3f}")
print(f"Gaussian Spatial range: {gauss_shap[:, idx].min():.3f} to {gauss_shap[:, idx].max():.3f}")

# ============================================================
# 4. Diagnose the most extreme KNN point
# ============================================================

values = knn_shap[:, idx]
weird_idx = np.argmax(np.abs(values))

print(f"\n--- Most extreme KNN point: row {weird_idx} ---")
print(f"KNN spatial SHAP value: {values[weird_idx]:.3f}")
print(f"Standard SHAP value for comparison: {shap_values_s[weird_idx, idx]:.3f}")
print(f"Location: {coord_test_s.iloc[weird_idx].to_dict()}")

neighbor_idx = np.argsort(dist_matrix_s[weird_idx])[:K]
neighbor_data = X_train.iloc[neighbor_idx]

print(f"\nThis house's MedInc: {X_test_s.iloc[weird_idx]['MedInc']:.3f}")
print(f"Its {K} neighbours' MedInc - mean: {neighbor_data['MedInc'].mean():.3f}, "
      f"std: {neighbor_data['MedInc'].std():.3f}, "
      f"min: {neighbor_data['MedInc'].min():.3f}, "
      f"max: {neighbor_data['MedInc'].max():.3f}")

neighbor_dists = dist_matrix_s[weird_idx][neighbor_idx]
print(f"Distance to neighbours (degrees) - min: {neighbor_dists.min():.3f}, "
      f"max: {neighbor_dists.max():.3f}, mean: {neighbor_dists.mean():.3f}")


import numpy as np
import shap
import matplotlib.pyplot as plt

N_SUBSET = X_test_s.shape[0]
rng = np.random.default_rng(42)


def compute_knn_shap(K, feature_idx=None):
    """Returns spatial SHAP values for all features (or just one column if feature_idx given)."""
    n_features = X_test_s.shape[1]
    out = np.zeros((N_SUBSET, n_features))
    for i in range(N_SUBSET):
        neighbor_idx = np.argsort(dist_matrix_s[i])[:K]
        bg = X_train.iloc[neighbor_idx]
        exp = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
        out[i, :] = exp.shap_values(X_test_s.iloc[[i]])[0]
    return out


def compute_gaussian_shap(bw, B=60):
    n_features = X_test_s.shape[1]
    out = np.zeros((N_SUBSET, n_features))
    for i in range(N_SUBSET):
        w = np.exp(-(dist_matrix_s[i] ** 2) / (2 * bw ** 2))
        w = w / w.sum()
        sampled_idx = rng.choice(len(w), size=B, replace=True, p=w)
        bg = X_train.iloc[sampled_idx]
        exp = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
        out[i, :] = exp.shap_values(X_test_s.iloc[[i]])[0]
    return out


# ============================================================
# 1. Second feature: HouseAge
# ============================================================

feature2 = "HouseAge"
idx2 = list(X_test_s.columns).index(feature2)

print(f"Computing KNN spatial SHAP for {feature2} (K=60)...")
knn_shap_f2 = compute_knn_shap(K=60)

print(f"Computing Gaussian spatial SHAP for {feature2} (bw=0.5)...")
gauss_shap_f2 = compute_gaussian_shap(bw=0.5)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
panels = [
    (shap_values_s[:, idx2], "Standard SHAP"),
    (knn_shap_f2[:, idx2], "KNN Spatial Shapley"),
    (gauss_shap_f2[:, idx2], "Gaussian Spatial Shapley"),
]
for ax, (values, title) in zip(axes, panels):
    sc = ax.scatter(coord_test_s["Longitude"], coord_test_s["Latitude"],
                     c=values, cmap="coolwarm", s=30)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(sc, ax=ax)

plt.suptitle(f"Comparison of SHAP Frameworks for {feature2} (n={N_SUBSET})")
plt.tight_layout()
plt.savefig(f"Comparison_{feature2}.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\n{feature2} - Standard SHAP range: {shap_values_s[:, idx2].min():.3f} to {shap_values_s[:, idx2].max():.3f}")
print(f"{feature2} - KNN Spatial range:    {knn_shap_f2[:, idx2].min():.3f} to {knn_shap_f2[:, idx2].max():.3f}")
print(f"{feature2} - Gaussian Spatial range: {gauss_shap_f2[:, idx2].min():.3f} to {gauss_shap_f2[:, idx2].max():.3f}")


# ============================================================
# 2. Sensitivity analysis - KNN, 2 levels (K=30 vs K=90)
# ============================================================

feature_sens = "MedInc"
idx_sens = list(X_test_s.columns).index(feature_sens)

print("\nRunning KNN sensitivity (K=30)...")
knn_K30 = compute_knn_shap(K=30)
print("Running KNN sensitivity (K=90)...")
knn_K90 = compute_knn_shap(K=90)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (values, title) in zip(axes, [
    (knn_K30[:, idx_sens], "K = 30"),
    (knn_K90[:, idx_sens], "K = 90"),
]):
    sc = ax.scatter(coord_test_s["Longitude"], coord_test_s["Latitude"],
                     c=values, cmap="coolwarm", s=30)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(sc, ax=ax)

plt.suptitle(f"Sensitivity Analysis: KNN Spatial Shapley for {feature_sens}")
plt.tight_layout()
plt.savefig(f"Sensitivity_KNN_{feature_sens}.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nK=30 range: {knn_K30[:, idx_sens].min():.3f} to {knn_K30[:, idx_sens].max():.3f}")
print(f"K=90 range: {knn_K90[:, idx_sens].min():.3f} to {knn_K90[:, idx_sens].max():.3f}")


# ============================================================
# 3. Sensitivity analysis - Gaussian, 2 levels (bw=0.2 vs bw=1.0)
# ============================================================

print("\nRunning Gaussian sensitivity (bandwidth=0.2)...")
gauss_bw_narrow = compute_gaussian_shap(bw=0.2)
print("Running Gaussian sensitivity (bandwidth=1.0)...")
gauss_bw_wide = compute_gaussian_shap(bw=1.0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (values, title) in zip(axes, [
    (gauss_bw_narrow[:, idx_sens], "Bandwidth = 0.2"),
    (gauss_bw_wide[:, idx_sens], "Bandwidth = 1.0"),
]):
    sc = ax.scatter(coord_test_s["Longitude"], coord_test_s["Latitude"],
                     c=values, cmap="coolwarm", s=30)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(sc, ax=ax)

plt.suptitle(f"Sensitivity Analysis: Gaussian Spatial Shapley for {feature_sens}")
plt.tight_layout()
plt.savefig(f"Sensitivity_Gaussian_{feature_sens}.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nBandwidth=0.2 range: {gauss_bw_narrow[:, idx_sens].min():.3f} to {gauss_bw_narrow[:, idx_sens].max():.3f}")
print(f"Bandwidth=1.0 range: {gauss_bw_wide[:, idx_sens].min():.3f} to {gauss_bw_wide[:, idx_sens].max():.3f}")

# ============================================================
# Figure X:
# Standard SHAP vs KNN Spatial SHAP vs Gaussian Spatial SHAP
# for MedInc
# ============================================================

feature_compare = "MedInc"

feature_idx = list(X_test_s.columns).index(feature_compare)


# ------------------------------------------------------------
# Select same observations
# ------------------------------------------------------------

standard_medinc = shap_values_s[:, feature_idx]
knn_medinc = knn_shap[:, feature_idx]
gaussian_medinc = gauss_shap[:, feature_idx]


# ------------------------------------------------------------
# Create comparison table
# ------------------------------------------------------------

comparison_df = pd.DataFrame({

    "Standard SHAP": standard_medinc,

    "KNN Spatial SHAP": knn_medinc,

    "Gaussian Spatial SHAP": gaussian_medinc

})


print("\nSHAP comparison for MedInc")
print(comparison_df.head())


comparison_df.to_csv(
    "California_MedInc_SHAP_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# Plot comparison
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)


plots = [
    (
        standard_medinc,
        "Standard SHAP"
    ),
    (
        knn_medinc,
        "KNN Spatial SHAP"
    ),
    (
        gaussian_medinc,
        "Gaussian Spatial SHAP"
    )
]


for ax, (values, title) in zip(axes, plots):

    scatter = ax.scatter(
        coord_test_s["Longitude"],
        coord_test_s["Latitude"],
        c=values,
        cmap="coolwarm",
        s=35
    )

    ax.set_title(title)

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )

    plt.colorbar(
        scatter,
        ax=ax,
        label="SHAP value"
    )


plt.suptitle(
    "Comparison of SHAP Explanations for MedInc\nCalifornia Housing Dataset"
)


plt.tight_layout()


plt.savefig(
    "Figure_Standard_KNN_Gaussian_MedInc_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------

summary = pd.DataFrame({

    "Method": [
        "Standard SHAP",
        "KNN Spatial SHAP",
        "Gaussian Spatial SHAP"
    ],

    "Mean absolute SHAP": [

        np.mean(np.abs(standard_medinc)),

        np.mean(np.abs(knn_medinc)),

        np.mean(np.abs(gaussian_medinc))

    ],

    "Minimum": [

        standard_medinc.min(),

        knn_medinc.min(),

        gaussian_medinc.min()

    ],

    "Maximum": [

        standard_medinc.max(),

        knn_medinc.max(),

        gaussian_medinc.max()

    ]

})


print("\nSummary statistics:")
print(summary.round(3))


summary.to_csv(
    "California_MedInc_SHAP_summary.csv",
    index=False
)