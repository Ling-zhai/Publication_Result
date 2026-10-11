import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from scipy.spatial import KDTree
from tqdm import tqdm
OUTPUT_DIR = "publication_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def haversine_distance_matrix(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distances between two sets of coordinates.
    Input latitudes and longitudes are in degrees.
    Output distances are in kilometres.
    """
    R = 6371.0

    lat1 = np.radians(lat1)[:, None]
    lon1 = np.radians(lon1)[:, None]
    lat2 = np.radians(lat2)[None, :]
    lon2 = np.radians(lon2)[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))

# ============================================================
# 1. Load and prepare data
# ============================================================

df = pd.read_csv("kc_house_data.csv")

df = df.drop(columns=["id", "date"])

# Use the same 3,000 observations as the previous experiments
sample = df.sample(
    n=3000,
    random_state=42
).copy()

# Spatial coordinates are stored separately.
# They are not included in the prediction model.
coords = sample[["lat", "long"]].copy()

# Target
y = sample["price"].copy()

# Non-spatial model features
X = sample.drop(
    columns=["price", "lat", "long", "zipcode"]
).copy()

print("Model features:")
print(X.columns.tolist())


# ============================================================
# 2. Train-test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

coord_train = coords.loc[X_train.index].copy()
coord_test = coords.loc[X_test.index].copy()

print("\nTraining observations:", len(X_train))
print("Test observations:", len(X_test))


# ============================================================
# 3. Train Random Forest
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print("\nModel performance:")
print("R2:", round(r2, 4))
print("RMSE:", round(rmse, 2))
print("MAE:", round(mae, 2))


# ============================================================
# 4. Select 50 test houses
# ============================================================

# Fifty houses are randomly selected from the test set.
# The same random_state makes the selection reproducible.
selected_indices = X_test.sample(
    n=50,
    random_state=42
).index

X_selected = X_test.loc[selected_indices].copy()
y_selected = y_test.loc[selected_indices].copy()
coord_selected = coord_test.loc[selected_indices].copy()

print("\nSelected houses:", len(X_selected))


# ============================================================
# 5. Standard SHAP for the selected houses
# ============================================================

# Use one global training background
global_background = X_train.sample(
    n=100,
    random_state=42
)

standard_explainer = shap.TreeExplainer(
    model,
    data=global_background,
    feature_perturbation="interventional"
)

standard_shap_values = standard_explainer.shap_values(
    X_selected,
    check_additivity=False
)

standard_shap_values = np.asarray(standard_shap_values)

print("\nStandard SHAP completed.")
print("Standard SHAP shape:", standard_shap_values.shape)


# ============================================================
# 6. KNN Spatial SHAP for the selected houses
# ============================================================

K_NEIGHBOURS = 20

# Compute great-circle distances from selected test houses to training houses
knn_distance_matrix = haversine_distance_matrix(
    coord_selected["lat"].to_numpy(),
    coord_selected["long"].to_numpy(),
    coord_train["lat"].to_numpy(),
    coord_train["long"].to_numpy()
)

spatial_shap_values = np.zeros(
    shape=(len(X_selected), X_selected.shape[1]),
    dtype=float
)
# Store local mean prediction and location effect
train_predictions = model.predict(X_train)
global_mean_prediction = np.mean(train_predictions)

local_mean_predictions = np.zeros(
    shape=len(X_selected),
    dtype=float
)

location_effects = np.zeros(
    shape=len(X_selected),
    dtype=float
)

for i in tqdm(
    range(len(X_selected)),
    desc="KNN Spatial SHAP"
):
    current_x = X_selected.iloc[[i]]

    neighbour_positions = np.argsort(
        knn_distance_matrix[i]
    )[:min(K_NEIGHBOURS, len(X_train))]

    neighbour_positions = np.atleast_1d(
        neighbour_positions
    )

    local_background = X_train.iloc[
        neighbour_positions
    ]
    local_background_predictions = model.predict(
        local_background
    )

    local_mean_predictions[i] = np.mean(
        local_background_predictions
    )

    location_effects[i] = (
            local_mean_predictions[i]
            - global_mean_prediction
    )

    local_explainer = shap.TreeExplainer(
        model,
        data=local_background,
        feature_perturbation="interventional"
    )

    local_values = local_explainer.shap_values(
        current_x,
        check_additivity=False
    )

    spatial_shap_values[i, :] = np.asarray(
        local_values
    )[0]

print("\nKNN Spatial SHAP completed.")
print("Spatial SHAP shape:", spatial_shap_values.shape)


# ============================================================
# 7. Extract grade attribution
# ============================================================

grade_index = X_selected.columns.get_loc("grade")

standard_grade = standard_shap_values[:, grade_index]
spatial_grade = spatial_shap_values[:, grade_index]

difference_grade = spatial_grade - standard_grade
absolute_difference = np.abs(difference_grade)


# ============================================================
# 8. Create the complete 50-house comparison table
# ============================================================

comparison_table = pd.DataFrame({
    "Original index": X_selected.index,
    "Grade": X_selected["grade"].to_numpy(),
    "Living area": X_selected["sqft_living"].to_numpy(),
    "Observed price": y_selected.to_numpy(),
    "Latitude": coord_selected["lat"].to_numpy(),
    "Longitude": coord_selected["long"].to_numpy(),
    "Global mean prediction": global_mean_prediction,
    "Local mean prediction": local_mean_predictions,
    "Location effect": location_effects,
    "Standard SHAP": standard_grade,
    "Spatial SHAP": spatial_grade,
    "Spatial minus Standard": difference_grade,
    "Absolute difference": absolute_difference
})

# Sort the table by the largest attribution difference
comparison_table = comparison_table.sort_values(
    by="Absolute difference",
    ascending=False
).reset_index(drop=True)

print("\nComplete comparison table for 50 houses:\n")

print(
    comparison_table.round(2).to_string(index=False)
)

comparison_table.to_csv(
    "standard_vs_spatial_shap_50_houses.csv",
    index=False
)
location_effect_table = pd.DataFrame({
    "Original index": X_selected.index,
    "Latitude": coord_selected["lat"].to_numpy(),
    "Longitude": coord_selected["long"].to_numpy(),
    "Observed price": y_selected.to_numpy(),
    "Predicted price": model.predict(X_selected),
    "Global mean prediction": global_mean_prediction,
    "Local mean prediction": local_mean_predictions,
    "Location effect": location_effects
})

location_effect_table.to_csv(
    os.path.join(OUTPUT_DIR, "KingCounty_location_effects_50_houses.csv"),
    index=False
)

print(
    "\nLocation effect table saved as: "
    "KingCounty_location_effects_50_houses.csv"
)
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    coord_selected["long"],
    coord_selected["lat"],
    c=location_effects,
    s=65,
    alpha=0.85
)

plt.colorbar(
    scatter,
    label="Location effect"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Location Effect for 50 Test Houses")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure14_location_effect_50_houses.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nFigure saved as: "
    "Figure14_location_effect_50_houses.png"
)
location_effect_summary = pd.DataFrame({
    "Statistic": [
        "Number of houses",
        "Mean location effect",
        "Median location effect",
        "Minimum location effect",
        "Maximum location effect",
        "Mean absolute location effect"
    ],
    "Value": [
        len(X_selected),
        np.mean(location_effects),
        np.median(location_effects),
        np.min(location_effects),
        np.max(location_effects),
        np.mean(np.abs(location_effects))
    ]
})

print("\nLocation effect summary:\n")
print(location_effect_summary.round(2).to_string(index=False))

location_effect_summary.to_csv(
    os.path.join(OUTPUT_DIR, "KingCounty_location_effect_summary.csv"),
    index=False
)

print(
    "\nLocation effect summary saved as: "
    "KingCounty_location_effect_summary.csv"
)

print(
    "\nFull table saved as: "
    "standard_vs_spatial_shap_50_houses.csv"
)


# ============================================================
# 9. Create a smaller case-study table
# ============================================================

# Select the five houses with the largest differences
case_study_table = comparison_table.head(5).copy()

print("\nFive houses with the largest differences:\n")

print(
    case_study_table.round(2).to_string(index=False)
)

case_study_table.to_csv(
    os.path.join(OUTPUT_DIR, "spatial_shap_case_study_top5.csv"),
    index=False
)

print(
    "\nCase-study table saved as: "
    "spatial_shap_case_study_top5.csv"
)


# ============================================================
# 10. Figure 11:
# Standard SHAP versus KNN Spatial SHAP
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    standard_grade,
    spatial_grade,
    s=55,
    alpha=0.8
)

minimum_value = min(
    standard_grade.min(),
    spatial_grade.min()
)

maximum_value = max(
    standard_grade.max(),
    spatial_grade.max()
)

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    linestyle="--",
    linewidth=1.5,
    label="Equal attribution"
)

plt.xlabel("Standard SHAP value for grade")
plt.ylabel("KNN Spatial SHAP value for grade")
plt.title(
    "Standard SHAP and KNN Spatial SHAP "
    "for 50 Test Houses"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure11_standard_vs_spatial_grade_50.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nFigure 11 saved as: "
    "Figure11_standard_vs_spatial_grade_50.png"
)


# ============================================================
# 11. Overall summary statistics
# ============================================================

sign_changes = (
    np.sign(standard_grade)
    != np.sign(spatial_grade)
)

summary_table = pd.DataFrame({
    "Statistic": [
        "Number of houses",
        "Mean absolute difference",
        "Median absolute difference",
        "Maximum absolute difference",
        "Sign-change rate (%)"
    ],
    "Value": [
        len(X_selected),
        np.mean(absolute_difference),
        np.median(absolute_difference),
        np.max(absolute_difference),
        np.mean(sign_changes) * 100
    ]
})

print("\nOverall summary:\n")

print(
    summary_table.round(2).to_string(index=False)
)

summary_table.to_csv(
    os.path.join(OUTPUT_DIR, "spatial_shap_summary_statistics.csv"),
    index=False
)

print(
    "\nSummary table saved as: "
    "spatial_shap_summary_statistics.csv"
)

print("\nAll analyses completed successfully.")

# ============================================================
# 12. Gaussian Kernel Spatial SHAP
# Standard SHAP vs Gaussian Spatial SHAP
# ============================================================

print("\n--- Starting Gaussian Spatial SHAP ---")


# Use the same 50 houses
gaussian_shap_values = np.zeros(
    shape=(len(X_selected), X_selected.shape[1]),
    dtype=float
)


# Compute great-circle distances from selected test houses to training houses
gaussian_distance_matrix = haversine_distance_matrix(
    coord_selected["lat"].to_numpy(),
    coord_selected["long"].to_numpy(),
    coord_train["lat"].to_numpy(),
    coord_train["long"].to_numpy()
)


# Gaussian parameters
N_NEIGHBOURS = 200
bandwidth = 5


for i in tqdm(
    range(len(X_selected)),
    desc="Gaussian Spatial SHAP"
):

    current_x = X_selected.iloc[[i]]

    # Find nearby training houses using great-circle distance
    neighbour_positions = np.argsort(
        gaussian_distance_matrix[i]
    )[:N_NEIGHBOURS]

    distances = gaussian_distance_matrix[i][
        neighbour_positions
    ]


    neighbour_positions = np.atleast_1d(
        neighbour_positions
    )


    local_background = X_train.iloc[
        neighbour_positions
    ]


    # Gaussian weights

    weights = np.exp(
        -(distances ** 2)
        /
        (2 * bandwidth ** 2)
    )


    weights = weights / np.sum(weights)


    # Weighted sampling background

    sampled_positions = np.random.choice(
        len(local_background),
        size=100,
        replace=True,
        p=weights
    )


    gaussian_background = (
        local_background.iloc[sampled_positions]
    )


    # SHAP

    gaussian_explainer = shap.TreeExplainer(
        model,
        data=gaussian_background,
        feature_perturbation="interventional"
    )


    gaussian_values = gaussian_explainer.shap_values(
        current_x,
        check_additivity=False
    )


    gaussian_shap_values[i, :] = np.asarray(
        gaussian_values
    )[0]


print("\nGaussian Spatial SHAP completed")

print(
    "Gaussian SHAP shape:",
    gaussian_shap_values.shape
)



# ============================================================
# 13. Compare grade feature
# ============================================================

grade_index = X_selected.columns.get_loc("grade")


standard_grade = standard_shap_values[:, grade_index]

gaussian_grade = gaussian_shap_values[:, grade_index]


difference = (
    gaussian_grade - standard_grade
)


absolute_difference = np.abs(
    difference
)



# ============================================================
# 14. Comparison table
# ============================================================


gaussian_table = pd.DataFrame({

    "Original index": X_selected.index,

    "Grade": X_selected["grade"].to_numpy(),

    "Living area": X_selected["sqft_living"].to_numpy(),

    "Observed price": y_selected.to_numpy(),

    "Latitude": coord_selected["lat"].to_numpy(),

    "Longitude": coord_selected["long"].to_numpy(),

    "Standard SHAP": standard_grade,

    "Gaussian Spatial SHAP": gaussian_grade,

    "Gaussian minus Standard": difference,

    "Absolute difference": absolute_difference

})


gaussian_table = gaussian_table.sort_values(
    by="Absolute difference",
    ascending=False
)


print("\nGaussian comparison table:")
print(
    gaussian_table.round(2).to_string(index=False)
)


gaussian_table.to_csv(
    os.path.join(OUTPUT_DIR, "standard_vs_gaussian_shap_50_houses.csv"),
    index=False
)



# ============================================================
# 15. Figure 12
# ============================================================


plt.figure(figsize=(8,6))


plt.scatter(
    standard_grade,
    gaussian_grade,
    s=55,
    alpha=0.8
)


min_value = min(
    standard_grade.min(),
    gaussian_grade.min()
)


max_value = max(
    standard_grade.max(),
    gaussian_grade.max()
)


plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=1.5,
    label="Equal attribution"
)


plt.xlabel(
    "Standard SHAP value for grade"
)


plt.ylabel(
    "Gaussian Spatial SHAP value for grade"
)


plt.title(
    "Standard SHAP and Gaussian Spatial SHAP "
    "for 50 Test Houses"
)


plt.legend()

plt.tight_layout()


plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure12_standard_vs_gaussian_grade_50.png"),
    dpi=300,
    bbox_inches="tight"
)


plt.show()



# ============================================================
# 16. Summary statistics
# ============================================================


summary_gaussian = pd.DataFrame({

    "Statistic":[
        "Number of houses",
        "Mean absolute difference",
        "Median absolute difference",
        "Maximum absolute difference"
    ],

    "Value":[
        len(X_selected),
        np.mean(absolute_difference),
        np.median(absolute_difference),
        np.max(absolute_difference)
    ]

})


print("\nGaussian summary:")
print(
    summary_gaussian.round(2).to_string(index=False)
)


summary_gaussian.to_csv(
    os.path.join(OUTPUT_DIR, "gaussian_shap_summary_statistics.csv"),
    index=False
)


print("\nGaussian analysis finished.")

# ============================================================
# KNN Spatial SHAP Sensitivity Analysis
# Pairwise stability across K = 30, 60, 90 for top 5 features
# ============================================================

import itertools
from scipy.stats import pearsonr, spearmanr

# ------------------------------------------------------------
# 1. Parameter settings
# ------------------------------------------------------------

K_values = [10, 15, 20, 25, 30]

# Select top 5 features according to Standard SHAP mean absolute value
mean_abs_standard_shap = np.abs(standard_shap_values).mean(axis=0)

top_feature_indices = np.argsort(mean_abs_standard_shap)[::-1][:5]

top_features = X_selected.columns[top_feature_indices].tolist()

print("\nTop 5 features based on Standard SHAP:")
print(top_features)


# ------------------------------------------------------------
# 2. Compute great-circle distances for sensitivity analysis
# ------------------------------------------------------------

sensitivity_distance_matrix = haversine_distance_matrix(
    coord_selected["lat"].to_numpy(),
    coord_selected["long"].to_numpy(),
    coord_train["lat"].to_numpy(),
    coord_train["long"].to_numpy()
)


# ------------------------------------------------------------
# 3. Calculate KNN Spatial SHAP for each K
# ------------------------------------------------------------

# Store full Spatial SHAP matrices for each K
spatial_shap_by_k = {}

for K in K_values:

    print(f"\nRunning KNN Spatial SHAP sensitivity for K = {K}")

    spatial_values_K = np.zeros(
        shape=(len(X_selected), X_selected.shape[1]),
        dtype=float
    )

    for i in tqdm(
        range(len(X_selected)),
        desc=f"K={K}"
    ):

        current_x = X_selected.iloc[[i]]

        neighbour_positions = np.argsort(
            sensitivity_distance_matrix[i]
        )[:min(K, len(X_train))]

        neighbour_positions = np.atleast_1d(
            neighbour_positions
        )

        local_background = X_train.iloc[
            neighbour_positions
        ]

        local_explainer = shap.TreeExplainer(
            model,
            data=local_background,
            feature_perturbation="interventional"
        )

        local_values = local_explainer.shap_values(
            current_x,
            check_additivity=False
        )

        spatial_values_K[i, :] = np.asarray(
            local_values
        )[0]

    spatial_shap_by_k[K] = spatial_values_K.copy()

    np.save(
        os.path.join(OUTPUT_DIR, f"KingCounty_spatial_shap_K{K}.npy"),
        spatial_values_K
    )

print("\nAll KNN sensitivity calculations completed.")


# ------------------------------------------------------------
# 4. Pairwise comparison for top 5 features
# ------------------------------------------------------------

pairwise_results = []

for feature_name in top_features:

    feature_index = X_selected.columns.get_loc(feature_name)

    for K1, K2 in itertools.combinations(K_values, 2):

        values_1 = spatial_shap_by_k[K1][:, feature_index]
        values_2 = spatial_shap_by_k[K2][:, feature_index]

        pearson_correlation = pearsonr(
            values_1,
            values_2
        ).statistic

        spearman_correlation = spearmanr(
            values_1,
            values_2
        ).statistic

        sign_agreement = np.mean(
            np.sign(values_1) == np.sign(values_2)
        ) * 100

        mean_absolute_difference = np.mean(
            np.abs(values_1 - values_2)
        )

        pairwise_results.append({
            "Feature": feature_name,
            "Comparison": f"K={K1} vs K={K2}",
            "Pearson correlation": pearson_correlation,
            "Spearman correlation": spearman_correlation,
            "Sign agreement (%)": sign_agreement,
            "Mean absolute difference": mean_absolute_difference
        })


sensitivity_table = pd.DataFrame(
    pairwise_results
)

print("\nPairwise KNN sensitivity results for top 5 features:\n")

print(
    sensitivity_table.round(4).to_string(
        index=False
    )
)

sensitivity_table.to_csv(
    os.path.join(OUTPUT_DIR, "KingCounty_KNN_pairwise_sensitivity_top5.csv"),
    index=False
)

print(
    "\nTable saved as: "
    "KingCounty_KNN_pairwise_sensitivity_top5.csv"
)


# ------------------------------------------------------------
# 5. Summary table across top 5 features
# ------------------------------------------------------------

summary_sensitivity = (
    sensitivity_table
    .groupby("Comparison")
    .agg({
        "Pearson correlation": "mean",
        "Spearman correlation": "mean",
        "Sign agreement (%)": "mean",
        "Mean absolute difference": "mean"
    })
    .reset_index()
)

print("\nAverage sensitivity results across top 5 features:\n")

print(
    summary_sensitivity.round(4).to_string(
        index=False
    )
)

summary_sensitivity.to_csv(
    os.path.join(OUTPUT_DIR, "KingCounty_KNN_sensitivity_top5_summary.csv"),
    index=False
)


# ------------------------------------------------------------
# 6. Pearson correlation matrix for the most important feature
# ------------------------------------------------------------

main_feature = top_features[0]
main_feature_index = X_selected.columns.get_loc(main_feature)

pearson_matrix = np.zeros(
    shape=(len(K_values), len(K_values))
)

for i, K1 in enumerate(K_values):

    for j, K2 in enumerate(K_values):

        pearson_matrix[i, j] = pearsonr(
            spatial_shap_by_k[K1][:, main_feature_index],
            spatial_shap_by_k[K2][:, main_feature_index]
        ).statistic


pearson_table = pd.DataFrame(
    pearson_matrix,
    index=[f"K={K}" for K in K_values],
    columns=[f"K={K}" for K in K_values]
)

pearson_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        f"KingCounty_KNN_correlation_matrix_{main_feature}.csv"
    )
)

print(f"\nPearson correlation matrix for {main_feature}:\n")
print(pearson_table.round(4))


# ------------------------------------------------------------
# 7. Correlation heatmap for the most important feature
# ------------------------------------------------------------

plt.figure(figsize=(6, 5))

image = plt.imshow(
    pearson_matrix,
    vmin=0,
    vmax=1
)

plt.colorbar(
    image,
    label="Pearson correlation"
)

plt.xticks(
    range(len(K_values)),
    [f"K={K}" for K in K_values]
)

plt.yticks(
    range(len(K_values)),
    [f"K={K}" for K in K_values]
)

for i in range(len(K_values)):

    for j in range(len(K_values)):

        plt.text(
            j,
            i,
            f"{pearson_matrix[i, j]:.3f}",
            ha="center",
            va="center"
        )

plt.xlabel("Neighbourhood size")
plt.ylabel("Neighbourhood size")

plt.title(
    f"Correlation of KNN Spatial SHAP Values for {main_feature}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        f"Figure13_KNN_correlation_sensitivity_{main_feature}.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nFigure saved as: "
    f"Figure13_KNN_correlation_sensitivity_{main_feature}.png"
)


# ------------------------------------------------------------
# 8. Save attribution values for top 5 features
# ------------------------------------------------------------

attribution_table = pd.DataFrame({
    "Original index": X_selected.index
})

for feature_name in top_features:

    feature_index = X_selected.columns.get_loc(feature_name)

    for K in K_values:

        attribution_table[f"{feature_name}_K={K}"] = (
            spatial_shap_by_k[K][:, feature_index]
        )

attribution_table.to_csv(
    os.path.join(OUTPUT_DIR, "KingCounty_KNN_sensitivity_top5_values.csv"),
    index=False
)

print(
    "\nAttribution values saved as: "
    "KingCounty_KNN_sensitivity_top5_values.csv"
)

print("\nKNN sensitivity analysis for top 5 features finished successfully.")

# ============================================================
# Expanded location effect map and decomposition check
# Using 300 test houses
# ============================================================

print("\n--- Starting expanded location effect and decomposition check ---")

# ------------------------------------------------------------
# 1. Select a larger sample of test houses
# ------------------------------------------------------------

N_LARGE = min(300, len(X_test))

selected_indices_large = X_test.sample(
    n=N_LARGE,
    random_state=123
).index

X_large = X_test.loc[selected_indices_large].copy()
y_large = y_test.loc[selected_indices_large].copy()
coord_large = coord_test.loc[selected_indices_large].copy()

print("\nExpanded sample selected:")
print("Number of houses:", len(X_large))


# ------------------------------------------------------------
# 2. Standard SHAP for the larger sample
# ------------------------------------------------------------

standard_shap_large = standard_explainer.shap_values(
    X_large,
    check_additivity=False
)

standard_shap_large = np.asarray(
    standard_shap_large
)

print("\nStandard SHAP for expanded sample completed.")
print("Standard SHAP large shape:", standard_shap_large.shape)


# ------------------------------------------------------------
# 3. KNN Spatial SHAP for the larger sample
# ------------------------------------------------------------

K_LARGE = 20

large_distance_matrix = haversine_distance_matrix(
    coord_large["lat"].to_numpy(),
    coord_large["long"].to_numpy(),
    coord_train["lat"].to_numpy(),
    coord_train["long"].to_numpy()
)

spatial_shap_large = np.zeros(
    shape=(len(X_large), X_large.shape[1]),
    dtype=float
)

local_base_values_large = np.zeros(
    shape=len(X_large),
    dtype=float
)

for i in tqdm(
    range(len(X_large)),
    desc="Expanded KNN Spatial SHAP"
):

    current_x = X_large.iloc[[i]]

    neighbour_positions = np.argsort(
        large_distance_matrix[i]
    )[:min(K_LARGE, len(X_train))]

    neighbour_positions = np.atleast_1d(
        neighbour_positions
    )

    local_background = X_train.iloc[
        neighbour_positions
    ]

    local_explainer = shap.TreeExplainer(
        model,
        data=local_background,
        feature_perturbation="interventional"
    )

    local_values = local_explainer.shap_values(
        current_x,
        check_additivity=False
    )

    spatial_shap_large[i, :] = np.asarray(
        local_values
    )[0]

    local_base_values_large[i] = float(
        np.ravel(local_explainer.expected_value)[0]
    )


print("\nExpanded KNN Spatial SHAP completed.")
print("Spatial SHAP large shape:", spatial_shap_large.shape)


# ------------------------------------------------------------
# 4. Location effect for the larger sample
# ------------------------------------------------------------

global_base_value = float(
    np.ravel(standard_explainer.expected_value)[0]
)

location_effect_large = (
    local_base_values_large
    - global_base_value
)

location_effect_large_table = pd.DataFrame({
    "Original index": X_large.index,
    "Latitude": coord_large["lat"].to_numpy(),
    "Longitude": coord_large["long"].to_numpy(),
    "Observed price": y_large.to_numpy(),
    "Predicted price": model.predict(X_large),
    "Global baseline": global_base_value,
    "Local baseline": local_base_values_large,
    "Location effect": location_effect_large
})

location_effect_large_table.to_csv(
    os.path.join(OUTPUT_DIR, "KingCounty_location_effect_300_houses.csv"),
    index=False
)

print(
    "\nLocation effect table saved as: "
    "KingCounty_location_effect_300_houses.csv"
)


# ------------------------------------------------------------
# 5. Location effect map for 300 houses
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    coord_large["long"],
    coord_large["lat"],
    c=location_effect_large,
    s=45,
    alpha=0.85
)

plt.colorbar(
    scatter,
    label="Location effect"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Location Effect for 300 Test Houses")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure14_location_effect_300_houses.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nFigure saved as: "
    "Figure14_location_effect_300_houses.png"
)


# ------------------------------------------------------------
# 6. Location effect summary
# ------------------------------------------------------------

location_effect_large_summary = pd.DataFrame({
    "Statistic": [
        "Number of houses",
        "Mean location effect",
        "Median location effect",
        "Minimum location effect",
        "Maximum location effect",
        "Mean absolute location effect"
    ],
    "Value": [
        len(X_large),
        np.mean(location_effect_large),
        np.median(location_effect_large),
        np.min(location_effect_large),
        np.max(location_effect_large),
        np.mean(np.abs(location_effect_large))
    ]
})

print("\nLocation effect summary for 300 houses:\n")
print(
    location_effect_large_summary.round(2).to_string(
        index=False
    )
)

location_effect_large_summary.to_csv(
    os.path.join(OUTPUT_DIR, "KingCounty_location_effect_summary_300_houses.csv"),
    index=False
)

print(
    "\nLocation effect summary saved as: "
    "KingCounty_location_effect_summary_300_houses.csv"
)


# ------------------------------------------------------------
# 7. Decomposition check
# Standard SHAP total ≈ location effect + Spatial SHAP total
# ------------------------------------------------------------

standard_total_large = np.sum(
    standard_shap_large,
    axis=1
)

spatial_total_large = np.sum(
    spatial_shap_large,
    axis=1
)

right_hand_side_large = (
    location_effect_large
    + spatial_total_large
)

decomposition_error_large = (
    standard_total_large
    - right_hand_side_large
)

predicted_large = model.predict(
    X_large
)

standard_reconstruction_large = (
    global_base_value
    + standard_total_large
)

spatial_reconstruction_large = (
    local_base_values_large
    + spatial_total_large
)

standard_prediction_error_large = (
    predicted_large
    - standard_reconstruction_large
)

spatial_prediction_error_large = (
    predicted_large
    - spatial_reconstruction_large
)

decomposition_check_table = pd.DataFrame({
    "Original index": X_large.index,
    "Predicted price": predicted_large,
    "Global baseline": global_base_value,
    "Local baseline": local_base_values_large,
    "Location effect": location_effect_large,
    "Standard SHAP total": standard_total_large,
    "Spatial SHAP total": spatial_total_large,
    "Location effect plus Spatial SHAP total": right_hand_side_large,
    "Decomposition error": decomposition_error_large,
    "Standard reconstruction": standard_reconstruction_large,
    "Spatial reconstruction": spatial_reconstruction_large,
    "Standard prediction error": standard_prediction_error_large,
    "Spatial prediction error": spatial_prediction_error_large
})

decomposition_check_table.to_csv(
    os.path.join(OUTPUT_DIR, "KingCounty_decomposition_check_300_houses.csv"),
    index=False
)

print(
    "\nDecomposition check table saved as: "
    "KingCounty_decomposition_check_300_houses.csv"
)


# ------------------------------------------------------------
# 8. Decomposition summary
# ------------------------------------------------------------

decomposition_summary = pd.DataFrame({
    "Statistic": [
        "Number of houses",
        "Mean decomposition error",
        "Mean absolute decomposition error",
        "Median absolute decomposition error",
        "Maximum absolute decomposition error",
        "RMSE of decomposition error",
        "Mean absolute standard prediction error",
        "Mean absolute spatial prediction error"
    ],
    "Value": [
        len(X_large),
        np.mean(decomposition_error_large),
        np.mean(np.abs(decomposition_error_large)),
        np.median(np.abs(decomposition_error_large)),
        np.max(np.abs(decomposition_error_large)),
        np.sqrt(np.mean(decomposition_error_large ** 2)),
        np.mean(np.abs(standard_prediction_error_large)),
        np.mean(np.abs(spatial_prediction_error_large))
    ]
})

print("\nDecomposition summary:\n")
print(
    decomposition_summary.round(6).to_string(
        index=False
    )
)

decomposition_summary.to_csv(
    os.path.join(OUTPUT_DIR, "KingCounty_decomposition_summary_300_houses.csv"),
    index=False
)

print(
    "\nDecomposition summary saved as: "
    "KingCounty_decomposition_summary_300_houses.csv"
)


# ------------------------------------------------------------
# 9. Decomposition error histogram
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    decomposition_error_large,
    bins=30
)

plt.xlabel("Decomposition error")
plt.ylabel("Frequency")
plt.title("Distribution of Decomposition Error for 300 Test Houses")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure15_decomposition_error_300_houses.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nFigure saved as: "
    "Figure15_decomposition_error_300_houses.png"
)

print("\nExpanded location effect and decomposition check completed.")

# ============================================================
# K selection using cross-validation-type criterion
# ============================================================

print("\n--- Starting K selection using cross-validation criterion ---")

# Candidate K values
K_CV_VALUES = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]

# Model predictions on training data
train_predictions = model.predict(X_train)

# Great-circle distance matrix among training locations
train_distance_matrix = haversine_distance_matrix(
    coord_train["lat"].to_numpy(),
    coord_train["long"].to_numpy(),
    coord_train["lat"].to_numpy(),
    coord_train["long"].to_numpy()
)

# Do not allow each observation to use itself as its nearest neighbour
np.fill_diagonal(train_distance_matrix, np.inf)

cv_results = []

for K in K_CV_VALUES:

    print(f"\nCalculating CV criterion for K = {K}")

    local_prediction_estimates = np.zeros(
        shape=len(X_train),
        dtype=float
    )

    for i in tqdm(
        range(len(X_train)),
        desc=f"CV K={K}"
    ):

        neighbour_positions = np.argsort(
            train_distance_matrix[i]
        )[:min(K, len(X_train) - 1)]

        local_prediction_estimates[i] = np.mean(
            train_predictions[neighbour_positions]
        )

    errors = (
        train_predictions
        - local_prediction_estimates
    )

    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    cv_results.append({
        "K": K,
        "CV MSE": mse,
        "CV RMSE": rmse,
        "CV MAE": mae
    })


cv_table = pd.DataFrame(cv_results)

best_row = cv_table.loc[
    cv_table["CV RMSE"].idxmin()
]

best_K = int(best_row["K"])

print("\nK cross-validation results:\n")
print(
    cv_table.round(4).to_string(index=False)
)

print("\nBest K based on CV RMSE:", best_K)

cv_table.to_csv(
    os.path.join(OUTPUT_DIR, "KingCounty_K_selection_CV.csv"),
    index=False
)

print(
    "\nK selection table saved as: "
    "KingCounty_K_selection_CV.csv"
)


# ------------------------------------------------------------
# Plot K selection curve
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    cv_table["K"],
    cv_table["CV RMSE"],
    marker="o"
)

plt.axvline(
    best_K,
    linestyle="--",
    label=f"Best K = {best_K}"
)

plt.xlabel("Neighbourhood size K")
plt.ylabel("CV RMSE")
plt.title("K Selection Using Cross-Validation Criterion")

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure16_K_selection_CV.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nFigure saved as: "
    "Figure16_K_selection_CV.png"
)

print("\nK selection using cross-validation criterion completed.")

# ============================================================
# Standard SHAP map vs Spatial SHAP map
# For grade feature, using 300 test houses
# ============================================================

print("\n--- Starting Standard SHAP map vs Spatial SHAP map comparison ---")

# Feature to map
MAP_FEATURE = "grade"

map_feature_index = X_large.columns.get_loc(
    MAP_FEATURE
)

standard_map_values = standard_shap_large[
    :,
    map_feature_index
]

spatial_map_values = spatial_shap_large[
    :,
    map_feature_index
]

map_comparison_table = pd.DataFrame({
    "Original index": X_large.index,
    "Latitude": coord_large["lat"].to_numpy(),
    "Longitude": coord_large["long"].to_numpy(),
    "Observed price": y_large.to_numpy(),
    "Standard SHAP": standard_map_values,
    "Spatial SHAP": spatial_map_values,
    "Spatial minus Standard": spatial_map_values - standard_map_values
})

map_comparison_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        f"KingCounty_standard_vs_spatial_map_{MAP_FEATURE}_300.csv"
    ),
    index=False
)

print(
    "\nMap comparison table saved as: "
    f"KingCounty_standard_vs_spatial_map_{MAP_FEATURE}_300.csv"
)


# ------------------------------------------------------------
# Use common colour scale for fair comparison
# ------------------------------------------------------------

combined_min = min(
    standard_map_values.min(),
    spatial_map_values.min()
)

combined_max = max(
    standard_map_values.max(),
    spatial_map_values.max()
)


# ------------------------------------------------------------
# Standard SHAP map
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    coord_large["long"],
    coord_large["lat"],
    c=standard_map_values,
    s=45,
    alpha=0.85,
    vmin=combined_min,
    vmax=combined_max
)

plt.colorbar(
    scatter,
    label=f"Standard SHAP value for {MAP_FEATURE}"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Standard SHAP Map for {MAP_FEATURE} - 300 Test Houses")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        f"Figure17_standard_SHAP_map_{MAP_FEATURE}_300.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nFigure saved as: "
    f"Figure17_standard_SHAP_map_{MAP_FEATURE}_300.png"
)


# ------------------------------------------------------------
# Spatial SHAP map
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    coord_large["long"],
    coord_large["lat"],
    c=spatial_map_values,
    s=45,
    alpha=0.85,
    vmin=combined_min,
    vmax=combined_max
)

plt.colorbar(
    scatter,
    label=f"KNN Spatial SHAP value for {MAP_FEATURE}"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"KNN Spatial SHAP Map for {MAP_FEATURE} - 300 Test Houses")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        f"Figure18_spatial_SHAP_map_{MAP_FEATURE}_300.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nFigure saved as: "
    f"Figure18_spatial_SHAP_map_{MAP_FEATURE}_300.png"
)


# ------------------------------------------------------------
# Difference map: Spatial SHAP minus Standard SHAP
# ------------------------------------------------------------

difference_map_values = (
    spatial_map_values
    - standard_map_values
)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    coord_large["long"],
    coord_large["lat"],
    c=difference_map_values,
    s=45,
    alpha=0.85
)

plt.colorbar(
    scatter,
    label=f"Spatial minus Standard SHAP for {MAP_FEATURE}"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Difference Map for {MAP_FEATURE}: Spatial SHAP minus Standard SHAP")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        f"Figure19_difference_map_{MAP_FEATURE}_300.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nFigure saved as: "
    f"Figure19_difference_map_{MAP_FEATURE}_300.png"
)


# ------------------------------------------------------------
# Summary statistics for map comparison
# ------------------------------------------------------------

map_summary = pd.DataFrame({
    "Statistic": [
        "Number of houses",
        "Mean absolute difference",
        "Median absolute difference",
        "Maximum absolute difference",
        "Sign-change rate (%)"
    ],
    "Value": [
        len(X_large),
        np.mean(np.abs(difference_map_values)),
        np.median(np.abs(difference_map_values)),
        np.max(np.abs(difference_map_values)),
        np.mean(
            np.sign(standard_map_values)
            != np.sign(spatial_map_values)
        ) * 100
    ]
})

print("\nMap comparison summary:\n")
print(
    map_summary.round(4).to_string(index=False)
)

map_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        f"KingCounty_map_comparison_summary_{MAP_FEATURE}_300.csv"
    ),
    index=False
)

print(
    "\nMap comparison summary saved as: "
    f"KingCounty_map_comparison_summary_{MAP_FEATURE}_300.csv"
)

print("\nStandard SHAP map vs Spatial SHAP map comparison completed.")

# ============================================================
# Final K selection experiment
# Combining CV error and SHAP stability
# ============================================================

print("\n--- Starting final K selection experiment ---")

# ------------------------------------------------------------
# 1. Candidate K values
# ------------------------------------------------------------

K_CANDIDATES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]

MIN_K_FOR_SHAP = 20

print("\nCandidate K values:")
print(K_CANDIDATES)


# ------------------------------------------------------------
# 2. CV criterion on training data
# ------------------------------------------------------------

train_predictions = model.predict(X_train)

train_distance_matrix = haversine_distance_matrix(
    coord_train["lat"].to_numpy(),
    coord_train["long"].to_numpy(),
    coord_train["lat"].to_numpy(),
    coord_train["long"].to_numpy()
)

# Exclude each observation itself
np.fill_diagonal(train_distance_matrix, np.inf)

cv_rows = []

for K in K_CANDIDATES:

    print(f"\nCalculating CV criterion for K = {K}")

    local_prediction_estimates = np.zeros(
        shape=len(X_train),
        dtype=float
    )

    for i in tqdm(
        range(len(X_train)),
        desc=f"CV K={K}"
    ):

        neighbour_positions = np.argsort(
            train_distance_matrix[i]
        )[:min(K, len(X_train) - 1)]

        local_prediction_estimates[i] = np.mean(
            train_predictions[neighbour_positions]
        )

    errors = (
        train_predictions
        - local_prediction_estimates
    )

    cv_mse = np.mean(errors ** 2)
    cv_rmse = np.sqrt(cv_mse)
    cv_mae = np.mean(np.abs(errors))

    cv_rows.append({
        "K": K,
        "CV MSE": cv_mse,
        "CV RMSE": cv_rmse,
        "CV MAE": cv_mae
    })


cv_selection_table = pd.DataFrame(cv_rows)

min_cv_rmse = cv_selection_table["CV RMSE"].min()

cv_selection_table["CV RMSE relative to minimum"] = (
    cv_selection_table["CV RMSE"] / min_cv_rmse
)

cv_selection_table.to_csv(
    os.path.join(OUTPUT_DIR, "KingCounty_final_K_selection_CV.csv"),
    index=False
)

print("\nCV results:")
print(
    cv_selection_table.round(4).to_string(index=False)
)


# ------------------------------------------------------------
# 3. Select top 5 features for stability check
# ------------------------------------------------------------

mean_abs_standard = np.abs(standard_shap_values).mean(axis=0)

top_feature_indices = np.argsort(mean_abs_standard)[::-1][:5]

top_features_for_k_selection = (
    X_selected.columns[top_feature_indices].tolist()
)

print("\nTop 5 features used for SHAP stability:")
print(top_features_for_k_selection)


# ------------------------------------------------------------
# 4. Compute Spatial SHAP for each candidate K
#    on the 50 selected houses
# ------------------------------------------------------------

k_selection_distance_matrix = haversine_distance_matrix(
    coord_selected["lat"].to_numpy(),
    coord_selected["long"].to_numpy(),
    coord_train["lat"].to_numpy(),
    coord_train["long"].to_numpy()
)

spatial_shap_by_candidate_k = {}

for K in K_CANDIDATES:

    print(f"\nComputing Spatial SHAP for K = {K}")

    shap_values_for_K = np.zeros(
        shape=(len(X_selected), X_selected.shape[1]),
        dtype=float
    )

    for i in tqdm(
        range(len(X_selected)),
        desc=f"SHAP K={K}"
    ):

        current_x = X_selected.iloc[[i]]

        neighbour_positions = np.argsort(
            k_selection_distance_matrix[i]
        )[:min(K, len(X_train))]

        neighbour_positions = np.atleast_1d(
            neighbour_positions
        )

        local_background = X_train.iloc[
            neighbour_positions
        ]

        local_explainer = shap.TreeExplainer(
            model,
            data=local_background,
            feature_perturbation="interventional"
        )

        local_values = local_explainer.shap_values(
            current_x,
            check_additivity=False
        )

        shap_values_for_K[i, :] = np.asarray(
            local_values
        )[0]

    spatial_shap_by_candidate_k[K] = shap_values_for_K.copy()

print("\nSpatial SHAP values for all candidate K values completed.")


# ------------------------------------------------------------
# 5. SHAP stability between neighbouring K values
# ------------------------------------------------------------

stability_rows = []

for index in range(len(K_CANDIDATES) - 1):

    K1 = K_CANDIDATES[index]
    K2 = K_CANDIDATES[index + 1]

    for feature_name in top_features_for_k_selection:

        feature_index = X_selected.columns.get_loc(
            feature_name
        )

        values_1 = spatial_shap_by_candidate_k[K1][:, feature_index]
        values_2 = spatial_shap_by_candidate_k[K2][:, feature_index]

        pearson_value = pearsonr(
            values_1,
            values_2
        ).statistic

        spearman_value = spearmanr(
            values_1,
            values_2
        ).statistic

        sign_agreement = np.mean(
            np.sign(values_1)
            == np.sign(values_2)
        ) * 100

        mean_abs_difference = np.mean(
            np.abs(values_1 - values_2)
        )

        stability_rows.append({
            "K1": K1,
            "K2": K2,
            "Comparison": f"K={K1} vs K={K2}",
            "Feature": feature_name,
            "Pearson correlation": pearson_value,
            "Spearman correlation": spearman_value,
            "Sign agreement (%)": sign_agreement,
            "Mean absolute difference": mean_abs_difference
        })


stability_table = pd.DataFrame(stability_rows)

stability_table.to_csv(
    os.path.join(OUTPUT_DIR, "KingCounty_final_K_selection_stability_pairwise.csv"),
    index=False
)

print("\nPairwise SHAP stability table:")
print(
    stability_table.round(4).to_string(index=False)
)


# ------------------------------------------------------------
# 6. Average stability for each adjacent K pair
# ------------------------------------------------------------

stability_pair_summary = (
    stability_table
    .groupby(["K1", "K2", "Comparison"])
    .agg({
        "Pearson correlation": "mean",
        "Spearman correlation": "mean",
        "Sign agreement (%)": "mean",
        "Mean absolute difference": "mean"
    })
    .reset_index()
)

stability_pair_summary.to_csv(
    os.path.join(OUTPUT_DIR, "KingCounty_final_K_selection_stability_summary.csv"),
    index=False
)

print("\nAverage stability by neighbouring K pair:")
print(
    stability_pair_summary.round(4).to_string(index=False)
)


# ------------------------------------------------------------
# 7. Convert pairwise stability into a K-level stability score
# ------------------------------------------------------------

k_level_rows = []

for K in K_CANDIDATES:

    related_rows = stability_pair_summary[
        (stability_pair_summary["K1"] == K)
        |
        (stability_pair_summary["K2"] == K)
    ]

    if len(related_rows) == 0:
        avg_pearson = np.nan
        avg_spearman = np.nan
        avg_sign = np.nan
        avg_mad = np.nan
    else:
        avg_pearson = related_rows["Pearson correlation"].mean()
        avg_spearman = related_rows["Spearman correlation"].mean()
        avg_sign = related_rows["Sign agreement (%)"].mean()
        avg_mad = related_rows["Mean absolute difference"].mean()

    k_level_rows.append({
        "K": K,
        "Average neighbouring Pearson": avg_pearson,
        "Average neighbouring Spearman": avg_spearman,
        "Average neighbouring sign agreement (%)": avg_sign,
        "Average neighbouring MAD": avg_mad
    })


k_level_stability = pd.DataFrame(k_level_rows)

final_k_selection_table = cv_selection_table.merge(
    k_level_stability,
    on="K",
    how="left"
)

# ------------------------------------------------------------
# 8. Final selection rule
# ------------------------------------------------------------
# Rule:
# 1. K should not be smaller than MIN_K_FOR_SHAP
# 2. CV RMSE should be within 5% of the minimum
# 3. Average Pearson stability should be at least 0.95
# 4. Average sign agreement should be at least 90%

final_k_selection_table["Within 5 percent of min CV"] = (
    final_k_selection_table["CV RMSE relative to minimum"]
    <= 1.05
)

final_k_selection_table["Pass stability rule"] = (
    (final_k_selection_table["Average neighbouring Pearson"] >= 0.95)
    &
    (final_k_selection_table["Average neighbouring sign agreement (%)"] >= 90)
)

final_k_selection_table["Pass final rule"] = (
    (final_k_selection_table["K"] >= MIN_K_FOR_SHAP)
    &
    final_k_selection_table["Within 5 percent of min CV"]
    &
    final_k_selection_table["Pass stability rule"]
)

eligible_k_table = final_k_selection_table[
    final_k_selection_table["Pass final rule"]
].copy()

if len(eligible_k_table) > 0:

    # Choose the smallest eligible K to keep the background local
    FINAL_SELECTED_K = int(
        eligible_k_table.sort_values("K").iloc[0]["K"]
    )

else:

    # Fallback: choose K with the lowest combined rank
    final_k_selection_table["CV rank"] = (
        final_k_selection_table["CV RMSE"].rank(method="min")
    )

    final_k_selection_table["Stability rank"] = (
        (-final_k_selection_table["Average neighbouring Pearson"])
        .rank(method="min")
    )

    final_k_selection_table["Combined rank"] = (
        final_k_selection_table["CV rank"]
        +
        final_k_selection_table["Stability rank"]
    )

    fallback_table = final_k_selection_table[
        final_k_selection_table["K"] >= MIN_K_FOR_SHAP
    ].copy()

    FINAL_SELECTED_K = int(
        fallback_table.sort_values("Combined rank").iloc[0]["K"]
    )


final_k_selection_table.to_csv(
    os.path.join(OUTPUT_DIR, "KingCounty_final_K_selection_combined.csv"),
    index=False
)

print("\nFinal K selection table:")
print(
    final_k_selection_table.round(4).to_string(index=False)
)

print("\nFINAL SELECTED K:", FINAL_SELECTED_K)


# ------------------------------------------------------------
# 9. Plot CV curve with final selected K
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    final_k_selection_table["K"],
    final_k_selection_table["CV RMSE"],
    marker="o"
)

plt.axvline(
    FINAL_SELECTED_K,
    linestyle="--",
    label=f"Selected K = {FINAL_SELECTED_K}"
)

plt.xlabel("Neighbourhood size K")
plt.ylabel("CV RMSE")
plt.title("K Selection: CV Criterion")

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure20_final_K_selection_CV.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 10. Plot stability curve
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    final_k_selection_table["K"],
    final_k_selection_table["Average neighbouring Pearson"],
    marker="o",
    label="Average neighbouring Pearson"
)

plt.axhline(
    0.95,
    linestyle="--",
    label="Pearson threshold = 0.95"
)

plt.axvline(
    FINAL_SELECTED_K,
    linestyle="--",
    label=f"Selected K = {FINAL_SELECTED_K}"
)

plt.xlabel("Neighbourhood size K")
plt.ylabel("SHAP stability")
plt.title("K Selection: SHAP Stability")

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure21_final_K_selection_stability.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFinal K selection experiment completed.")