import os
import glob
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.warp import reproject, Resampling

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_val_predict
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from google.colab import drive

warnings.filterwarnings('ignore')


# ==============================================================================
# 1. CONNECT TO GOOGLE DRIVE
# ==============================================================================

drive.mount('/content/drive')

workspace = '/content/drive/MyDrive/Benin_Gully_Project/'


# ==============================================================================
# 2. LOAD TRAINING POINTS AND PREDICTOR RASTERS
# ==============================================================================

pts = gpd.read_file(
    workspace + 'Final_Training_Points.shp'
)

raster_files = sorted(
    glob.glob(workspace + '*.tif')
)

raster_names = [
    os.path.basename(f)
    .replace('.tif', '')
    .replace('ExtractByMask_OutRaster_', '')
    for f in raster_files
]


# ==============================================================================
# 3. EXTRACT RASTER VALUES AT TRAINING LOCATIONS
# ==============================================================================

print("Extracting raster values for model training...")

extracted_data = []

for index, row in pts.iterrows():

    coord = [
        (
            row.geometry.x,
            row.geometry.y
        )
    ]

    point_values = {}

    for r_file, r_name in zip(
        raster_files,
        raster_names
    ):

        with rasterio.open(r_file) as src:

            val = next(
                src.sample(coord)
            )[0]

            point_values[r_name] = val

    extracted_data.append(point_values)


df = pd.DataFrame(
    extracted_data
)

df['Class'] = pts['Class'].values


# ==============================================================================
# 4. PREPARE MODEL INPUTS
# ==============================================================================

X = df.drop(
    'Class',
    axis=1
)

y = df['Class']


# ==============================================================================
# 5. RANDOM FOREST MODEL
# ==============================================================================

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=3,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42
)


# ==============================================================================
# 6. FIVE-FOLD STRATIFIED CROSS-VALIDATION
# ==============================================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=cv,
    scoring='accuracy'
)

mean_acc = (
    cv_scores.mean() * 100
)


# ==============================================================================
# 7. CROSS-VALIDATED PREDICTIONS
# ==============================================================================

predictions = cross_val_predict(
    rf_model,
    X,
    y,
    cv=cv
)

pred_prob = cross_val_predict(
    rf_model,
    X,
    y,
    cv=cv,
    method='predict_proba'
)[:, 1]


# ==============================================================================
# 8. MODEL DIAGNOSTICS
# ==============================================================================

conf_matrix = confusion_matrix(
    y,
    predictions
)

class_report = classification_report(
    y,
    predictions
)

fpr, tpr, thresholds = roc_curve(
    y,
    pred_prob
)

roc_auc = auc(
    fpr,
    tpr
)


# ==============================================================================
# 9. FIT FINAL MODEL USING FULL LABELLED DATASET
# ==============================================================================

rf_model.fit(
    X,
    y
)


# ==============================================================================
# 10. VARIABLE IMPORTANCE
# ==============================================================================

importances = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(
    ascending=False
)


# ==============================================================================
# 11. EXPORT MODEL DIAGNOSTICS
# ==============================================================================

print(
    "Exporting Chapter 4 Diagnostics "
    "Text Report to Drive..."
)

with open(
    workspace +
    'Chapter_4_Advanced_Diagnostics.txt',
    'w'
) as f:

    f.write(
        "--- RANDOM FOREST "
        "CROSS-VALIDATION REPORT ---\n"
    )

    f.write(
        f"Mean Cross-Validation Accuracy: "
        f"{mean_acc:.2f}%\n"
    )

    f.write(
        f"Model AUC-ROC Score: "
        f"{roc_auc:.4f}\n\n"
    )

    f.write(
        f"Confusion Matrix:\n"
        f"{conf_matrix}\n\n"
    )

    f.write(
        "Detailed Accuracy Table "
        "(Precision, Recall, F1-Score):\n"
    )

    f.write(
        class_report +
        "\n\n"
    )

    f.write(
        "Variable Importance:\n"
    )

    f.write(
        importances.to_string()
    )


# ==============================================================================
# 12. GENERATE CONTINUOUS SUSCEPTIBILITY PROBABILITY MAP
# ==============================================================================

print(
    "Generating continuous probability "
    "susceptibility map across the metropolis..."
)

master_file = raster_files[0]

with rasterio.open(
    master_file
) as ref:

    meta = ref.meta.copy()

    height, width = ref.shape

    nodata_val = (
        ref.nodata
        if ref.nodata is not None
        else -9999
    )

    master_transform = ref.transform

    master_crs = ref.crs

    meta.update(
        dtype=rasterio.float32,
        nodata=-9999.0
    )

    first_band = ref.read(1)

    valid_pixels_2d = (
        (first_band != nodata_val) &
        (~np.isnan(first_band))
    )

    valid_pixels = (
        valid_pixels_2d.flatten()
    )


# ==============================================================================
# 13. ALIGN ALL PREDICTOR RASTERS TO MASTER GRID
# ==============================================================================

raster_data = []

for r_file in raster_files:

    with rasterio.open(
        r_file
    ) as src:

        if src.shape != (
            height,
            width
        ):

            aligned_arr = np.empty(
                (height, width),
                dtype=src.meta['dtype']
            )

            reproject(

                source=rasterio.band(
                    src,
                    1
                ),

                destination=aligned_arr,

                src_transform=src.transform,

                src_crs=src.crs,

                dst_transform=master_transform,

                dst_crs=master_crs,

                resampling=Resampling.nearest
            )

            raster_data.append(
                aligned_arr.flatten()
            )

        else:

            raster_data.append(
                src.read(1).flatten()
            )


# ==============================================================================
# 14. PREPARE PIXELS FOR SPATIAL PREDICTION
# ==============================================================================

X_map = np.column_stack(
    raster_data
)

is_finite = np.all(
    np.isfinite(X_map),
    axis=1
)

is_within_bounds = np.all(
    (
        (X_map > -1e10) &
        (X_map < 1e10)
    ),
    axis=1
)

strict_valid_pixels = (
    valid_pixels &
    is_finite &
    is_within_bounds
)

X_valid = pd.DataFrame(
    X_map[strict_valid_pixels],
    columns=X.columns
)


# ==============================================================================
# 15. PREDICT CONTINUOUS GULLY SUSCEPTIBILITY PROBABILITY
# ==============================================================================

map_predictions = (
    rf_model.predict_proba(
        X_valid
    )[:, 1]
)


# ==============================================================================
# 16. BUILD FINAL RASTER
# ==============================================================================

final_map = np.full(
    first_band.size,
    -9999.0,
    dtype=np.float32
)

final_map[
    strict_valid_pixels
] = map_predictions

final_map_2d = (
    final_map.reshape(
        (height, width)
    )
)


# ==============================================================================
# 17. EXPORT FINAL SUSCEPTIBILITY RASTER
# ==============================================================================

out_tif = (
    workspace +
    'Benin_Gully_Susceptibility.tif'
)

with rasterio.open(
    out_tif,
    'w',
    **meta
) as dst:

    dst.write(
        final_map_2d,
        1
    )


print(
    "\nSUCCESS! Probability Map and "
    "Diagnostic Text Report have been "
    "saved to Google Drive."
)
