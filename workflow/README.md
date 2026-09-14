# Modelling workflow

The completed portfolio workflow now includes the original Google Earth Engine preprocessing script and the original Python / Colab Random Forest modelling script.

## Implemented workflow

1. Build the Benin Metropolis boundary from Oredo, Egor and Ikpoba-Okha LGAs.
2. Prepare/export terrain, vegetation, land-cover, rainfall and soil rasters in Google Earth Engine.
3. Prepare the labelled gully inventory (`Final_Training_Points.shp`).
4. Load every `.tif` predictor present in the Google Drive workspace.
5. Extract predictor values at each labelled point with Rasterio.
6. Train a Random Forest classifier in Python / scikit-learn.
7. Evaluate it using **5-fold stratified cross-validation**.
8. Generate cross-validated predictions and probabilities.
9. Calculate the confusion matrix, classification report, variable importance and ROC-AUC.
10. Fit the Random Forest on the full labelled dataset.
11. Align raster grids where necessary using Rasterio reprojection.
12. Predict a continuous susceptibility probability for valid pixels.
13. Export the final `Benin_Gully_Susceptibility.tif`.

## Random Forest configuration in the supplied code

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=3,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42
)
```

## Validation in the supplied code

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

The diagnostics are produced from cross-validated predictions/probabilities using `cross_val_predict`.

## Reproducibility note

Both original code components supplied for this portfolio package are included:

- `../gee/benin_metropolis_master_extraction.js`
- `../python/random_forest_susceptibility_model.py`

The raw geospatial rasters and shapefile training inventory are not bundled because they are working/source datasets and may be large. The Python script expects them inside the configured Google Drive workspace.
