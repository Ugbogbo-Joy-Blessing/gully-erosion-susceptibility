# Gully Erosion Susceptibility Modelling in Benin Metropolis Using Random Forest

A geospatial machine-learning project integrating topographic, environmental, soil, climatic and anthropogenic variables to model gully erosion susceptibility across Benin Metropolis, Edo State, Nigeria.

The workflow combines **Google Earth Engine, GIS, remote sensing, field-supported gully inventory data and Python-based Random Forest modelling** to produce a continuous spatial susceptibility surface and identify the environmental variables most strongly associated with observed gully occurrence.

---

## Project Overview

Gully erosion is a major environmental and infrastructural challenge across Benin Metropolis. Intense rainfall, highly erodible soils, changing vegetation and land cover, terrain characteristics, concentrated runoff and urban development interact to create conditions favourable to gully formation.

This project applies a spatial data science workflow to investigate those relationships and answer two main questions:

1. Where are areas with relatively high gully erosion susceptibility across Benin Metropolis?
2. Which environmental and topographic variables are most important in distinguishing observed gully locations from stable-terrain locations?

A **Random Forest classifier** was trained using a balanced inventory of gully-presence and stable-terrain observations together with raster-based conditioning factors.

The resulting model was then applied spatially to generate a continuous gully susceptibility probability surface.

---

## Study Area

The analysis covers **Benin Metropolis, Edo State, Nigeria**, with emphasis on:

- Oredo LGA
- Egor LGA
- Ikpoba-Okha LGA

The study area lies within the humid tropical environment of southern Nigeria and is strongly influenced by the poorly consolidated sandy materials associated with the Benin Formation.

![Study Area](outputs/maps/study_area.png)

---

## Project Objectives

The project was designed to:

- acquire and preprocess multi-source geospatial datasets;
- derive relevant topographic, hydrological and environmental conditioning factors;
- compile a spatial inventory of known gully locations and contrasting stable-terrain locations;
- extract raster predictor values at training locations;
- train a Random Forest model for binary gully-occurrence classification;
- evaluate model performance using stratified cross-validation and standard classification metrics;
- examine model-derived variable importance;
- generate a continuous gully susceptibility probability raster; and
- present the continuous model output as interpretable susceptibility classes for GIS-based decision support.

---

## Data Sources

The project integrates several geospatial datasets from different sources.

| Dataset | Source | Approximate Resolution | Application |
|---|---|---:|---|
| Elevation | SRTM | 30 m | Terrain representation |
| Slope | SRTM-derived | 30 m | Terrain steepness |
| Aspect | SRTM-derived | 30 m | Slope orientation |
| TWI | MERIT Hydro / terrain-derived | 30 m | Topographic wetness |
| NDVI | Sentinel-2 | 10 m | Vegetation condition |
| LULC | ESA WorldCover | 10 m | Land-cover characteristics |
| Rainfall | CHIRPS | ~5 km native | Climatic influence |
| Sand fraction | OpenLandMap | 250 m | Soil texture |
| Clay fraction | OpenLandMap | 250 m | Soil texture |
| Additional terrain/proximity predictors | GIS-derived | Analysis dependent | Final model inputs |

The Google Earth Engine script included in this repository exports **nine core raster datasets**. Additional predictors represented in the final model were prepared during the subsequent GIS modelling workflow.

---

## Gully Inventory

The modelling dataset was built using two classes:

- **120 gully-presence locations**
- **120 contrasting stable-terrain locations**

Gully-presence locations were compiled using a combination of:

- historical scientific literature;
- high-resolution Google Earth imagery; and
- direct field GPS coordinate surveys.

Stable-terrain locations were used as the contrasting absence class for binary model training.

This produced a balanced labelled dataset of **240 observations**.

---

## Final Model Predictors

The written project report contains inconsistent references to the total number of conditioning factors.

For this portfolio, the **final reported variable-importance output is used as the reference for the fitted feature set and contains 12 predictors**.

The supplied Python script does not hard-code predictor names. Instead, it dynamically loads the `.tif` predictor rasters available in the modelling workspace.

The predictors represented in the final reported model output are:

1. Rainfall
2. NDVI
3. Clay
4. Elevation
5. Sand Fraction
6. LULC
7. TWI
8. Slope
9. SPI
10. Plan Curvature
11. Aspect
12. Distance to Road

This distinction keeps the repository aligned with the final model output while preserving the behaviour of the original executable script.

---

## Workflow

The project follows the general workflow:

```text
Multi-source geospatial datasets
            ↓
Google Earth Engine preprocessing
            ↓
Terrain / environmental conditioning factors
            ↓
GIS processing and preparation of additional predictors
            ↓
Gully-presence + stable-terrain inventory
            ↓
Raster values extracted at training locations
            ↓
Python / scikit-learn
            ↓
Random Forest binary classifier
            ↓
5-fold stratified cross-validation
            ↓
Accuracy + Precision + Recall + F1 + ROC-AUC
            ↓
Variable importance
            ↓
Final model fitted to labelled dataset
            ↓
Pixel-wise probability prediction
            ↓
Continuous susceptibility raster (0–1)
            ↓
GIS susceptibility classification
            ↓
Very Low → Low → Moderate → High → Very High
```

---

## Google Earth Engine Preprocessing

Google Earth Engine was used to prepare several of the core geospatial predictors.

The supplied GEE script:

- constructs the Benin Metropolis boundary from Oredo, Egor and Ikpoba-Okha;
- exports the study-area boundary;
- extracts SRTM elevation;
- derives slope and aspect;
- combines MERIT Hydro contributing-area information with terrain slope for TWI;
- creates a Sentinel-2 NDVI composite;
- extracts ESA WorldCover LULC;
- calculates a 2016–2025 mean annual rainfall surface from CHIRPS;
- extracts OpenLandMap sand fraction; and
- extracts OpenLandMap clay fraction.

The script exports nine raster layers for subsequent GIS and machine-learning processing.

See:

`gee/benin_metropolis_master_extraction.js`

---

## Python / Random Forest Implementation

The machine-learning stage was implemented in **Python using scikit-learn, GeoPandas and Rasterio**.

The supplied script:

1. mounts the Google Drive project workspace;
2. reads `Final_Training_Points.shp`;
3. discovers the `.tif` predictor rasters in the workspace;
4. extracts raster values at each labelled training point;
5. constructs the predictor matrix and binary target;
6. defines the Random Forest classifier;
7. performs 5-fold stratified cross-validation;
8. generates cross-validated class predictions;
9. generates cross-validated probability predictions;
10. calculates model diagnostics;
11. fits the Random Forest to the complete labelled dataset;
12. calculates feature importance;
13. aligns raster dimensions to a master raster where required;
14. applies the fitted model spatially; and
15. exports a continuous gully-susceptibility probability raster.

See:

`python/random_forest_susceptibility_model.py`

---

## Random Forest Configuration

The supplied final Python implementation uses:

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=3,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42
)
```

Model evaluation uses:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

together with:

- `cross_val_score`
- `cross_val_predict`
- confusion matrix
- classification report
- ROC curve
- AUC

The final model is subsequently fitted to the complete labelled dataset before spatial prediction.

---

## Implementation Note

The repository contains the **original Google Earth Engine preprocessing script and original Python/Colab Random Forest modelling script** used for the project.

The executable workflow is documented directly from these supplied scripts.

The final Python implementation uses fixed Random Forest parameters and **5-fold stratified cross-validation** rather than a 70/30 train-test implementation.

Cross-validated class predictions and probabilities are used for model diagnostics. The model is subsequently fitted to the full labelled dataset to generate the continuous spatial susceptibility surface.

Where narrative methodology text in the written project differs from the executable implementation, this repository reports the behaviour of the supplied code.

---

## Model Performance

The final reported model performance was:

| Metric | Result |
|---|---:|
| Overall Accuracy | **89.50%** |
| Kappa Coefficient | **0.79** |
| AUC-ROC | **0.92** |
| Stable Terrain Precision | 0.91 |
| Stable Terrain Recall | 0.88 |
| Stable Terrain F1-Score | 0.89 |
| Gully Precision | 0.89 |
| Gully Recall | 0.92 |
| Gully F1-Score | 0.90 |

The **AUC-ROC of 0.92** indicates strong discrimination between observed gully-presence and stable-terrain observations within the validation framework used in this project.

### ROC Curve

![ROC-AUC Curve](outputs/figures/roc_auc_curve.png)

---

## Variable Importance

Random Forest feature importance was used to examine the relative contribution of the predictor variables.

The final reported ranking was:

1. Rainfall
2. NDVI
3. Clay
4. Elevation
5. Sand Fraction
6. LULC
7. TWI
8. Slope
9. SPI
10. Plan Curvature
11. Aspect
12. Distance to Road

![Variable Importance](outputs/figures/variable_importance.png)

Rainfall and NDVI emerged as the strongest individual predictors in the final importance output, followed by soil and terrain-related variables.

These importance values describe the model's predictive structure and should not be interpreted as proof that an individual variable independently causes gully formation.

---

## From Model Probability to Susceptibility Classes

The Random Forest script predicts a **continuous probability of gully occurrence from 0 to 1** using `predict_proba`.

For final GIS interpretation and cartographic communication, this continuous probability surface was represented using five susceptibility classes:

```text
Random Forest probability (0–1)
              ↓
GIS susceptibility classification
              ↓
Very Low
Low
Moderate
High
Very High
```

The five susceptibility categories are therefore a **cartographic interpretation of the continuous model output**.

The Random Forest itself was trained as a **binary classifier** using:

```text
0 = Stable terrain
1 = Gully occurrence
```

It was **not trained as a five-class classifier**.

---

## Final Gully Susceptibility Map

The fitted Random Forest model was applied across the predictor raster stack to produce a continuous probability surface.

The final cartographic product represents this surface as:

- Very Low susceptibility
- Low susceptibility
- Moderate susceptibility
- High susceptibility
- Very High susceptibility

![Gully Erosion Susceptibility Map](outputs/maps/gully_susceptibility.png)

The susceptibility map is intended as a **spatial screening and decision-support product**, identifying locations whose environmental characteristics are similar to those associated with observed gully occurrence in the training inventory.

It should not be interpreted as a deterministic prediction of where or when a future gully will form.

---

## Selected Conditioning-Factor Maps

### Elevation

![Elevation](outputs/maps/elevation.png)

### Slope

![Slope](outputs/maps/slope.png)

### NDVI

![NDVI](outputs/maps/ndvi.png)

### Rainfall

![Rainfall](outputs/maps/rainfall.png)

### Topographic Wetness Index

![TWI](outputs/maps/twi.png)

### Stream Power Index

![SPI](outputs/maps/spi.png)

### Plan Curvature

![Plan Curvature](outputs/maps/plan_curvature.png)

### Sand Fraction

![Sand Fraction](outputs/maps/sand_fraction.png)

### Clay Fraction

![Clay Fraction](outputs/maps/clay_fraction.png)

### Aspect

![Aspect](outputs/maps/aspect.png)

---

## Key Findings

The project demonstrates that gully susceptibility across Benin Metropolis reflects the combined influence of climatic, vegetation, soil and terrain conditions.

The final Random Forest importance output identified **rainfall and NDVI as the strongest individual predictors**, followed by soil characteristics and terrain-related variables.

The susceptibility surface also demonstrates that gully susceptibility is spatially heterogeneous across the metropolis rather than uniformly distributed.

The resulting map can support preliminary identification of areas requiring closer field investigation, environmental monitoring or erosion-control planning.

---

## Tools and Technologies

- Google Earth Engine
- ArcGIS / GIS processing
- Remote Sensing
- Google Earth
- GPS field verification
- Python
- Google Colab
- GeoPandas
- Rasterio
- NumPy
- Pandas
- scikit-learn
- Random Forest
- Stratified Cross-Validation
- ROC-AUC Analysis
- Raster Susceptibility Modelling

---

## Repository Structure

```text
gully-erosion-susceptibility-benin/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── gee/
│   ├── benin_metropolis_master_extraction.js
│   └── README.md
│
├── python/
│   ├── random_forest_susceptibility_model.py
│   └── README.md
│
├── workflow/
│   └── README.md
│
└── outputs/
    │
    ├── maps/
    │   ├── study_area.png
    │   ├── aspect.png
    │   ├── clay_fraction.png
    │   ├── elevation.png
    │   ├── sand_fraction.png
    │   ├── slope.png
    │   ├── ndvi.png
    │   ├── plan_curvature.png
    │   ├── rainfall.png
    │   ├── spi.png
    │   ├── twi.png
    │   ├── gully_susceptibility.png
    │   └── README.md
    │
    ├── figures/
    │   ├── roc_auc_curve.png
    │   ├── variable_importance.png
    │   └── README.md
    │
    └── tables/
        ├── final_model_predictors.csv
        ├── model_performance.csv
        ├── gully_inventory_summary.csv
        └── README.md
```

---

## Reproducibility

To reproduce the modelling workflow:

1. Prepare the study-area predictor rasters.
2. Ensure all predictors are available as `.tif` files in the modelling workspace.
3. Prepare `Final_Training_Points.shp` with a binary `Class` field.
4. Place the training shapefile and predictor rasters in:

```text
/content/drive/MyDrive/Benin_Gully_Project/
```

5. Run:

```text
python/random_forest_susceptibility_model.py
```

The script will:

- sample predictor values at the training locations;
- perform 5-fold stratified cross-validation;
- calculate classification diagnostics and ROC-AUC;
- fit the final Random Forest model;
- calculate variable importance; and
- generate `Benin_Gully_Susceptibility.tif`.

The exact model feature set depends on the `.tif` predictor rasters present in the workspace.

---

## Limitations

Several limitations should be considered when interpreting the results:

- predictor datasets have different native spatial resolutions;
- raster resampling and alignment introduce scale-related uncertainty;
- susceptibility represents statistical association rather than deterministic future gully occurrence;
- the model is dependent on the spatial distribution and quality of the gully inventory;
- randomly generated stable-terrain locations may not represent every possible non-gully landscape condition;
- standard random stratified cross-validation does not fully measure geographic transferability to spatially independent areas;
- medium-resolution elevation data may not capture small-scale terrain features relevant to gully initiation;
- the analysis does not include detailed subsurface geotechnical testing;
- the project does not model three-dimensional slope stability or physically simulate drainage hydraulics.

Future work could incorporate spatial-block validation, higher-resolution UAV or LiDAR-derived elevation data, expanded field inventories and comparison with additional machine-learning algorithms.

---

## Practical Application

The resulting susceptibility map can support:

- preliminary geohazard screening;
- environmental monitoring;
- urban development planning;
- prioritisation of field inspections;
- erosion-control planning;
- drainage and infrastructure assessment; and
- identification of locations where more detailed engineering investigation may be warranted.

The map is best used as a **decision-support layer**, alongside field observations and engineering or geotechnical assessment where site-specific decisions are required.

---

## Author

**Joy Ugbogbo**

GIS & Remote Sensing Analyst | Spatial Data Science

Areas of interest:

- GIS & Spatial Analysis
- Remote Sensing
- Earth Observation
- Environmental Modelling
- Machine Learning for Geospatial Applications
- Python for Spatial Data Analysis
