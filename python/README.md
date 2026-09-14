# Python / Random Forest modelling

`random_forest_susceptibility_model.py` is the original Colab-oriented modelling script supplied for the project.

## What the script actually does

1. Mounts Google Drive and reads `Final_Training_Points.shp`.
2. Automatically loads **all `.tif` rasters** present in the project workspace.
3. Extracts raster values at every labelled training point.
4. Uses the point `Class` field as the target variable.
5. Fits a `RandomForestClassifier` with:
   - `n_estimators=500`
   - `max_depth=3`
   - `min_samples_leaf=4`
   - `max_features='sqrt'`
   - `random_state=42`
6. Evaluates the model with **5-fold stratified cross-validation** using `StratifiedKFold`.
7. Produces cross-validated class predictions and probabilities with `cross_val_predict`.
8. Calculates the confusion matrix, classification report, feature importance and ROC-AUC.
9. Fits the model on the full labelled dataset for final spatial prediction.
10. Aligns rasters to the first raster's grid where dimensions differ.
11. Predicts **continuous gully-susceptibility probabilities (0–1)** across valid pixels.
12. Exports `Benin_Gully_Susceptibility.tif` and a text diagnostics report to Google Drive.

## Important implementation note

The script does **not** hard-code the final predictor names. It trains on every `.tif` in the workspace, so the exact final feature set is determined by the raster files that were present when the model was run.

The supplied script also uses fixed Random Forest parameters and 5-fold cross-validation. It does **not** contain a 70/30 train-test split or `GridSearchCV`. The repository therefore documents the code implementation exactly as supplied rather than carrying those earlier thesis-methodology descriptions into the code section.
