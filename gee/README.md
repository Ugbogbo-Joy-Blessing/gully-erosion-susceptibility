# Google Earth Engine preprocessing

`benin_metropolis_master_extraction.js` is the original project code supplied for the GEE preprocessing stage.

It:

- builds the Benin Metropolis boundary from Oredo, Egor and Ikpoba-Okha LGAs using FAO GAUL;
- exports the merged boundary as a shapefile;
- derives/exports Elevation, Slope, Aspect and TWI;
- computes Sentinel-2 NDVI;
- extracts ESA WorldCover LULC;
- calculates a 2016–2025 CHIRPS mean-annual-rainfall layer;
- extracts OpenLandMap sand and clay fractions; and
- batch-exports the raster layers to Google Drive.

The GEE script exports **nine raster layers**. The final Random Forest result contains additional predictors that were derived or prepared later in the overall workflow, so the repository keeps the GEE preprocessing stage distinct from the final model-input stage.
