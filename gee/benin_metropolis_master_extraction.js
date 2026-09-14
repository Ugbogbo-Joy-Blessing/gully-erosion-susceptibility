// ==============================================================================
// BENIN METROPOLIS MASTER EXTRACTION & BOUNDARY EXPORT
// ==============================================================================

// 1. FILTER AND MERGE THE 3 LGAs DIRECTLY
var gaul = ee.FeatureCollection("FAO/GAUL/2015/level2");
var edoLGAs = gaul.filter(ee.Filter.eq('ADM1_NAME', 'Edo'));

// Match Oredo, Egor, and Ikpoba-Okha
var metropolis = edoLGAs.filter(ee.Filter.or(
  ee.Filter.eq('ADM2_NAME', 'Oredo'),
  ee.Filter.eq('ADM2_NAME', 'Egor'),
  ee.Filter.stringContains('ADM2_NAME', 'Ikpoba')
));

// Merge into a single boundary
var beninBoundary = metropolis.union(100);
var roi = beninBoundary.geometry();

// Visualize boundary on the map
Map.centerObject(roi, 11);
Map.addLayer(beninBoundary, {color: 'red'}, 'Benin Metropolis Boundary');

// ------------------------------------------------------------------------------
// 2. EXPORT BOUNDARY AS A SHAPEFILE (.SHP) TO GOOGLE DRIVE
// ------------------------------------------------------------------------------
Export.table.toDrive({
  collection: beninBoundary,
  description: 'Benin_Metropolis_Boundary_Shapefile',
  fileFormat: 'SHP'
});

// ------------------------------------------------------------------------------
// 3. GENERATE ALL PARAMETERS CLIPPED TO THE EXACT BOUNDARY
// ------------------------------------------------------------------------------

// A. Topography (SRTM 30m)
var dem = ee.Image('USGS/SRTMGL1_003').clip(roi);
var elevation = dem.rename('Elevation');
var slope = ee.Terrain.slope(dem).rename('Slope');
var aspect = ee.Terrain.aspect(dem).rename('Aspect');

// B. Hydrology (Topographic Wetness Index - TWI)
var upa = ee.Image("MERIT/Hydro/v1_0_1").select('upa').clip(roi);

var twi = ee.Image().expression(
  'log((upa * 1000000) / tan(slope * 3.14159 / 180))', {
    upa: upa,
    slope: slope.where(slope.eq(0), 0.001)
}).rename('TWI');

// C. Vegetation (Sentinel-2 10m Cloud-Masked NDVI)
function maskS2clouds(image) {
  var qa = image.select('QA60');

  var mask = qa
    .bitwiseAnd(1 << 10).eq(0)
    .and(qa.bitwiseAnd(1 << 11).eq(0));

  return image.updateMask(mask);
}

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(roi)
  .filterDate('2025-01-01', '2025-12-31')
  .map(maskS2clouds)
  .median()
  .clip(roi);

var ndvi = s2
  .normalizedDifference(['B8', 'B4'])
  .rename('NDVI');

// D. Land Cover (ESA WorldCover 10m)
var lulc = ee.ImageCollection("ESA/WorldCover/v200")
  .first()
  .clip(roi)
  .rename('LULC');

// E. Climate (CHIRPS 10-Year Average Rainfall)
var rainfall = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
  .filterDate('2016-01-01', '2025-12-31')
  .sum()
  .divide(10)
  .clip(roi)
  .rename('Rainfall');

// F. Soil Texture (OpenLandMap 250m Sand & Clay)
var sand = ee.Image(
  "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02"
)
  .select('b0')
  .clip(roi)
  .rename('Sand');

var clay = ee.Image(
  "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02"
)
  .select('b0')
  .clip(roi)
  .rename('Clay');

// ------------------------------------------------------------------------------
// 4. BATCH EXPORT ALL RASTERS TO GOOGLE DRIVE
// ------------------------------------------------------------------------------
var rasters = [
  {img: elevation, name: '1_Elevation_Benin', res: 30},
  {img: slope,     name: '2_Slope_Benin',     res: 30},
  {img: aspect,    name: '3_Aspect_Benin',    res: 30},
  {img: twi,       name: '4_TWI_Benin',       res: 30},
  {img: ndvi,      name: '5_NDVI_Benin',      res: 10},
  {img: lulc,      name: '6_LULC_Benin',      res: 10},
  {img: rainfall,  name: '7_Rainfall_Benin',  res: 30},
  {img: sand,      name: '8_Sand_Benin',      res: 250},
  {img: clay,      name: '9_Clay_Benin',      res: 250}
];

rasters.forEach(function(item) {

  Export.image.toDrive({
    image: item.img,
    description: item.name,
    scale: item.res,
    region: roi,
    maxPixels: 1e13
  });

});
