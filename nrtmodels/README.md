# NRT Models

This directory contains various model plug-ins that can be run by the **nrt-predict** package or directly called by importing the `nrtmodels` package.

## Change detection

The module `change` contains various models for change detection.

### Sentinel-2 cloud and shadow detection

This is a machine learning model for cloud and shadow detection. It requires a reference image, such as a Sentinel-2 Geomedian, and stacks two models (a temporal model and a spectral model) to obtain a classification. 

This model requires LightGBM installed.

``` yaml
models:
  - name: CloudAndShadowDetect
    nodata: 0
    output: clouds.tif
    inputs:
      - filename: s2be.vrt
        scale: 0.0001
        bands: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
```

### Unsupervised vegetation change model 1

This is an unsupervised vegetation change detection model. It requires a reference image (e.g., Geomedian or Previous clear) to determine if change to vegetation has occured.

A 'previous clear' image can be generated in the process using the `PreviousClear` model (see below).

The model breaks the change into positive (class 1) or negative change (class 2).

``` yaml
models:
  - name: PreviousClear
    output: prev.tif

  - name: UnsupervisedVegetation1
    output: veg1.tif
    nodata: 0
    inputs:
      - filename: prev.tif
        scale: 0.0001
        bands: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
```

### Supervised vegetation change model 1

This is a supervised vegetation change detection model. It requires a reference image (e.g., Geomedian or Previous clear) to determine if change to vegetation has occured.

The model breaks the change into positive (class 1) or negative change (class 2).

``` yaml
models:
  - name: PreviousClear
    output: prev.tif

  - name: SupervisedVegetationChange1
    output: veg2.tif
    nodata: 0
    inputs:
      - filename: prev.tif
        scale: 0.0001
        bands: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
```

### Vegetation change ternary mosaic

The model `VegetationChangeTernary` generates a three-band (RGB) ternary mosaic to visualise changes in vegetation by comparing results between the observation and a reference image. 

``` yaml
models:
  - name: PreviousClear
    output: prev.tif

  - name: VegetationChangeTernary
    output: change.tif
    inputs:
      - filename: prev.tif
        scale: 0.0001
        bands: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
```


### Excess water detection model 1

This is an unsupervised excess water detection model. It requires a reference image (e.g., Geomedian or Previous clear) to determine if change to water has occured.

The aim of the model is to *not* to classify all water in the observation but to only show where new water has occured. This is useful for flood detection.

If necessary, a 'previous clear' image can be generated in the process using the `PreviousClear` model (see below).

It is recommended to use the Sentinel-2 Barest Earth as the reference image.

``` yaml
models:
  - name: ExcessWater1
    output: water1.tif
    nodata: 0
    inputs:
      - filename: s2be.vrt
        scale: 0.0001
        bands: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
```

### Excess water detection model 2

This is an unsupervised excess water detection model. It requires a reference image (e.g., Geomedian or Previous clear) to determine if change to water has occured.

The aim of the model is to *not* to classify all water in the observation but to only show where new water has occured. This is useful for flood detection.

The algorithm uses a semi-supervised approach combined with some domain knowledge.

If necessary, a 'previous clear' image can be generated in the process using the `PreviousClear` model (see below).

``` yaml
models:
  - name: ExcessWater2
    output: water2.tif
    nodata: 0
    inputs:
      - filename: s2be.vrt
        scale: 0.0001
        bands: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
```

### Water change ternary

The model `WaterChangeTernary` generates a three-band (RGB) ternary mosaic to visualise changes in water presence by comparing results between the observation and a reference image. 

``` yaml
models:
  - name: WaterChangeTernary
    output: water_ternary.tif
    inputs:
      - filename: s2be.vrt
        scale: 0.0001
        bands: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
```

## Reference generation

The module `reference` contains the models for generation of reference datasets.

### Previous Clear Observation

This model looks through the NRT archive and finds a last  observation with minimal cloud.

``` yaml
models:
  - name: PreviousClear
    output: prev.tif
```

## Hotspots


## Burnt area

The module `burnscar` contains the models for burnt area detection.

### Geomedian NBR Difference

The model `GeomedianNBR` calculates (B08-B11)/(B08+B11) for the observation and the Geomedian reference image and then returns  the difference between the two. It is recommended to use the Sentinel-2 Barest Earth reference image.

### Geomedian BSI Difference

The model `GeomedianBSI` calculates ((B11 + B04) - (B08 - B02)) / ((B11 + B04) + (B08 - B02) for the observation and the Geomedian reference image and then returns  the difference between the two. It is recommended to use the Sentinel-2 Barest Earth reference image.

### Geomedian NDVI Difference

The model `GeomedianNDVI` calculates (B08 - B04)/(B08 + B04) for the observation and the Geomedian reference image and then returns  the difference between the two. It is recommended to use the Sentinel-2 Barest Earth reference image.

### Geomedian Ternary Diff

The model `GeomedianDiff` generates a three-band (RGB) ternary mosaic to visualise burnt areas by comparing results between the observation and the Geomedian reference image. It is recommended to use the Sentinel-2 Barest Earth reference image.

If visualised as an RGB with 2%-cutoff stretch, burnt areas should appear in white.

### Unsupervised model 1

``` yaml
models:
   - name: UnsupervisedBurnscarDetect1
     required_bands: [B08, B11, B04, B02]
     output: burn1.tif
     driver: GTiff
     inputs:
         - filename: s2be.vrt
           scale: 0.0001
           bands: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
```

### Unsupervised model 2

``` yaml
models:
   - name: UnsupervisedBurnscarDetect2
     required_bands: [B08, B11, B04, B02]
     output: burn1.tif
     driver: GTiff
     inputs:
         - filename: s2be.vrt
           scale: 0.0001
           bands: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
```

### Supervised model 1

``` yaml
models:
   - name: SupervisedBurnscarDetect1
     output: burn-supervised1.tif
     inputs:
         - filename: s2be.vrt
           scale: 0.0001
           bands: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
```