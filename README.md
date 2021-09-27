# nrt-predict

Run prediction models on [Digital Earth Australia](https://www.ga.gov.au/dea) Near-Real-Time (NRT) satellite observations. The NRT data is an effort to acquire, atmospherically-correct, and package the data as quickly as possible from when a [Sentinel-2](http://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2) satellite flies over an area in Australia. 

**nrt-predict** provides a customisable workflow framework for running various machine learning, AI, and statistical models on the data to produce additional outputs. The aim is for the (user-defined) models to be both easy to write (minimal boilerplate) but also customisable (if needed). Ancillary datasets can be used by the models and **nrt-predict** retrieves and crops these datasets automatically for the model.

## Options

**nrt-predict* is expected to be run with a yaml file providing the configuration of the models and other various options. If the filename for the configuration is not provided the default of `nrtpredict.yaml` is used.

### Quiet

This option makes the output more quiet.

``` yaml
quiet: False
```

### Product

This option allows you to switch between different products. The default is the NBART product.

``` yaml
product: NBART
```

### Saving the observation

Save the observation. If this option is set to a filename then the observation will be saved to disk as a GeoTiff.

``` yaml
obstmp: /tmp/obs.tif
```

The observation can be saved in memory by using the vsimem capabilities of GDAL.
``` yaml
obstmp: /vsimem/obs.tif
```

### Tiling

If this is enabled, **nrt-predict** will tile the predictions to save memory into `tilewidth x tilewidth` sized tiles. Note that tiling reduces the prediction speed. An optimal tilewidth would need to be worked out by the user to get the optimal maximum peak memory as this is dependent on the model.

``` yaml
tilewidth: 1000
```

### Observation scaling

DEA has saved the surface reflectances using fixed point with 2 decimal places of accuracy as a integer, so 1 = 100.00% reflantance is saved as 10000. This option provides a user-defined scaling (value * obsscale), this example converts 10000 to 1.

``` yaml
obsscale: 0.0001
```

### Temporary directory

This allow to specify the temporary directory. Allow environment variables in name e.g., `$HOME` or `$PBS_JOBFS`

``` yaml
tmpdir: /tmp
```

### No cleanup

Do not clean up any files once the processing is complete. Useful for debugging.

``` yaml
nocleanup: True
```

### Clip shape

**nrt-predict** clips and warps all ancillary files to match the observation. Before it does this it prints out the target projection and also saves the area as a GeoJSON in the file `clipshpfn`. This allows you to take that information and manually clip very large files (e.g., global or continental datasets that may be located on other systems) and transfer a subset of the data to the local system for processing.

``` yaml
clipshpfn: clip.json
```

This allows you to transfer `clip.json` to another system and then manually perform the clip with something like the below where the appropriate `-t_srs` is pasted in (see nrt-predict output).

```
gdalwarp -overwrite -t_srs 'PROJCS["WGS 84 / UTM zone 55S",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",147],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",10000000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32755"]]'  -cutline clip.json -crop_to_cutline verylarge.vrt small.tif
```

### GDAL Configuration

All standard GDAL configuration parameters are definable inside the yaml config.

For example: this can be used to configure the /vsis3/ settings for S3 bucket writing, AWS credentials, etc.

Note: AWS credentials only need to be set if they're not already in the
environment, `~/.aws/config`, or you're running on an EC2 instance with an IAM role.

For example:
``` yaml
gdalconfig:
  GDAL_DISABLE_READDIR_ON_OPEN: YES
  CPL_VSIL_CURL_ALLOWED_EXTENSIONS: '.tif,.geojson'
  CPL_CURL_VERBOSE: NO
  CPL_DEBUG: YES
  AWS_HTTPS: YES
  AWS_NO_SIGN_REQUEST: YES
```

### Models

Models are provided as a list of models to apply to the observation sequentially. This means that multiple models can be applied to the same observation and the outputs of one model can become the input to another model.

 The models to run. Each model can be given various configuration options. The output can be an in-memory file by setting 'output: /vsimem/filename'. All other parameters are set on the model object as attributes.

 Models can be loaded from the `nrtmodels` directory in the current path (default), from a pickled file on disk (using joblib.dump) or from a pickled model in a public s3 bucket. The later cases, the name should be of the form:
```
   file://path_to_file
```
 or
```
   s3://bucket/key
```

Models and their descriptions are located in the `nrtmodels` folder.

Some basic models to provide some examples are as follows.

#### NoOp

This is the 'No Operation' model that simply takes the input observation and saves it to the file specified by `output`. A driver can be specified in GDAL format, the default is `COG`.

``` yaml
models:
  - name: NoOp
    output: result.tif
    driver: GTiff
```

#### FirstBand

This is a model that simply returns the first band of the observation.

``` yaml
models:
  - name: FirstBand
    output: B02.tif
    driver: GTiff
```

#### BandTransform

This model can apply various formula to a band. Band names can be used and all standard `numpy.*` functions are available.

``` yaml
models:
   - name: BandTransform
     output: clipped-B02.tif
     driver: COG
     expr: 'clip(B02/10000., 0.01, 0.3)'
```

## Data

### NRT data

The Digital Earth Australia NRT data can be found in Amazon s3 buckets of the form:
```
s3://dea-public-data/L2/sentinel-2-nrt/S2MSIARD/<date>/<package>
```
The data can also be accessed through HTTPS at:
```
https://data.dea.ga.gov.au/L2/sentinel-2-nrt/S2MSIARD/
```
An example of running **nrt-predict** is:
```
./nrtpredict.py s3://dea-public-data/L2/sentinel-2-nrt/S2MSIARD/2021-09-02/S2B_OPER_MSI_ARD_TL_VGS4_20210902T013037_A023451_T55HDB_N03.01
```

### Archive data

**nrt-predict** can now also run on the archive packages that are available here:
```
https://data.dea.ga.gov.au/?prefix=baseline/
```
For example:
```
./nrtpredict.py s3://dea-public-data/baseline/s2a_ard_granule/2021-03-25/S2A_OPER_MSI_ARD_TL_VGS4_20210325T014951_A030057_T56HKH_N02.09/
```

### Test data

A (minified!) version of a package layout can be found in the [data/test](https://github.com/daleroberts/nrt-predict/tree/main/data/test/) directory of this repo.