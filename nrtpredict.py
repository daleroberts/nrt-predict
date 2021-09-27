#!/usr/bin/env python3
"""
Run models on DEA Near-Real-Time (NRT) and Archive data.

Browseable NRT data:

    https://data.dea.ga.gov.au/?prefix=L2/sentinel-2-nrt/

Or the archive:

    https://data.dea.ga.gov.au/?prefix=baseline/
"""
from posixpath import expanduser
import numpy as np
import argparse
import hashlib
import joblib
import psutil
import uuid
import yaml
import sys
import os
import re

from osgeo import gdal, ogr, osr
from datetime import datetime
from urllib.parse import urlparse
from copy import deepcopy
from pydoc import locate
from io import BytesIO

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

gdal.PushErrorHandler("CPLQuietErrorHandler")

np.set_printoptions(precision=4, linewidth=120, suppress=True, sign='+')

DEBUG = False

MODELDIR = "nrtmodels"

RST = "\033[0m"
RED = "\033[38;5;9m"
GREEN = "\033[38;5;10m"
BLUE = "\033[38;5;4m"


class URLParsingError(ValueError):
    """
    Raised if a URL cannot be parsed.
    """


class InputDataError(ValueError):
    """
    Raised if there is a problem with input data.
    """


class IncorrectChecksumError(IOError):
    """
    Raised if the model SHA256 checksum doesn't match what is expected.
    """


class ConnectionError(IOError):
    """
    Raised if the data cannot be found or accessed.
    """


class Source:
    def get_observations(self, url: str) -> tuple:
        raise NotImplementedError


class DEALandsat(Source):
    def get_observations(self, url: str) -> tuple:
        pass


class DEASentinel2(Source):
    def __init__(self):
        self.bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

    def openfile(self, fn: str):
        return gdal.Open(fn)

    def get_observations(self, url: str, product: str = "NBART", onlymask: bool = False, **args) -> tuple:
        """
        Get the NRT observation from the S3 or public (HTTP) bucket and load the
        data into memory in a numpy array of shape (ysize, xsize, nbands). This is
        assuming the DEA package format.
        """
        bands = args.pop("bands_required", self.bands)

        pkg = parse_pkg(url)

        if url.startswith("/vsis3"):
            stripped_url = url.replace("/vsis3/dea-public-data", "https://data.dea.ga.gov.au")
        elif url.startswith("/vsicurl"):
            stripped_url = url.replace("/vsicurl/", "")
        else:
            stripped_url = url

        fn = [fn for fn in gdal.ReadDir(f"{url}/QA") if fn.endswith("FMASK.TIF")][0]
        fn = f"{url}/QA/{fn}"

        fd = self.openfile(fn)
        mask = fd.ReadAsArray()

        pnodata = np.count_nonzero(mask == 0) / np.prod(mask.shape)
        pclear = np.count_nonzero(mask == 1) / np.prod(mask.shape)

        log(f"Package:   {pkg}")
        log(f"Thumbnail: {stripped_url}/{product}/{product}_THUMBNAIL.JPG")
        log(f"Location:  {stripped_url}/map.html")
        log(f"Pixels:    {mask.shape[0]} x {mask.shape[1]}")
        log(f"Clear %:   {pclear:.4f}")

        geo = fd.GetGeoTransform()
        prj = fd.GetProjection()

        ysize = mask.shape[0]
        xsize = mask.shape[1]

        if onlymask:
            return (geo, prj, mask[:, :, np.newaxis])

        log("# Loading data")

        hc = checksum_array(mask)
        log(f"MASK (sha256:{hc})")

        # Handle possible name changes done by DEA between NRT and Archive by creating a mapping
        # between bands and filenames.

        fns = [fn for fn in gdal.ReadDir(f"{url}/{product}") if fn.endswith(".TIF") or fn.endswith(".tif")]
        bfm = {fn.split(f"{product}_")[1].split(".")[0]: fn for fn in fns}

        # Load other bands

        data = np.empty((ysize, xsize, len(bands)), dtype=np.float32)
        for i, band in enumerate(bands):
            fn = f"{url}/{product}/{bfm[band]}"
            fd = self.openfile(fn)
            fd.ReadAsArray(buf_obj=data[:, :, i], buf_ysize=ysize, buf_xsize=xsize)
            hc = checksum_array(data[:, :, i])
            log(f" {band} (sha256:{hc})")

        nodata = fd.GetRasterBand(1).GetNoDataValue()
        data[data == nodata] = np.nan

        log(f"\nShape: {data.shape}")

        return (geo, prj, data, mask)


class TiledPrediction:

    """
    A basic tiling strategy for running predictions over smaller tiles
    and then reassembling the resulting prediction. This is useful for
    reducing memory usage.
    """

    def __init__(self, model, tilewidth: int = 1000):
        self.model = model
        self.tilewidth = tilewidth

        self._predict = self.model.predict
        self.model.predict = self.predict

    def predict(self, mask, *datas):
        log(f"Prediction tiling enabled in configuration (tilewidth: {self.tilewidth})")

        shapes = np.array([data.shape for data in datas])
        assert ((shapes - shapes[0]) == 0).all()

        h = self.tilewidth

        oshape = shapes[0]
        augshape = shapes[0, 0] + (h - (shapes[0, 0] % h))
        dtypes = [d.dtype for d in datas]

        log(f"Data types: {dtypes}")
        log(f"Tile shape: {(self.tilewidth, self.tilewidth)}")
        log(f"Data shape: {tuple(oshape)} -> Augmented shape: {(augshape, augshape)}")
        log(f"   # Tiles: {(augshape//self.tilewidth)**2}")

        yhat = None

        log("Starting tiled prediction")

        for i in range(0, augshape, h):
            for j in range(0, augshape, h):
                if yhat is None:
                    # detect the shape of predictions from the first tile computed
                    tmask = mask[i : i + h, j : j + h]
                    tdatas = [dd[i : i + h, j : j + h, :] for dd in datas]
                    result = self._predict(tmask, *tdatas)
                    yhat = np.zeros((augshape, augshape, result.shape[-1]), np.float32)
                    yhat[i : i + h, j : j + h, :] = result
                    continue

                # Do this edge case seperately to avoid copies for bulk of cases
                if (oshape[0] < i + h) or (oshape[1] < j + h):
                    tdatas = []
                    for dd in datas:
                        ddd = dd[i : min(dd.shape[0], i + h), j : min(dd.shape[1], j + h), :]
                        tmp = np.nan * np.zeros((h, h, ddd.shape[-1]), dtype=np.float32)
                        tmp[: ddd.shape[0], : ddd.shape[1], :] = ddd
                        tdatas.append(tmp)
                    shp = ddd.shape
                    tmask = np.zeros((h, h), dtype=tmask.dtype)
                    tmask[: shp[0], : shp[1]] = mask[i : min(mask.shape[0], i + h), j : min(mask.shape[1], j + h)]
                    yhat[i : i + h, j : j + h, :] = self._predict(tmask, *tdatas)
                    continue

                # Bulk of cases
                tmask = mask[i : i + h, j : j + h]
                tdatas = [dd[i : i + h, j : j + h, :] for dd in datas]
                yhat[i : i + h, j : j + h, :] = self._predict(tmask, *tdatas)

            log(f"... Completed: {(i+h)/augshape*100:3.2f}%")

        yhat = yhat[: oshape[0], : oshape[1], :]

        log(f"Output shape: {yhat.shape}")

        return yhat

    def predict_and_save(self, fn: str, *datas):
        self.model.predict_and_save(fn, *datas)


def get_s3_client():
    """
    Get a s3 client without the need for credentials.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def checksum_array(arr: np.ndarray) -> str:
    """
    Checksum a numpy array.
    """
    hasher = hashlib.sha256()
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def checksum_file(f, blocksize: int = 2 << 15) -> str:
    """
    Checksum a file-like object.
    """
    hasher = hashlib.sha256()
    for block in iter(lambda: f.read(blocksize), b""):
        hasher.update(block)
    f.seek(0)
    return hasher.hexdigest()


def check_checksum(f, checksum: str, blocksize: int = 2 << 15):
    """
    Check that the SHA256 checksum of a file-type object matches the 'checksum'. Raises
    an 'IncorrectChecksumError' if they don't match.

    Checksums can be generated on the command line with:

      % shasum -a 256 filename

    """
    actual = checksum_file(f)

    log(f"Expected SHA256 checksum: {checksum}")
    log(f"Actual SHA256 checksum: {actual}")

    if actual != checksum:
        raise IncorrectChecksumError()

    log("Checksum matches.")


# TODO: path@sha256:123123 format?


def get_model(name: str, **config):
    """
    Given 'name', load a `Model` from disk and update its settings.
    This handles the standard case where the models are stored in
    the current directory, the case where a direct path to a pickled
    model is given, and the case where the model is stored in a public
    s3 bucket.
    """
    # Handle case where name is a path a pickled model on disk

    if name[:7] == "file://":
        path = name[7:]
        path, checksum = path.split(":")
        log(f"Loading model from '{path}'")
        with open(path, "rb") as df:
            f = BytesIO(df.read())
            check_checksum(f, checksum)
            f.seek(0)
            model = joblib.load(f)
            model.update(**config)

    # Handle model stored in a s3 bucket

    elif name[:5] == "s3://":
        log(f"Loading model from '{name}'")
        bucket, key = name.split("/")[2], name.split("/")[3:]
        key = "/".join(key)
        with BytesIO() as f:
            s3 = get_s3_client()
            s3.download_fileobj(Bucket=bucket, Key=key, Fileobj=f)
            f.seek(0)
            # check_checksum(f, checksum)
            f.seek(0)
            model = joblib.load(f)
            model.update(**config)

    # Handle standard case where model is in the MODELDIR directory
    # in the current working directory

    else:
        modelname = f"{MODELDIR}.{name}"
        sys.path.insert(0, os.getcwd())
        impl = locate(modelname)
        sys.path = sys.path[1:]
        if impl is None:
            raise ImportError(modelname)
        model = impl(**config)

    return model


def normalise_url(url: str) -> str:
    """
    Simple check to see if it is a valid URL.
    """
    try:

        p = urlparse(url)

    except ValueError as e:
        raise URLParsingError(f"Cannot parse the URL {url}: {e}")

    if p.scheme == "s3":
        url = url.replace("s3://", "/vsis3/")
    elif p.scheme == "http":
        url = f"/vsicurl/{url}"
    elif p.scheme == "https":
        url = f"/vsicurl/{url}"
    elif p.scheme == "" and os.path.exists(url):
        pass
    elif p.scheme == "" and not os.path.exists(url):
        pass
    else:
        raise URLParsingError(f"Cannot normalise the URL {url}")

    if url.endswith("/"):
        url = url[:-1]

    return url


def listfmt(lst: list) -> str:
    """
    Format a list as a str with 4 decimal places of accuracy.
    """
    return "(" + ", ".join([f"{x:.4f}" for x in lst]) + ")"


def wktfmt(wkt: str) -> str:
    """
    Round numbers in WKT str to 4 decimal places of accuracy.
    """
    return re.sub(r"([+-]*\d*\.\d\d\d\d)(\d*)", r"\1", wkt)


def sizefmt(num: int, suffix="B") -> str:
    """
    Format sizes in a human readible style.
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def log(msg: str = "", noinfo: bool = False, color=GREEN):
    """
    Log a message.
    """
    pid = os.getpid()
    process = psutil.Process(pid)
    mem = psutil.virtual_memory()
    used = sizefmt(mem.total - mem.available)
    rss = sizefmt(process.memory_info().rss)
    if noinfo:
        mem = ""
    else:
        mem = f"[MEM {rss}]"
    if isinstance(msg, str) and msg.startswith("#"):
        msg = "\n" + color + str(msg) + RST + " " + mem + "\n"
    elif isinstance(msg, str) and msg.startswith("@"):
        msg = BLUE + msg[1:] + RST
    else:
        msg = str(msg)
    if DEBUG and not msg.startswith("#"):
        print(mem + "\t" + msg, file=sys.stderr)
    else:
        print(msg, file=sys.stderr)


def warning(msg: str):
    """
    Warning!
    """
    print("\n" + RED + str(msg) + RST, file=sys.stderr)


def parse_pkg(url: str) -> str:
    """
    Package name parsing.
    """
    return [x for x in url.split("/") if len(x) > 0][-1]


def parse_obsdate(url: str) -> str:
    """
    Parse observation date from url.
    """
    pkg = parse_pkg(url)
    return datetime.strptime(pkg.split("_")[6], "%Y%m%dT%H%M%S")


def get_bounds(url: str) -> str:
    """
    Return bounds of url as WKT string. Given in EPSG:4326.
    """

    fn = f"{url}/bounds.geojson"

    fd = ogr.Open(fn)

    if fd is None:
        error = gdal.GetLastErrorMsg()
        raise ConnectionError(error)

    layer = fd.GetLayer()
    ftr = layer.GetFeature(0)
    poly = ftr.GetGeometryRef()
    return poly.ExportToWkt()


def polygon_from_geobox(geo: tuple, xsize: int, ysize: int):
    """
    Generate a polygon from a geobox and the number of pixels.
    """
    poly = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(geo[0] + 0 * geo[1] + 0 * geo[2], geo[3] + 0 * geo[4] + 0 * geo[5])
    ring.AddPoint(geo[0] + xsize * geo[1] + 0 * geo[2], geo[3] + xsize * geo[4] + 0 * geo[5])
    ring.AddPoint(geo[0] + xsize * geo[1] + ysize * geo[2], geo[3] + 0 * geo[4] + ysize * geo[5])
    ring.AddPoint(geo[0] + 0 * geo[1] + ysize * geo[2], geo[3] + 0 * geo[4] + ysize * geo[5])
    ring.AddPoint(geo[0] + 0 * geo[1] + 0 * geo[2], geo[3] + 0 * geo[4] + 0 * geo[5])
    poly.AddGeometry(ring)
    return poly


def generate_clip_shape_from(fn: str, shpfn: str) -> str:
    """
    Take a filename to a raster object (could be in /vsimem) and
    generate the shape of the observation.
    """
    fd = gdal.Open(fn)

    geo = fd.GetGeoTransform()

    xsize = fd.RasterXSize
    ysize = fd.RasterYSize

    driver = ogr.GetDriverByName("GeoJSON")
    ds = driver.CreateDataSource(shpfn)

    srs = osr.SpatialReference()
    srs.ImportFromWkt(fd.GetProjectionRef())
    layer = ds.CreateLayer("obs", srs)

    poly = polygon_from_geobox(geo, xsize, ysize)

    log(f"Wkt: {poly.ExportToWkt()}")

    fdef = layer.GetLayerDefn()
    oftr = ogr.Feature(fdef)

    oftr.SetGeometry(poly)

    layer.CreateFeature(oftr)

    oftr = None
    ds = None

    return shpfn


def run(
    url=None,
    obstmp=None,
    clipshpfn=None,
    inputs=None,
    tmpdir=None,
    models=None,
    nocleanup=False,
    **args,
):
    """
    Load and prepare the data required for the change detection algorithms
    and then pass this data to the algorithm. Use `args` to parametrise.
    """

    log("# Loading models")

    loaded_models = []

    for m in models:

        config = deepcopy(m)

        name = config.pop("name")
        outfn = config.pop("output")
        driver = config.pop("driver")
        inputs = config.pop("inputs")

        log(f"Model: {name} -> {outfn}")

        try:

            model = get_model(name, **config)
            if model.verbose:
                model.log = log

        except IncorrectChecksumError as e:
            warning(f"Model has an incorrect SHA256 checksum, exiting...")
            sys.exit(1)

        loaded_models.append(model)

    log("# Retrieving NRT observation details")

    url = normalise_url(url)

    log(f"Obs. URL:  {url}")

    obswkt = get_bounds(url)
    obsdate = parse_obsdate(url)

    log(f"Obs. Date: {obsdate}")
    log(f"Obs. WKT:  {wktfmt(obswkt)}")

    source = DEASentinel2()

    # Determine the minimal set of bands required across all models

    def name(x):
        if isinstance(x, int):
            return source.bands[x]
        else:
            return x

    bands = {band: False for band in source.bands}
    for model in loaded_models:
        try:
            required = [name(b) for b in model.required_bands]
        except AttributeError:
            required = source.bands
        for band in required:
            bands[band] = True
    bands = [k for k, v in bands.items() if v is True]

    log(f"Req.Bands: {','.join(bands)}")

    if len(bands) == 0:
        # get at least one band
        bands = ["B02"]

    obsgeo, obsprj, obsdata, mask = source.get_observations(url, bands_required=bands)

    ysize, xsize = mask.shape
    obspoly = polygon_from_geobox(obsgeo, xsize, ysize)

    log(f"# Preparing ancillary data")

    log(f"Writing mask to mask.tif")

    driver = gdal.GetDriverByName("GTiff")
    fd = driver.Create("mask.tif", xsize, ysize, 1, gdal.GDT_Byte)
    fd.SetGeoTransform(obsgeo)
    fd.SetProjection(obsprj)
    ob = fd.GetRasterBand(1)
    ob.WriteArray(mask)
    ob.SetNoDataValue(0)
    del fd

    ysize, xsize, psize = obsdata.shape

    log(f"Writing observation data to {obstmp}. Data has {psize} bands.")

    driver = gdal.GetDriverByName("GTiff")
    fd = driver.Create(obstmp, xsize, ysize, psize, gdal.GDT_Float32)
    fd.SetGeoTransform(obsgeo)
    fd.SetProjection(obsprj)
    for i in range(fd.RasterCount):
        ob = fd.GetRasterBand(i + 1)
        ob.WriteArray(obsdata[:, :, i])
        ob.SetNoDataValue(np.nan)
        ob.SetDescription(bands[i])
    del fd

    log(f"Determining ancillary files required")

    outputs = []
    inputfns = []
    for model in models:
        name = model["name"]

        log(f"Checking '{name}' model")

        output = model["output"]

        ips = model["inputs"]
        for ip in ips:
            fn = ip["filename"]
            if fn not in outputs:
                inputfns.append(fn)

        outputs.append(output)

    log(f"# Warping and clipping ancillary data")

    # Get the unique inputs

    inputfns = [*{*inputfns}]

    if len(inputfns) > 0:
        log("Determining clip area from NRT observation")

        clipshpfn = generate_clip_shape_from(obstmp, clipshpfn)

        if not clipshpfn.startswith("/vsimem"):
            log(f"Proj: {obsprj}")
            log(f"Saving clipping area to disk as '{clipshpfn}'")
    else:
        log(f"No ancillary datas are required!")

    datamap = {}

    obssr = osr.SpatialReference()
    obssr.ImportFromProj4(obsprj)

    for afn in inputfns:
        ofn = f"{tmpdir}/{uuid.uuid4()}"

        fd = gdal.Open(afn)
        geo = fd.GetGeoTransform()
        prj = fd.GetProjection()

        insr = osr.SpatialReference()
        insr.ImportFromProj4(prj)

        insr_to_obssr = osr.CoordinateTransformation(insr, obssr)
        poly = polygon_from_geobox(geo, fd.RasterXSize, fd.RasterYSize)
        poly.Transform(insr_to_obssr)

        if not poly.Intersects(obspoly):
            raise InputDataError(f"Input data '{afn}' does not intersect observation.")

        log(f"Clipping and warping input '{afn}' to '{ofn}'")

        fd = gdal.Warp(ofn, fd, cutlineDSName=clipshpfn, cropToCutline=True, dstSRS=obsprj)

        datamap[afn] = ofn

    # Get processing configuration parameters

    tilewidth = args.pop("tilewidth", None)
    obsscale = args.pop("obsscale", None)

    # Scale observation data

    if obsscale is not None:
        log(f"Scaling observation data by {obsscale}")
        obsdata *= float(obsscale)

    log("# Applying loaded models to data")

    for model, m in zip(loaded_models, models):

        config = deepcopy(m)

        name = config.pop("name")

        log(f"@Running '{name}' model")

        # Update model config based on new information from observation

        config["obsurl"] = url
        config["obswkt"] = obswkt
        config["obsdate"] = obsdate
        config["geo"] = obsgeo
        config["prj"] = obsprj
        config["bands"] = bands  # possibly reduced set of bands

        log("Observation data:")
        log(f"   data min: {np.nanmin(obsdata)} max: {np.nanmax(obsdata)}")
        log(f"   pixel resolution: {obsgeo[1]:.4f} x {obsgeo[5]:.4f}")

        model.update(**config)

        # Prepare all the appropriate ancillary data sets and pass the
        # observation data as the last one in the list.

        datas = [mask.copy()]

        outfn = m["output"]
        inputfns = m["inputs"]

        log("Loading model inputs:")
        for ip in inputfns:
            fn = datamap[ip["filename"]]

            fd = gdal.Open(fn)

            geo = fd.GetGeoTransform()

            # First assume bands are the same as source
            ipbands = source.bands
            try:
                # Then see if they are overwritten in config
                ipbands = ip["bands"]
            except KeyError:
                # If that fails, try to get bandnames from file
                fbands = []
                for i in range(fd.RasterCount):
                    rb = fd.GetRasterBand(i + 1)
                    desc = rb.GetDescription()
                    if len(desc) > 0:
                        fbands.append(desc)
                if len(fbands) == fd.RasterCount:
                    ipbands = fbands

            log(f" - path:    {ip['filename']}")
            log(f"   bands:   {','.join(ipbands)}")

            notreq = set(source.bands) - set(bands)
            toload = [b for b in ipbands if b not in notreq]
            bandidx = [i + 1 for i, b in zip(range(fd.RasterCount), ipbands) if b not in notreq]
            log(f"   loading: {','.join(toload)}")

            nbands = len(bandidx)

            data = np.empty((ysize, xsize, nbands), dtype=np.float32)
            for i, bi in enumerate(bandidx):
                band = fd.GetRasterBand(bi)
                band.ReadAsArray(
                    buf_type=gdal.GDT_Float32,
                    buf_xsize=xsize,
                    buf_ysize=ysize,
                    buf_obj=data[:, :, i],
                )

            nodata = fd.GetRasterBand(1).GetNoDataValue()
            data[data == nodata] = np.nan

            nnan = np.count_nonzero(np.isnan(data))
            nval = xsize * ysize * nbands
            pnan = nnan / nval
            if pnan > 0.9:
                warning(f"clipped input '{afn}' has more than 90% no data")

            scale = ip.pop("scale", None)
            if scale is not None:
                log(f"   scaling: {scale}")
                data *= scale

            log(f"   data min: {np.nanmin(data)} max: {np.nanmax(data)}")
            log(f"   pixel resolution: {geo[1]:.4f} x {geo[5]:.4f}")

            datas.append(data)

        datas.append(obsdata)

        log(f"Output: {outfn}")

        log("Running model predictions")

        # The model is responsible for saving its prediction to disk (or memory
        # using /vsimem) as it is best placed to make a decision on the format, etc.
        # A simple model only needs to implement the `predict` method but can also
        # implement `predict_and_save` if more control of writing output is needed.

        if tilewidth:
            model = TiledPrediction(model, int(tilewidth))

        model.predict_and_save(outfn, *datas)

        datamap[outfn] = outfn

        log("Finished running predictions")

    if nocleanup:
        return

    log("# Cleaning up")

    for k, fn in datamap.items():
        log(f"Removing {fn}")
        gdal.Unlink(fn)


def astuple(v):
    """
    Cast to tuple but handle the case where 'v' could be a
    str of values '(a,b,c,...)' and then parse the values
    as floats.
    """
    if isinstance(v, str):
        return tuple([float(x) for x in v[1:-1].split(",")])
    else:
        return tuple(v)


def check_config(args):
    """
    Check config and set some defaults if necessary.
    """
    log(f"# Checking configuration")

    errors = False

    # Set some defaults

    defaults = [
        ("quiet", False),
        ("product", "NBAR"),
        ("obstmp", "/tmp/obs.tif"),
        ("clipshpfn", "/tmp/clip.json"),
        ("tmpdir", "/tmp"),
        ("nocleanup", False),
        ("gdalconfig", {}),
        ("models", []),
    ]

    for arg, value in defaults:
        if not arg in args:
            log(f"'{arg}' not set, setting to default: {arg} = {value}")
            args[arg] = value

    # Set GDAL config

    for k, v in args["gdalconfig"].items():
        if v == True:
            v = "YES"
        if v == False:
            v = "NO"
        gdal.SetConfigOption(k, v)
        log(f"GDAL option {k} = {v}")

    ## Check that models exists

    if not isinstance(args["models"], list):
        log("'models' must be a list of models")
        errors = True

    # Set some model defaults

    avail = [gdal.GetDriver(i).ShortName for i in range(gdal.GetDriverCount())]

    # Check the models

    models = args["models"]
    for m in models:

        name = m["name"]

        if name[:7] == "file://":
            try:
                path = name[7:]
                path, checksum = path.split(":")
            except ValueError:
                log(f"Incorrect model name format, it should be file://filename:sha256checksum")
                errors = True

        if name[:5] == "s3://":
            try:
                path = name[5:]
                path, checksum = path.split(":")
            except ValueError:
                log(f"Incorrect model name format, it should be s3://bucket/key:sha256checksum")
                errors = True

        if "driver" not in m:
            if gdal.GetDriverByName("COG"):
                m["driver"] = "COG"
            else:
                m["driver"] = "GTiff"

        if m["driver"] not in avail:
            log(f"'driver' for model '{name}' is not available")
            log(f"available drivers: {avail}")
            errors = True

        if "inputs" not in m:
            m["inputs"] = []

        # Normalise the urls

        for ip in m["inputs"]:
            fn = normalise_url(ip["filename"])
            ip["filename"] = fn

        # Parse tuples as tuples of numbers for models

        for k, v in m.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, str) and vv.startswith("(") and vv.endswith(")"):
                        v[kk] = astuple(vv)

        defaults = [
            ("output", "result.tif"),
        ]

        for arg, value in defaults:
            if not arg in m.keys():
                log(f"Model {name} '{arg}' not set, setting to default: {arg} = {value}")
                m[arg] = value

    args["models"] = models

    if errors:
        sys.exit(1)

    # TODO: Check GPU resources if needed?

    return args


def cli_entry(url=None, **kwargs):
    """
    Parse settings from command line and settings file into `args`,
    run, and CLI interface.
    """
    parser = argparse.ArgumentParser()

    if url is None:
        parser.add_argument("url")

    parser.add_argument("-config", default="nrtpredict.yaml", metavar=("yamlfile"))
    parser.add_argument("-tilewidth", default=None)

    # ...

    args = vars(parser.parse_args())

    if url:
        args["url"] = url

    # Try to load configuration file.

    try:

        fn = args["config"]
        with open(fn) as fd:
            fargs = yaml.safe_load(fd)
            if fargs:
                args = {**fargs, **args}

        log(f"Loading configuration from '{fn}'")

    except yaml.parser.ParserError as e:
        warning(f"Configuration file '{fn}' has incorrect YAML syntax: {e}")
        sys.exit(100)

    except FileNotFoundError:
        warning(f"Configuration file '{fn}' not found.")
        warning("\nContinuing without configuration file...")

    # Overwrite based on what is passed to main

    for k, v in kwargs.items():
        args[k] = v

    # Run this thing

    try:
        args = check_config(args)

        run(**args)

        log("Finished.")

    except URLParsingError as e:
        warning(str(e))
        sys.exit(1)

    except ConnectionError as e:
        warning(f"Connection Error: {e}")
        sys.exit(2)

    except RuntimeError as e:
        warning(f"Runtime Error: {e}")
        sys.exit(3)

    except NameError as e:
        warning(f"Unknown: {e}")
        sys.exit(4)

    except InputDataError as e:
        warning(f"{e}")
        sys.exit(5)

    except KeyboardInterrupt:
        warning(f"Processing interrupted, exiting...")
        sys.exit(1000)


if __name__ == "__main__":
    cli_entry()
