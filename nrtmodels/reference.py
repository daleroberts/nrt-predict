import numpy as np
import boto3
import yaml
import re

from osgeo import gdal
from botocore import UNSIGNED
from botocore.client import Config

from datetime import datetime, date, timedelta

from .base import Model

BANDNAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

s3 = None


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


def list_dir(bucket: str, prefix: str = "") -> list:
    """
    List a directory stored in a s3 bucket.
    """
    result = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    return [o.get("Prefix") for o in result.get("CommonPrefixes")]


def get_nrt_obs_daterange() -> tuple:
    """
    Get the minimum and maximum date range for NRT observations.
    """
    keys = list_dir("dea-public-data", prefix="L2/sentinel-2-nrt/S2MSIARD/")

    def parsedate(key):
        return date.fromisoformat(re.search(r"\d{4}-\d{2}-\d{2}", key).group())

    return (parsedate(keys[0]), parsedate(keys[-1]))


def get_metadata(key: str) -> dict:
    """
    Get the metadata for a NRT package.
    """
    resp = s3.get_object(Bucket="dea-public-data", Key=f"{key.strip('/')}/ARD-METADATA.yaml")
    return yaml.safe_load(resp["Body"])


def get_cloud_cover(key: str) -> float:
    """
    Get cloud cover of observation.
    """
    metadata = get_metadata(key)
    return float(metadata["lineage"]["source_datasets"]["level1"]["image"]["cloud_cover_percentage"])


def find_prev_clear(current, cutoff=5):
    """
    Find the previous clear observation.
    """
    mindate, maxdate = get_nrt_obs_daterange()
    s = current.split("/")
    bucket, obsdate, package = s[0], date.fromisoformat(s[-2]), s[-1]
    tile = package.split("_")[-2]
    possible = []
    for i in range(1, (obsdate - mindate).days - 1):
        dt = obsdate - timedelta(days=i)
        keys = [p for p in list_dir("dea-public-data", prefix=f"L2/sentinel-2-nrt/S2MSIARD/{dt}/") if tile in p]
        for key in keys:
            cc = get_cloud_cover(key)
            if cc < cutoff:
                return key.strip("/")
                break
    return None


class PreviousClear(Model):

    """
    Get previous clear observation.

    Developed by Dale Roberts <dale.o.roberts@gmail.com>
    """

    def __init__(self, **kwargs):
        self.required_bands = BANDNAMES
        super().__init__(**kwargs)

        global s3
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    def predict(self, mask, obs):
        self.log(f"{self.obsurl}")
        pkg = parse_pkg(self.obsurl)
        odt = self.obsdate.date()

        key = find_prev_clear(f"dea-public-data/L2/sentinel-2-nrt/S2MSIARD/{odt}/{pkg}")
        url = f"/vsis3/dea-public-data/{key}"

        product = "NBART"
        pkg = parse_pkg(url)

        bands = self.required_bands
        self.description = self.required_bands

        fn = [fn for fn in gdal.ReadDir(f"{url}/QA") if fn.endswith("FMASK.TIF")][0]
        fn = f"{url}/QA/{fn}"

        fd = gdal.Open(fn)
        mask = fd.ReadAsArray()

        geo = fd.GetGeoTransform()
        prj = fd.GetProjection()

        ysize = mask.shape[0]
        xsize = mask.shape[1]

        fns = [fn for fn in gdal.ReadDir(f"{url}/{product}") if fn.endswith(".TIF") or fn.endswith(".tif")]
        bfm = {fn.split(f"{product}_")[1].split(".")[0]: fn for fn in fns}

        data = np.empty((ysize, xsize, len(bands)), dtype=np.float32)
        for i, band in enumerate(bands):
            fn = f"{url}/{product}/{bfm[band]}"
            fd = gdal.Open(fn)
            fd.ReadAsArray(buf_obj=data[:, :, i], buf_ysize=ysize, buf_xsize=xsize)

        nodata = fd.GetRasterBand(1).GetNoDataValue()
        data[data == nodata] = np.nan

        self.geo = geo
        self.prj = prj

        return data
