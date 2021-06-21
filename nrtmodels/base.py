import numpy as np
import sys

from osgeo import gdal

DTYPEMAP = {"float32": gdal.GDT_Float32,
            "int16": gdal.GDT_Int16,
            "int8": gdal.GDT_Byte,
            "bool": gdal.GDT_Byte}

class Model:

    def __init__(self, **kwargs):
        self.verbose = True
        self.update(**kwargs)

    def update(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def eval_expr(self, ftrexpr, data):
        try:
            env = {
                **{b: data[b] for b in data.keys()},
                **{s: getattr(np, s) for s in dir(np)},
            }
        except IndexError:
            bandnos = {b: i for i, b in enumerate(self.bands)}
            env = {
                **{b: data[:,:,bandnos[b]] for b in self.bands},
                **{s: getattr(np, s) for s in dir(np)},
            }
        return eval(ftrexpr, {"__builtins__": {}}, env)

    def log(self, s):
        if self.verbose:
            print(s, file=sys.stderr)

    def predict(self, *datas):
        raise NotImplementedError

    def predict_and_save(self, fn, *datas):
        result = self.predict(*datas)

        if len(result.shape) == 2:
            result = result[:, :, np.newaxis]

        driver = self.driver

        make_cog = False
        if driver == "COG":
            make_cog = True
            ofn = fn
            fn = '/vsimem/tmp.tif'
            driver = "GTiff"

        do = gdal.GetDriverByName(driver)
        dtype = DTYPEMAP[result.dtype.name]

        fd = do.Create(fn, result.shape[1], result.shape[0],
                           result.shape[2], dtype)

        fd.SetGeoTransform(self.geo)
        fd.SetProjection(self.prj)

        for i in range(fd.RasterCount):
            ob = fd.GetRasterBand(i + 1)
            ob.WriteArray(result[:, :, i])
            ob.SetNoDataValue(np.nan)
            #ob.SetDescription(self.description[i])

        if make_cog:
            ds = gdal.GetDriverByName('COG').CreateCopy(ofn, fd)
            gdal.Unlink(fn)
            del fd


class NoOp(Model):

    def predict(self, mask, *datas):
        return np.dstack([*datas])


class FirstBand(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.required_bands = [0]

    def predict(self, mask, *datas):
        return np.dstack([*datas])
