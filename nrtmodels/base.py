import numpy as np
import sys

from osgeo import gdal

DTYPEMAP = {"float32": gdal.GDT_Float32,
            "int16": gdal.GDT_Int16,
            "int8": gdal.GDT_Byte,
            "uint8": gdal.GDT_Byte,
            "bool": gdal.GDT_Byte}

class Model:

    def __init__(self, **kwargs):
        self.verbose = True
        self.nodata = np.nan
        self.update(**kwargs)

    def update(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def eval_expr(self, ftrexpr, data):
        self.log(f'Generating feature from: {ftrexpr}')
        self.log(f"data shape: {data.shape}")
        try:
            env = {
                **{b: data[b] for b in data.keys()},
                **{s: getattr(np, s) for s in dir(np)},
            }
            self.log('Evaluation environment created based on dict of data')
        except (IndexError, AttributeError):
            bandnos = {b: i for i, b in enumerate(self.bands)}
            env = {
                **{b: data[:,:,bandnos[b]] for b in self.bands},
                **{s: getattr(np, s) for s in dir(np)},
            }
            self.log('Evaluation environment created based on dense data')
        with np.errstate(all='ignore'):
            result = np.empty((data.shape[0], data.shape[1]), dtype=np.float32)
            result[:,:] = eval(ftrexpr, {"__builtins__": {}}, env)
            mask = ~np.isfinite(result)
            result[mask] = np.nan
        self.log('Expression evaluated')
        return result

    def log(self, s):
        if self.verbose:
            print(s, file=sys.stderr)

    def predict(self, *datas):
        raise NotImplementedError

    def predict_and_save(self, fn, *datas):
        result = self.predict(*datas)

        if result is None:
            return

        if len(result.shape) == 2:
            result = result[:, :, np.newaxis]

        driver = self.driver

        self.log(f"Writing {driver} output to {fn}")

        make_cog = False
        if driver == "COG":
            make_cog = True
            ofn = fn
            fn = '/vsimem/tmp.tif'
            driver = "GTiff"

        do = gdal.GetDriverByName(driver)
        dtype = DTYPEMAP[result.dtype.name]

        nodata = result.dtype.type(self.nodata).item()

        self.log(f"dtype: {result.dtype} nodata: {nodata} nbands: {result.shape[-1]}")

        fd = do.Create(fn, result.shape[1], result.shape[0],
                           result.shape[2], dtype)

        fd.SetGeoTransform(self.geo)
        fd.SetProjection(self.prj)

        for i in range(fd.RasterCount):
            ob = fd.GetRasterBand(i + 1)
            ob.WriteArray(result[:, :, i])
            ob.SetNoDataValue(nodata)
            try:
                ob.SetDescription(self.description[i])
            except AttributeError:
                pass

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


class BandTransform(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.expr = kwargs.pop('expr', 'B02')
        self.required_bands = ['B02']

    def predict(self, mask, *datas):
        print(f'expr: {self.expr}')
        results = []
        for data in datas:
            dd = {}
            for i, bn in enumerate(self.required_bands):
                dd[bn] = data[:,:,i]
            print(dd)
            result = self.eval_expr(self.expr, dd)
            result[mask] = 0
            bad = np.isnan(result)
            result[bad] = 0
            results.append(result)
        return np.dstack(results)
