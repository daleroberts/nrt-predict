import numpy as np

from osgeo import gdal, ogr, osr

DTYPEMAP = {"float32": gdal.GDT_Float32,
            "int16": gdal.GDT_Int16,
            "int8": gdal.GDT_Byte}

class Model:

    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

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

    def predict(self, data):
        if isinstance(data, list):
                return np.dstack(data)
        else:
            return data
