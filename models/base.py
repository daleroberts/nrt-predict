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

    def predict(self, datas):
        raise NotImplementedError

    def predict_and_save(self, datas, fn):
        result = self.predict(datas)

        if len(result.shape) == 2:
            result = result[:, :, np.newaxis]

        driver = gdal.GetDriverByName(self.driver)
        dtype = DTYPEMAP[result.dtype.name]

        fd = driver.Create(fn, result.shape[1], result.shape[0],
                           result.shape[2], dtype)

        fd.SetGeoTransform(self.geo)
        fd.SetProjection(self.prj)

        for i in range(fd.RasterCount):
            ob = fd.GetRasterBand(i + 1)
            ob.WriteArray(result[:, :, i])
            ob.SetNoDataValue(0)

        del fd
