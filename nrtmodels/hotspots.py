import numpy as np
import requests

from datetime import datetime, timedelta
from osgeo import gdal, ogr, osr
from skimage import filters, morphology
from uuid import uuid4

from .base import Model

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

PROBPERCCLIP = (0, 100)


def log(s):
    print(s)


def identify_epsg(prj):
    """
    Identify the EPSG from a Proj4 or EPSG string.
    """
    sr = osr.SpatialReference()
    if prj.startswith('EPSG'):
        sr.ImportFromEPSG(int(prj.split(":")[1]))
    else:
        sr.ImportFromProj4(prj)
    sr.AutoIdentifyEPSG()
    return sr.GetAuthorityCode(None)


def get_hotspots(wkt, fromdate, todate=None, s_prj='EPSG:4326',
                 t_prj=None, username=None, password=None,
                 maxpoints=30000):
    """
    """
    sepsg = identify_epsg(s_prj)
    tepsg = identify_epsg(t_prj)

    fromdt = fromdate.strftime('%Y-%m-%d')
    todt = todate.strftime('%Y-%m-%d')

    cql = f"datetime >= {fromdt}"
    if todate:
        cql += f" AND datetime <= {todt}"

    geom = ogr.CreateGeometryFromWkt(wkt)
    bbox = geom.GetEnvelope() # minx,maxx,miny,maxy

    # Googling says BBOX(location,minx,miny,maxx,maxy) but the only way
    # I can get this to work is the following ordering:
    
    cql += f" AND BBOX(location, {bbox[2]}, {bbox[1]}, {bbox[3]}, {bbox[0]})" 

    params = {
       'service': 'WFS',
       'version': '1.1.0',
       'request': 'GetFeature',
       'typeName': 'public:hotspots',
       'outputFormat': 'application/json',
       'maxFeatures': maxpoints,
       'srsName': f'EPSG:{sepsg}',
       'CQL_FILTER': cql,
       'sortBy': 'sensor',
    }
    url = f"https://hotspots.dea.ga.gov.au/geoserver/public/wfs"
    req = requests.get(url, params=params,
                       auth=(username, password))

    json = req.json()
    ncount = json['totalFeatures']

    #with open('raw.json', 'wb') as fd:
    #    fd.write(req.content)

    mmapfn = "/vsimem/" + uuid4().hex
    gdal.FileFromMemBuffer(mmapfn, req.content)

    fd = ogr.Open(mmapfn)
    layer = fd.GetLayer()
    rbbox = layer.GetExtent()

    fn = "/vsimem/" + uuid4().hex

    fd = gdal.VectorTranslate(fn, mmapfn, format='GeoJSON',
                              srcSRS=f"EPSG:{sepsg}",
                              dstSRS=f'EPSG:{tepsg}')
    fd = None # force writing

    return fn


class Hotspots(Model):

    def predict_and_save(self, datas, fn):
        geo = self.geo
        prj = self.prj
        xsize = self.xsize
        ysize = self.ysize
        obsdate = self.obsdate
        obswkt = self.obswkt
        username = self.username
        password = self.password
        translations = self.translations
        resolutions = self.resolutions

        todate = obsdate + timedelta(days=14)
        fromdate = obsdate - timedelta(days=14)

        ptsfn = get_hotspots(obswkt, fromdate=fromdate, todate=todate,
                             t_prj=prj, username=username, password=password)

        fd = ogr.Open(ptsfn)
        layer = fd.GetLayer()

        sensors = set()
        for ftr in layer:
            sensors.add(ftr.GetField('sensor'))

        driver = ogr.GetDriverByName('GeoJSON')
        
        tsr = osr.SpatialReference()
        tsr.ImportFromProj4(prj)

        datas = []

        for sensor in sensors:

            layer.SetAttributeFilter(f"sensor='{sensor}'")
            count = layer.GetFeatureCount()

            ds = driver.CreateDataSource(f"/vsimem/{sensor}.json")
            olayer = ds.CreateLayer(sensor, tsr, geom_type=ogr.wkbMultiPolygon)

            ldef = layer.GetLayerDefn()
            for i in range(ldef.GetFieldCount()):
                fdef = ldef.GetFieldDefn(i)
                olayer.CreateField(fdef)

            oldef = olayer.GetLayerDefn()

            for ftr in layer:
                geom = ftr.GetGeometryRef()

                dx, dy = translations[sensor]
                if dx != 0 or dy != 0:
                    x, y = geom.GetPoint_2D()
                    geom.SetPoint(0, x+dx, y+dy)

                geom = geom.Buffer(resolutions[sensor])

                oftr = ogr.Feature(oldef)
                oftr.SetGeometry(geom)

                for i in range(oldef.GetFieldCount()):
                    oftr.SetField(oldef.GetFieldDefn(i).GetNameRef(), ftr.GetField(i))

                olayer.CreateFeature(oftr)

            olayer.SyncToDisk() 
            ds = None
        
            geo = self.geo
            bounds = [geo[0], geo[3], geo[0]+xsize*geo[1], geo[3]+geo[5]*ysize]

            ds = gdal.Rasterize(f"{sensor}.tif", f"/vsimem/{sensor}.json",
                                outputBounds=bounds, width=xsize, height=ysize,
                                attribute="temp_kelvin")

            geo = ds.GetGeoTransform()
            prj = ds.GetProjection()

            data = ds.ReadAsArray()

            #pl, pu = np.percentile(data, PROBPERCCLIP)
            #if np.abs(pu - pl) > 0:
            #    data -= pl
            #    data /= (pu - pl)
            #    np.clip(data, 0, 1, out=data)

            #log(f"{sensor}\tpoints: {count} pl: {pl} pu: {pu}")

            #data /= 100.

            #data = morphology.convex_hull_object(data > 0, connectivity=2)
            filters.gaussian(data, sigma=25, preserve_range=True, truncate=3, output=data)
            mask = data > 0
            morphology.binary_erosion(mask, morphology.disk(10), out=mask)
            data[mask == 0] = np.nan

            rb = ds.GetRasterBand(1)
            rb.WriteArray(data)
            rb.SetNoDataValue(np.nan)

            datas.append(data)

        data = np.dstack(datas)

        #data = np.nanmean(data, axis=-1)
        #data = data[:,:,np.newaxis]
        #sensors = ['squashed']

        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(fn, data.shape[1], data.shape[0], data.shape[2], gdal.GDT_Float32)
        ds.SetGeoTransform(geo)
        ds.SetProjection(prj)

        for i, sensor in zip(range(data.shape[2]), sensors):
            rb = ds.GetRasterBand(i+1)
            rb.WriteArray(data[:,:,i].astype(np.float32))
            rb.SetNoDataValue(0.0)
            rb.SetDescription(sensor)

        del ds
