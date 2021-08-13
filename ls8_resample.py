# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 20:49:22 2021

@author: x51b783
"""
from osgeo import gdal, gdalconst
import topo_correction_util as tcu
import numpy as np

ls8_path = 'D:/field_data/YC/YC20210428/ls8/LC08_038029_20210428.tif'
ls8 = gdal.Open(ls8_path, gdalconst.GA_ReadOnly)
ls8_UTM = gdal.Warp('D:/field_data/YC/YC20210428/ls8/20210428_ls8_UTM.tif',ls8,dstSRS = 'EPSG:32612') # reproject elevation GeoTiff from WGS to UTM

ls8_albedo_band = ls8_UTM.GetRasterBand(1)
ndv = ls8_albedo_band.GetNoDataValue()
ls8_albedo = ls8_albedo_band.ReadAsArray()
ls8_albedo[ls8_albedo==0] = np.nan
ls8_albedo[ls8_albedo==ndv] = np.nan

tcu.write_geotiff('D:/field_data/YC/YC20210428/ls8/20210428_ls8_albedo.tif', np.shape(ls8_albedo), ls8_UTM.GetGeoTransform(), 32612, ls8_albedo)


bg_path = 'D:/field_data/YC/bareground/elevation/elevation_merged.tif'
bg = gdal.Open(bg_path, gdalconst.GA_ReadOnly)

usgs_path = 'D:/field_data/YC/3DEP/3DEP_DEM_UTM.tif'
usgs = gdal.Open(usgs_path, gdalconst.GA_ReadOnly)

x = gdal.Open('D:/field_data/YC/YC20210428/ls8/20210428_ls8_albedo.tif', gdalconst.GA_ReadOnly)
tcu.resample(x, bg, 'D:/field_data/YC/bareground/ls8/YC20210428_ls8_bg.tif')
tcu.resample(x, usgs, 'D:/field_data/YC/3DEP/ls8/YC20210428_ls8_3DEP.tif')


