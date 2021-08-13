# -*- coding: utf-8 -*-
"""
Created on Mon May 17 12:55:44 2021

@author: x51b783
"""

import topo_correction_util as tcu
import numpy as np
import pandas as pd
import seaborn as sns
import angles
import os
os.environ['PROJ_LIB'] = 'C:\\Users\\x51b783\\.conda\\envs\\gdal\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\x51b783\\.conda\\envs\gdal\\Library\\share'
from osgeo import gdal, gdalconst
from scipy import ndimage
from PIL import Image

original_usgs_path = 'D:/field_data/YC/3DEP/3DEP_DEM_UTM.tif'
original_sfm_path = 'D:/field_data/YC/YC20210428/sfm/elevation/dem_UTM_clipped.tif'

warped_usgs_path = 'D:/field_data/YC/3DEP/resampled/3DEP_DEM_UTM_res.tif'
warped_sfm_path = 'D:/field_data/YC/YC20210428/sfm/elevation/dem_UTM_clipped_res.tif'

merged_destination = 'D:/field_data/YC/YC20210428/sfm/elevation/dem_merged.tif'
#ls8_path = 'C:/Temp/0311/ls8/LC08_038029_20210311_albedo.tif'
usgs_offset = -13


usgs_source = gdal.Open(original_usgs_path, gdalconst.GA_ReadOnly)

sfm_source = gdal.Open(original_sfm_path, gdalconst.GA_ReadOnly)
#ls8_source = gdal.Open(ls8_path, gdalconst.GA_ReadOnly)


#ls8_gt = usgs_source.GetGeoTransform()
#ls8_proj = usgs_source.GetProjection()
#ls8_cols = usgs_source.RasterXSize
#ls8_rows = usgs_source.RasterYSize
#ls8_usgs = usgs_source.GetRasterBand(1).ReadAsArray()


usgs_gt = usgs_source.GetGeoTransform()
usgs_proj = usgs_source.GetProjection()
usgs_cols = usgs_source.RasterXSize
usgs_rows = usgs_source.RasterYSize
arr_usgs = usgs_source.GetRasterBand(1).ReadAsArray()




sfm_gt = sfm_source.GetGeoTransform()
sfm_proj = sfm_source.GetProjection()
sfm_cols = sfm_source.RasterXSize
sfm_rows = sfm_source.RasterYSize
#arr_sfm = sfm_source.GetRasterBand(1).ReadAsArray()

minX = usgs_gt[0]
maxX = usgs_gt[0] + (usgs_cols * usgs_gt[1])

minY = usgs_gt[3] + (usgs_rows * usgs_gt[5])
maxY = usgs_gt[3]

new_geo = [usgs_gt[0], sfm_gt[1], 0.0, usgs_gt[3], 0.0, sfm_gt[5]]


new_cols = usgs_gt[1] * usgs_cols / sfm_gt[1]
new_rows = usgs_gt[5] * usgs_rows / sfm_gt[5]



warp_options = gdal.WarpOptions(format = 'GTiff', outputBounds = [minX, minY, maxX, maxY], 
                                width = new_cols, height = new_rows,
                                srcSRS = sfm_proj, dstSRS = sfm_proj)
out = gdal.Warp(warped_sfm_path, sfm_source, options = warp_options)


warp2_options = gdal.WarpOptions(format = 'GTiff', outputBounds = [minX, minY, maxX, maxY], 
                                width = new_cols, height = new_rows,
                                srcSRS = usgs_proj, dstSRS = usgs_proj)
out2 = gdal.Warp(warped_usgs_path, usgs_source, options = warp2_options)


'''


warp3_options = gdal.WarpOptions(format = 'GTiff', outputBounds = [minX, minY, maxX, maxY], 
                                width = new_cols, height = new_rows,
                                srcSRS = ls8_proj, dstSRS = sfm_proj)
out2 = gdal.Warp('C:/Temp/0311/ls8/LC08_038029_20210311_albedo_res.tif', ls8_source, options = warp3_options)
'''

#dst_filename = 'C:/Users/x51b783/Documents/Mirror/Masters/Class/field_measurement_snow/project/CARC_DEM_resampled.tif'

#dst = gdal.GetDriverByName('GTiff').Create(dst_filename, usgs_cols, usgs_rows, 1, gdalconst.GDT_Float32)
#dst.SetGeoTransform(new_geo)
#dst.SetProjection(dem_proj)

# Do the work
#gdal.ReprojectImage(ls8, dst, ls8_proj, dem_proj)


warped_usgs_source = gdal.Open(warped_usgs_path, gdalconst.GA_ReadOnly)
warped_sfm_source = gdal.Open(warped_sfm_path, gdalconst.GA_ReadOnly)

warped_sfm_band = warped_sfm_source.GetRasterBand(1)
warped_usgs_band = warped_usgs_source.GetRasterBand(1)

sfm_ndv = warped_sfm_band.GetNoDataValue()
usgs_ndv = warped_usgs_band.GetNoDataValue()

warped_sfm_arr = warped_sfm_band.ReadAsArray()
warped_sfm_arr[warped_sfm_arr==sfm_ndv] = np.nan
warped_sfm_arr[warped_sfm_arr==0.0] = np.nan
print(np.nanmax(warped_sfm_arr))
print(np.nanmin(warped_sfm_arr))

warped_usgs_arr = warped_usgs_band.ReadAsArray()
warped_usgs_arr[warped_usgs_arr==usgs_ndv] = np.nan
warped_usgs_arr += usgs_offset

indices = np.where(np.isnan(warped_sfm_arr))

warped_sfm_arr[indices] = warped_usgs_arr[indices]

print(np.nanmax(warped_sfm_arr))
print(np.nanmin(warped_sfm_arr))

'''
warped_usgs_arr[indices] = warped_sfm_arr[indices]
warped_usgs_arr[indices] = 7
warped_usgs_arr[warped_usgs_arr == sfm_ndv] = np.nan
warped_usgs_arr[warped_usgs_arr == usgs_ndv] = np.nan
'''

tcu.write_geotiff(merged_destination, np.shape(warped_usgs_arr), new_geo, 32612, warped_sfm_arr)

sfm_source = None
usgs_source = None
warped_sfm_source = None
warped_usgs_source = None
out = None
out2 = None
