# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 08:47:48 2020

@author: aman3
"""


import rawpy
import imageio
import os
import rasterio
import shutil
import exifread
import pandas as pd
import pytz
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as ndimage
import exiftool
import dateutil
from pytz import timezone
from rasterio.plot import show
import math
import arcgis
import sys
from osgeo import gdal, gdalconst
import struct
from osgeo.gdalnumeric import *

path_to_dem =  'C:/Users/aman3/Documents/GradSchool/testing/50mDEM.tif'
path_to_output = 'C:/Users/aman3/Documents/GradSchool/testing/50mAspect.tif'
ortho_dir = 'C:/Users/aman3/Documents/GradSchool/testing/ortho/DJI_0726.tif'
path_to_log = 'C:/Users/aman3/Documents/GradSchool/testing/data/imageData.csv'
path_to_slope = 'C:/Users/aman3/Documents/GradSchool/testing/50mslope.tif'
path_to_aspect = 'C:/Users/aman3/Documents/GradSchool/testing/50maspect.tif'

#open orthophoto and extract information
ortho = gdal.Open(ortho_dir, gdalconst.GA_ReadOnly)
ortho_proj = ortho.GetProjection()
ortho_geotransform = ortho.GetGeoTransform()
ortho_band = ortho.GetRasterBand(1)  

######################################################################################################################

#open slope raster and extract information
slope = gdal.Open(path_to_slope, gdalconst.GA_ReadOnly)
slope_proj = slope.GetProjection()
slope_geotransform = slope.GetGeoTransform()
slope_band = slope.GetRasterBand(1)  

#########################################################################################################################

#open aspect raster and extract information      
aspect = gdal.Open(path_to_aspect, gdalconst.GA_ReadOnly)
aspect_XSize = aspect.RasterXSize
aspect_YSize = aspect.RasterYSize,
aspect_RasterCount = aspect.RasterCount
aspect_proj = aspect.GetProjection()
aspect_geotransform = aspect.GetGeoTransform()
aspect_band = aspect.GetRasterBand(1)  

###########################################################################################################################

#Preform resampling on orthophoto to match extent and resolution of aspect and slope rasters

# Create Output / destination
dst_filename = 'C:/Users/aman3/Documents/GradSchool/testing/ortho/DJI_0726_warped.tif'
dst = gdal.GetDriverByName('GTiff').Create(dst_filename, int(aspect_XSize), int(aspect_YSize[0]), 1, gdalconst.GDT_Float32)
dst.SetGeoTransform( aspect_geotransform )
dst.SetProjection( aspect_proj)

# Do the work
gdal.ReprojectImage(ortho, dst, ortho_proj, aspect_proj)

del dst # Flush

##########################################################################################################################

#Preform raster calculation to apply topographic correction to orthophoto

outFile = "DJI_0726_topographic.tif"

#Open the warped orthophoto
ds1 = gdal.Open(dst_filename, gdalconst.GA_ReadOnly)
warped_ortho_band = ds1.GetRasterBand(1)

#Read the data into numpy arrays
warped_ortho_array = BandReadAsArray(warped_ortho_band)
aspect_array = BandReadAsArray(aspect_band)
slope_array = BandReadAsArray(slope_band)

#Apply correction
slope_rad = np.radians(slope_array)
aspect_rad = np.radians(aspect_array)

direct_proportion = 0.5
incoming_irradiance = 65535
solar_zenith_angle = 66
solar_azimuth_angle = 123

solar_zenith_rad = np.radians(solar_zenith_angle)
solar_azimuth_rad = np.radians(solar_azimuth_angle)

correction_factor = np.divide( np.cos( slope_rad ) * np.cos( solar_zenith_rad ) + np.multiply(np.sin( solar_zenith_rad ) * np.sin( slope_rad ), np.cos(solar_azimuth_rad - aspect_rad)), np.cos(solar_zenith_rad))

topo_correction = np.divide(warped_ortho_array, (numpy.multiply (correction_factor, direct_proportion*incoming_irradiance) + (1 - direct_proportion)*incoming_irradiance))

#Write the out file
driver = gdal.GetDriverByName("GTiff")
dsOut = driver.Create('C:/Users/aman3/Documents/GradSchool/testing/ortho/DJI_0726_topo.tif', int(aspect_XSize), int(aspect_YSize[0]), 1, gdalconst.GDT_Float32)
CopyDatasetInfo(ds1,dsOut)
bandOut = dsOut.GetRasterBand(1)
BandWriteArray(bandOut, topo_correction)

#Close the datasets
dsOut = None





