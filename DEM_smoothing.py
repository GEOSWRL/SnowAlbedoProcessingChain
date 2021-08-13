# -*- coding: utf-8 -*-
"""
Created on Wed May 19 12:12:16 2021

@author: x51b783
"""

#
# Open a DEM, do a gaussian blur, then save it again
# Heavily based on:
#   http://gis.stackexchange.com/questions/9431/what-raster-smoothing-generalization-tools-are-available
#
# Syntax for running snippet:
#   python smoothDem.py input.tif output.tif 5
#


from scipy import exp, mgrid, signal, row_stack, column_stack, tile, misc
import numpy as np
import sys
import os
os.environ['PROJ_LIB'] = 'C:\\Users\\x51b783\\.conda\\envs\\py37\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\x51b783\\.conda\\envs\py37\\Library\\share'
from osgeo import gdal, gdalconst
import struct
from osgeo.gdalnumeric import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image


SOURCE_FILE = 'C:/Temp/bareground/elevation/YC20200805_DEM.tif'
OUTPUT_FILE = 'C:/Temp/bareground/elevation/YC20200805_DEM_smoothed.tif'
gaussian_kernel = 1



def plot(data, title):
    plot.i += 1
    print(plot.i)
    plt.subplots(2,1)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
    plt.subplots_adjust(bottom=.1, right=1, top=2)
    
plot.i = 0

def apply_smoothing(data, method = 'gaussian'): 
    
    data[data==-32767.0] = np.nan
    plt.subplot(211)
    plt.imshow(data)
    plt.title(r'Original')
    plt.gray()

    # A very simple and very narrow lowpass filter
    if method=='lowpass_3x3':
        
        kernel = np.array([[1/9, 1/9, 1/9],
                           [1/9,  1/9, 1/9],
                           [1/9, 1/9, 1/9]])
        
        lowpass_3x3 = ndimage.convolve(data, kernel)
        
        plot(lowpass_3x3, 'Simple 3x3 Highpass')
        
        return lowpass_3x3

    # A slightly "wider", but sill very simple lowpass filter 
    if method=='lowpass_5x5':
        
        kernel = np.array([[1/25, 1/25, 1/25, 1/25, 1/25],
                           [1/25, 1/25, 1/25, 1/25, 1/25],
                           [1/25, 1/25, 1/25, 1/25, 1/25],
                           [1/25, 1/25, 1/25, 1/25, 1/25],
                           [1/25, 1/25, 1/25, 1/25, 1/25]])
        
        lowpass_5x5 = ndimage.convolve(data, kernel)
        
        plot(lowpass_5x5, 'Simple 5x5 Highpass')
        
        return lowpass_5x5

    # Another way of making a highpass filter is to simply subtract a lowpass
    # filtered image from the original. Here, we'll use a simple gaussian filter
    # to "blur" (i.e. a lowpass filter) the original.
    if method=='gaussian':
        
        gaussian = ndimage.gaussian_filter(data, gaussian_kernel)
        
        #plt.subplot(212)
        #plt.imshow(gaussian)
        #plt.title(r'Gaussian Smoothing')
        #plt.gray()
        #plt.subplots_adjust(bottom=.1, right=1, top=2)
        
        return gaussian 

    
    
    
    
    
    return

def createRasterArray():
    
    # Open source file
    source = gdal.Open(SOURCE_FILE, gdalconst.GA_ReadOnly)
    gt = source.GetGeoTransform()
    proj = source.GetProjection()
    
    
    # Get rasterarray
    data = source.GetRasterBand(1).ReadAsArray()
    #blurredArray = gaussian_blur(band_array, SIZE)
    # Register driver
    driver = gdal.GetDriverByName('GTIFF')
    driver.Register()

    # Get source metadata
    cols = source.RasterXSize
    rows = source.RasterYSize
    bands = source.RasterCount
    band = source.GetRasterBand(1)
    

    # Create output image
    output = driver.Create(OUTPUT_FILE, cols, rows, bands, gdalconst.GDT_Float32)
    output.SetGeoTransform(gt)
    output.SetProjection(proj)

    # Get band from newly created image
    outBand = output.GetRasterBand(1) 
    outBand.SetNoDataValue(np.nan)
    # Write to file
    outBand.WriteArray(apply_smoothing(data), 0, 0)
    print("Your file is saved!")

    # Close all
    source = None # close raster
    output = None
    data = None
    outBand = None
    band = None

createRasterArray()