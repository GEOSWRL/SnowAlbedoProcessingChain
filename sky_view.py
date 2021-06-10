# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:40:48 2021

sky_view.py calculates the fraction of sky that is cloud covered

The algorithm does not discriminate between cloud and snow, 
so snow covered portions must be masked out manually.

@author: x51b783
"""

import image_processing as ip
import numpy as np
from osgeo import gdal, gdalconst
from osgeo.gdalnumeric import *
import matplotlib.pyplot as plt

#ip.raw_to_tiff()

path_to_tiff = 'C:/Temp/sky_view/IMG_4991.tiff'


def open_tiff(path):
    tiff = gdal.Open(path_to_tiff, gdalconst.GA_ReadOnly)
    red_band = tiff.GetRasterBand(1)
    green_band = tiff.GetRasterBand(2)
    blue_band = tiff.GetRasterBand(3)
    return BandReadAsArray(red_band), BandReadAsArray(green_band), BandReadAsArray(blue_band)

def get_cloud_fraction(path_to_tiff, threshold = 3):
    red_band, green_band, blue_band = open_tiff(path_to_tiff)

    ratio = blue_band/green_band + blue_band/red_band
    ratio[ratio >= 1E308] = np.nan
    tree_mask  = blue_band < 4000
    cloud_mask = ratio < threshold

    ratio[tree_mask] = np.nan
    all_sky_sum_pixels = np.count_nonzero(~np.isnan(ratio))

    ratio[cloud_mask] = np.nan
    blue_sky_sum_pixels = np.count_nonzero(~np.isnan(ratio))

    cloud_fraction = 1-(blue_sky_sum_pixels / all_sky_sum_pixels)
    

    fig = plt.figure(figsize=(6, 3.2))


    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    #ratio threshold between 3 and 4?
    plt.imshow(ratio, vmax = 10)
    #plt.imshow(green_band+blue_band)


    ax.set_aspect('equal')


    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()
    
    return cloud_fraction