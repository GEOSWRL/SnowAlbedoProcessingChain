# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:50:13 2020

@author: Andrew Mullen
"""

import os
import rasterio
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import earthpy as et

wd_path = os.path.join(et.io.HOME, 'Documents', 'SnowAlbedoProcessingChain', 'working_directory')

path_to_temp = 'C:/Users/aman3/Documents/GradSchool/testing/temp/'
path_to_raw_images = 'C:/Masters/DroneAlbedoProject/Field_Data/YC/YC20200219/imagery/DJI_0535/50m/'

path_to_tiffs = 'C:/Masters/DroneAlbedoProject/Field_Data/YC/YC20200219/processing/raw_tiff/'
path_to_output = 'C:/Masters/DroneAlbedoProject/Field_Data/YC/YC20200219/processing/'
path_to_log = 'C:/Masters/DroneAlbedoProject/Field_Data/YC/YC20200219/flight_logs/50m_merged.csv'
path_to_exif = 'C:/Users/aman3/Documents/GradSchool/testing/data/50m_exif.csv'
path_to_raw_vignette_images = 'C:/Users/aman3/Documents/GradSchool/testing/data/vignette/'
vignette_mask_dir = 'C:/Masters/DroneAlbedoProject/Field_Data/utils/vignette_masks/'

def create_geotiff (out_path, dataset):
    """
    Creates new Geotiff from numpy array

    Parameters
    ----------
    out_path : string
        desired output path for file, must end in .tiff ("C:/tiffs/image01.tiff")
    
    Returns
    -------
    result : null
        creates file at specified path

    """

    new_dataset = rasterio.open(
        out_path,
        'w',
        driver='GTiff',
        height=dataset.shape[0],
        width=dataset.shape[1],
        count=1,
        dtype=dataset.dtype
        )
    new_dataset.write(dataset,1)
    new_dataset.close()
    
def split_rgb(tiff_image):
    """
    Splits a TIFF image into red, green, and blue arrays

    Parameters
    ----------
    tiff_image : string
        .TIFF image filename e.g. ("image01.TIFF")
    
    Returns
    -------
    [[float]] : red channel, green channel, and blue channel numpy arrays

    """
    print("splitting int rgb")
   
    #open image
    image = rasterio.open(tiff_image)
    
    #read bands
    red = image.read(1)
    green = image.read(2)
    blue = image.read(3)
   
    return red, green, blue
    
def avg_rgb(red, green, blue, out_dir, tiff_image):
    """
    Takes the average of red, green, and blue arrays and saves to new tiff

    Parameters
    ----------
    red : [float]
        float array of red band of .tiff image
    green : [float]
        float array of green band of .tiff image
    blue : [float]
        float array of blue band of .tiff image
    tiff_image: string
        name of .tiff image to take average of
    out_dir: string
        image output directory
    
    Returns
    -------
    result : null
        geotiff of rgb average tiff image created in out_dir
    """
    
    #path to output image
    out_path = out_dir + tiff_image
    
    #split .tiff into r,g,b bands
    red, green, blue = split_rgb(tiff_image)
    
    #average of all 3 bands
    avg = ((red + green + blue)/3).astype('uint16')
    
    #create new tiff
    create_geotiff(out_path, avg)

def create_vignette_masks(path_to_raw_vignette_images):    
    """
    Creates radial vignette masks for red, green, and blue bands of imagery

    Parameters
    ----------
    path_to_raw_vignette_images : string
        path to .tiff images to be used to generate vignette filter
        
    path_to_output : string
        path to save csv file to
    
    Returns
    -------
    result : none
        creates a vignette mask for each band in tiff image at specified output directory
    """
    out_dir = 'C:/Users/aman3/Documents/GradSchool/testing/data/vignette/out/'
    average_dir = 'C:/Users/aman3/Documents/GradSchool/testing/data/vignette/out/avg/'
    filters_dir = 'C:/Users/aman3/Documents/GradSchool/testing/data/vignette/out/filters/'
    
    r = []
    g = []
    b = []
    
    numFiles = 0
    
    """
    for filename in os.listdir(path_to_raw_vignette_images):
        if filename.endswith(".DNG"):
            raw_to_tiff(filename, out_dir)
    """      
    
    for filename in os.listdir(out_dir):
                
        if filename.endswith(".tiff"):
            print(filename)
            tiff = out_dir+filename
            red, green, blue = split_rgb(tiff) #split tiff into R, G, and B tiffs
            
            #create image sum matrix for each band
            if numFiles == 0:
                r = red
                g = green
                b = blue
                        
            else:
                r += red
                g += green
                b += blue
            numFiles+=1
    
    #create image average for each band
    avg_R = np.divide(r,numFiles).astype('uint16')
    avg_G = np.divide(g,numFiles).astype('uint16')
    avg_B = np.divide(b,numFiles).astype('uint16')
    
    #apply gaussian smoothing to each average image
    avg_R_smoothed = ndimage.gaussian_filter(avg_R, sigma=10, order=0)
    avg_G_smoothed = ndimage.gaussian_filter(avg_G, sigma=10, order=0)
    avg_B_smoothed = ndimage.gaussian_filter(avg_B, sigma=10, order=0)
    
    #create tiff of average image for each band
    create_geotiff(average_dir + 'avg_R_smoothed.tiff', avg_R_smoothed)
    create_geotiff(average_dir + 'avg_G_smoothed.tiff', avg_G_smoothed)
    create_geotiff(average_dir + 'avg_B_smoothed.tiff', avg_B_smoothed)

    #get image dimensions                      
    height = avg_R.shape[0]
    width = avg_R.shape[1]
    center_height = int(height/2)
    center_width = int(width/2)
    center_pixel = np.array([center_height, center_width])
                            
    #create distance matrix
    print("computing distance matrix")
    distance_matrix = np.zeros((height, width))
    for y in range(0, height):
        for x in range(0, width):
            pixel = np.array([y,x])
            dist = np.linalg.norm(pixel-center_pixel)
            distance_matrix[y][x] = dist
            
    #reduce dimensionality of matrices to run polynomial fit              
    arr_x = distance_matrix.flatten()
    
    bands = {'R' : avg_R_smoothed, 'G' : avg_G_smoothed, 'B' : avg_B_smoothed}
    for band in bands:
       arr_y = bands.get(band).flatten()
       #compute polynomial
       print("fitting polynomial")
       z_3 = np.polyfit(arr_x, arr_y, 3)
       p_3 = np.poly1d(z_3)

       #generate vignette mask
       print("generating vignette mask")
       vignette_mask = np.zeros((height, width))
       for y in range(0, height):
           for x in range(0, width):
               dist = distance_matrix[y][x]
               vignette_mask[y][x] = p_3(dist) / p_3(0)
        
       print("saving mask to new tiff")
       create_geotiff(filters_dir + 'avg_' + band + '_smoothed_mask.tiff', vignette_mask.astype(np.float32))



def vignette_correction(mask_dir, path_to_tiff):
    
    """
    Applies vignette masks to .tiff image

    Parameters
    ----------
    mask_dir : string
        path to directory containing vignette masks
        
    path_to_tiff : string
        path to tiff image
    
    Returns
    -------
    result : [float],[float],[float]
        red, green, and blue float arrays corrected with vignette masks
    """
    
    red_mask = Image.open(mask_dir + 'avg_R_smoothed_mask.tiff')
    red_mask = np.array(red_mask)
    
    green_mask = Image.open(mask_dir + 'avg_G_smoothed_mask.tiff')
    green_mask = np.array(green_mask)
    
    blue_mask = Image.open(mask_dir + 'avg_B_smoothed_mask.tiff')
    blue_mask = np.array(blue_mask)
    
    print("correcting image")
    red, green, blue = split_rgb(path_to_tiff)
    corrected_red = red/red_mask
    corrected_green = green/green_mask
    corrected_blue = blue/blue_mask
    
    return corrected_red, corrected_green, corrected_blue


"""
for filename in os.listdir(path_to_raw_images):  
    if filename.endswith(".DNG"):
        raw_to_tiff(filename)  
"""


"""
for filename in os.listdir(path_to_tiffs):  
    if filename.endswith(".tiff"):
        r,g,b = vignette_correction(vignette_mask_dir, path_to_tiffs+filename)
        avg_rgb(r, g, b, avg_corrected_dir, filename)  
"""
