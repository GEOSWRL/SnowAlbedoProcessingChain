# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:50:13 2020

@author: Andrew Mullen
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

path_to_temp = 'C:/Users/aman3/Documents/GradSchool/testing/temp/'
path_to_images = 'C:/Users/aman3/Documents/GradSchool/testing/data/'
path_to_output = 'C:/Users/aman3/Documents/GradSchool/testing/data/vignette/out/'
path_to_log = 'C:/Users/aman3/Documents/GradSchool/testing/data/50m_merged.csv'
path_to_exif = 'C:/Users/aman3/Documents/GradSchool/testing/data/50m_exif.csv'
path_to_raw_vignette_images = 'C:/Users/aman3/Documents/GradSchool/testing/data/vignette/'




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
    

def LatLong_to_decimal(Latitude, LatitudeRef, Longitude, LongitudeRef):
    """
    Converts a lat/long coordinate from minutes, degrees, seconds to decimal

    Parameters
    ----------
    Latitude : string
        
    LatitudeRef : string
        
    Longitude : string
        
    LongitudeRef : string
        
    
    Returns
    -------
    result : list for coordinates in decimal degrees
        list[0] = latitude
        list[1] = longitude

    """
    LatSign = 1
    LongSign = -1
    
    if (LatitudeRef == 'S'):
        LatSign = -1
        
    if (LongitudeRef == 'E'):
        LongSign = 1
    
    Latitude = LatSign * (Latitude[0] + float(Latitude[1]) / 60 + float(Latitude[2]) / 3600)
    Longitude = LongSign * (Longitude[0] + float(Longitude[1]) / 60 + float(Longitude[2]) / 3600)
    return (Latitude, Longitude)


def split(s):
    try:
        return int(s)
    except ValueError:
        num, denom = s.split('/')
        return (float(num)/ int(denom))


def raw_to_tiff(raw_image, out_dir):
    """
    Converts a RAW file to a TIFF

    Parameters
    ----------
    raw_image : string
        .DNG image filename e.g. ("image01.DNG")
    
    Returns
    -------
    result : string
        path to .tiff file

    """
    
    raw_image_full_path = path_to_images + raw_image
    print("opening " + raw_image)
    
    out_path = out_dir + raw_image[:-3] + "tiff"
    
    
    print("converting to tiff")
    with rawpy.imread(raw_image_full_path) as raw:
        rgb = raw.postprocess(gamma=(1,1), no_auto_bright = True, output_bps=16)
        imageio.imsave(out_path, rgb)
    print("done")
    return out_path
    


    
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
    image = rasterio.open(tiff_image)
    red = image.read(1)
    green = image.read(2)
    blue = image.read(3)
    return red, green, blue
    
def avg_rgb(raw_image):
    """
    Converts a .DNG file into a TIFF where each pixel is the average of R,G, and B

    Parameters
    ----------
    raw_image : string
        .DNG image filename e.g. ("image01.DNG")
    
    Returns
    -------
    result : string
        path to RGB averaged tiff file
    """
    #convert to tiff
    tiff_image = raw_to_tiff(raw_image)
    
    #split tiff into R, G, and B tiffs
    red, green, blue = split_rgb(tiff_image)
    
    #average of all 3 bands
    avg = ((red + green + blue)/3)
    
    #multiply by correction factor
    avg_corrected = albedo_correction(raw_image, path_to_log, avg).astype('uint16')
    
    #create new tiff
    new_dataset = rasterio.open(
    path_to_output + raw_image[:-4] + '_avg.tiff',
    'w',
    driver='GTiff',
    height=avg_corrected.shape[0],
    width=avg_corrected.shape[1],
    count=1,
    dtype=avg_corrected.dtype
    )
    new_dataset.write(avg_corrected,1)
    new_dataset.close()
    
    
def albedo_correction(raw_image, path_to_log, tiff):
    #open raw image and extract timestamp
    raw = open(path_to_images + raw_image, 'rb')
    tags = exifread.process_file(raw)
    dateTime_str = (str(tags['Image DateTime']))

    #convert timestamp to correct format with timezone
    YMD = datetime.strptime(dateTime_str[:10], '%Y:%m:%d')
    HMS = datetime.strptime(dateTime_str[10:], ' %H:%M:%S')
    dateTime_obj = YMD.replace(hour = HMS.hour, minute = HMS.minute, second = HMS.second)   
    dateTime_obj = dateTime_obj.astimezone(pytz.timezone('US/Mountain'))

    #read in logfile
    df = pd.read_csv(path_to_log)

    #extract albedo value associated with image
    row = df.loc[df['key_0'] == str(dateTime_obj)].index
    
    if(row.values.size == 0):
        print('no measurements associated with ' + raw_image + ', no correction was applied')
        return tiff
    
    associated_albedo = float(df.iloc[row]['albedo'].values)
    
    #compute average pixel value in tiff
    avg_val = np.mean(tiff)
    
    #apply correction factor
    corrected_tiff = tiff*(avg_val/(associated_albedo*65536))
    
    return corrected_tiff
    
def read_EXIF(path_to_images, path_to_output):
    """
    Reads EXIF data from RAW images and creates a text file with GPS fields for use in Agisoft

    Parameters
    ----------
    path_to_images : string
        path to .DNG images
        
    path_to_output : string
        path to save csv file to
    
    Returns
    -------
    result : string
        path to .csv file
    """
    
    df = pd.DataFrame(columns = ['Image ID', 'GPSLatitude', 'GPSLongitude', 'GPSAltitude', 'Lens Model', 'Focal Length'
                         ])
    for filename in os.listdir(path_to_images):
        if filename.endswith(".DNG"):
        
            raw_open = open(path_to_images + filename, 'rb')
            tags = exifread.process_file(raw_open)


            GPSLatitude = str(tags['GPS GPSLatitude'].values).strip('][').split(', ')
            GPSLatitude = (int(GPSLatitude[0]), int(GPSLatitude[1]), split(GPSLatitude[2]))
            GPSLatitudeRef = str(tags['GPS GPSLatitudeRef'])
        

            GPSLongitude = str(tags['GPS GPSLongitude'].values).strip('][').split(', ')
            GPSLongitude = (int(GPSLongitude[0]), int(GPSLongitude[1]), split(GPSLongitude[2]))
            GPSLongitudeRef = str(tags['GPS GPSLongitudeRef'])
        
            [GPSLatitude, GPSLongitude] = LatLong_to_decimal(GPSLatitude, GPSLatitudeRef, GPSLongitude, GPSLongitudeRef)


            GPSAltitude = split(str(tags['GPS GPSAltitude']))
            #GPSAltitudeRef = int(str(tags['GPS GPSAltitudeRef']))
        
            LensModel = str(tags['EXIF LensModel'])
        
        
        
        
            #LensMake = str(tags['EXIF LensMake'])
            #LensSpecification = str(tags['EXIF LensSpecification'])
            FocalLength = str(tags['EXIF FocalLength'])
            #FocalLength35 = str(tags['EXIF FocalLengthIn35mmFilm'])
            #ShutterSpd = str(tags['EXIF ShutterSpeedValue'])
            #Aperture = str(tags['EXIF ApertureValue'])
            #ISO = str(tags['EXIF FNumber'])
        
            df = df.append(pd.Series([filename, GPSLatitude, GPSLongitude, GPSAltitude,
                       LensModel, FocalLength], index=df.columns), ignore_index = True)
            
    df.set_index('Image ID', inplace = True)
    df.to_csv(path_to_output + 'EXIF.csv')
    
    return (path_to_output + 'EXIF.csv')

def create_vignette_masks(path_to_raw_vignette_images):    
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
            #split tiff into R, G, and B tiffs
            red, green, blue = split_rgb(tiff)
            if numFiles == 0:
                r = red
                g = green
                b = blue
                        
            else:
                r += red
                g += green
                b += blue
            numFiles+=1
    
    avg_R = np.divide(r,numFiles).astype('uint16')
    avg_G = np.divide(g,numFiles).astype('uint16')
    avg_B = np.divide(b,numFiles).astype('uint16')
    
    avg_R_smoothed = ndimage.gaussian_filter(avg_R, sigma=10, order=0)
    avg_G_smoothed = ndimage.gaussian_filter(avg_G, sigma=10, order=0)
    avg_B_smoothed = ndimage.gaussian_filter(avg_B, sigma=10, order=0)
    
    create_geotiff(average_dir + 'avg_R_smoothed.tiff', avg_R_smoothed)
    create_geotiff(average_dir + 'avg_G_smoothed.tiff', avg_G_smoothed)
    create_geotiff(average_dir + 'avg_B_smoothed.tiff', avg_B_smoothed)

                            
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
              
    arr_x = distance_matrix.flatten()
    
    bands = {'R' : avg_R_smoothed, 'G' : avg_G_smoothed, 'B' : avg_B_smoothed}
    for band in bands:
       arr_y = bands.get(band).flatten()

       #compute polynomial
       print("fitting polynomial")
       z_3 = np.polyfit(arr_x, arr_y, 3)
       p_3 = np.poly1d(z_3)
       #xp = np.linspace(0, width, 100)
       #plt.plot(arr_x, arr_y, '.', label = 'Uncorrected DNs')
       #plt.plot( xp, p_3(xp), '-', label = '3rd order polynomial fit')
       #plt.show()

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
    red_mask = Image.open(mask_dir + 'avg_R_smoothed_mask.tiff')
    red_mask = np.array(red_mask)
    
    green_mask = Image.open(mask_dir + 'avg_G_smoothed_mask.tiff')
    green_mask = np.array(green_mask)
    
    blue_mask = Image.open(mask_dir + 'avg_B_smoothed_mask.tiff')
    blue_mask = np.array(blue_mask)
    
    print("correcting image")
    red, green, blue = split_rgb(path_to_tiff)
    create_geotiff('C:/Users/aman3/Documents/GradSchool/testing/data/vignette/out/DJI_0726_red.tiff', red)
    corrected_red = red/red_mask
    corrected_green = green/green_mask
    corrected_blue = blue/blue_mask
    
    return corrected_red, corrected_green, corrected_blue



red, green, blue = vignette_correction('C:/Users/aman3/Documents/GradSchool/testing/data/vignette/out/filters/','C:/Users/aman3/Documents/GradSchool/testing/data/vignette/out/DJI_0726.tiff')
create_geotiff('C:/Users/aman3/Documents/GradSchool/testing/data/vignette/out/DJI_0726_redcorrected.tiff', red)
avg_rgb_corrected = ((red + green + blue)/3)
create_geotiff('C:/Users/aman3/Documents/GradSchool/testing/data/vignette/out/DJI_0726_corrected.tiff', avg_rgb_corrected)
  
  

  
"""
shutil.rmtree(path_to_temp)
os.mkdir(path_to_temp)
for filename in os.listdir(path_to_images):
    
    if filename.endswith(".DNG"):
        avg_rgb(filename)
print("done.")
#shutil.rmtree(path_to_temp)
"""