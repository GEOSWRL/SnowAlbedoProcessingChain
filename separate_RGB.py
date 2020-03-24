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

path_to_temp = 'C:/Masters/DroneAlbedoProject/Field_Data/temp/'
path_to_images = 'C:/Masters/DroneAlbedoProject/Field_Data/YC/YC20200219/imagery/DJI_0535/50m/'
path_to_output = 'C:/Masters/DroneAlbedoProject/Field_Data/YC/YC20200219/imagery/DJI_0535/50m/avg/'
path_to_log = 'C:/Masters/DroneAlbedoProject/Field_Data/YC/YC20200219/flight_logs/50m_merged.csv'



def raw_to_tiff(raw_image):
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
    
    out_path = (path_to_temp + raw_image[:-3] + "tiff")
    
    print("converting to tiff")
    with rawpy.imread(raw_image_full_path) as raw:
        rgb = raw.postprocess(gamma=(1,1), no_auto_bright = True, output_bps=16)
        imageio.imsave(out_path, rgb)
    return out_path
    

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
    raw_to_tiff(raw_image)
    
    #split tiff into R, G, and B tiffs
    image = rasterio.open(path_to_temp + raw_image[:-3] + "tiff")
    red = image.read(1)
    green = image.read(2).astype(float)
    blue = image.read(3)
    
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
    
    

    
    

shutil.rmtree(path_to_temp)
os.mkdir(path_to_temp)
for filename in os.listdir(path_to_images):
    
    if filename.endswith(".DNG"):
        avg_rgb(filename)
print("done.")
shutil.rmtree(path_to_temp)