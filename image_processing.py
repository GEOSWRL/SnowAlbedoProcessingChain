# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:50:13 2020

@author: Andrew Mullen
"""


import os

import pandas as pd
import pytz
from datetime import datetime
import rawpy
import exiftool
import earthpy as ep
import imageio

wd_path = os.path.join(ep.io.HOME, 'Documents', 'SnowAlbedoProcessingChain', 'working_directory')

path_to_raw_images = os.path.join(wd_path, 'imagery', 'RAW')
path_to_tiffs = os.path.join(wd_path, 'imagery', 'TIFF')
path_to_merged_logs = os.path.join(wd_path,'logfiles', 'merged')


EXIFTOOL_PATH = os.path.join(ep.io.HOME, 'exiftool')

timezone = 'US/Mountain'

def raw_to_tiff():
    """
    Converts a RAW files to TIFFs
    Parameters
    ----------
    raw_image : string
        .DNG image filename e.g. ("image01.DNG")
    
    Returns
    -------
    result : string
        path to .tiff file
    """

    #convert to .tiff
    for filename in os.listdir(path_to_raw_images):
            if filename.endswith(".DNG"):
                raw_image_full_path = os.path.join(path_to_raw_images, filename)
                
                out_path = os.path.join(path_to_tiffs, filename[:-3] + '.tiff')
                print(out_path)
                print("converting " + filename + " to tiff")
                with rawpy.imread(raw_image_full_path) as raw:
                    rgb = raw.postprocess(gamma=(1,1), no_auto_bright = True, output_bps=16)
                    imageio.imsave(out_path, rgb)
                    print("done")
    
    return

def read_EXIF(path_to_raw_images, path_to_output):
    """
    Reads EXIF data from DJI images and creates a text file with GPS fields for use in Agisoft

    Parameters
    ----------
    path_to_raw_images : string
        path to DJI images
        
    path_to_output : string
        path to save csv file to, must end in .csv
    
    Returns
    -------
    result : string
        path to .csv file
        
    """
    
    df = pd.DataFrame(columns = ['Image ID', 'Timestamp', 'GPSLatitude', 'GPSLongitude', 'GPSAltitude', 'UAVPitch', 'UAVRoll', 'UAVTilt', 'UAVTiltDirection', 'CameraRoll', 'CameraYaw', 'Downwelling Irradiance', 'Upwelling Irradiance', 'Albedo'])
    li = []
    for filename in os.listdir(path_to_merged_logs):   
        if filename.endswith('.csv'):
            df_temp = pd.read_csv(os.path.join(path_to_merged_logs, filename))
            li.append(df_temp)
    
    df2 = pd.concat(li, axis=0, ignore_index=True)
    print(df2.head())
   
    with exiftool.ExifTool(EXIFTOOL_PATH) as et:
        for filename in os.listdir(path_to_raw_images):
            if filename.endswith(".DNG"):

                metadata = et.get_metadata_batch([os.path.join(path_to_raw_images, filename)])[0] 

                Timestamp = metadata['EXIF:CreateDate']
                YMD = datetime.strptime(Timestamp[:10], '%Y:%m:%d')
                HMS = datetime.strptime(Timestamp[10:], ' %H:%M:%S')
                Timestamp = YMD.replace(hour = HMS.hour, minute = HMS.minute, second = HMS.second)   
                Timestamp = Timestamp.astimezone(pytz.timezone(timezone))
                
                #read in logfile
                
                
                #extract albedo value associated with image
                row = df2.loc[df2['Unnamed: 0'] == str(Timestamp)].index
                    
                if(row.values.size == 0):
                    print('no measurements associated with ' + filename)
                else:
                    downwelling = float(df2.iloc[row]['incoming (W/m^2)'].values)
                    upwelling = float(df2.iloc[row]['reflected (W/m^2)'].values)
                    albedo = float(df2.iloc[row]['albedo'].values)
                    UAV_pitch = float(df2.iloc[row]['IMU_ATTI(0):pitch'].values)
                    UAV_roll = float(df2.iloc[row]['IMU_ATTI(0):roll'].values)
                    UAV_tilt = float(df2.iloc[row]['IMU_ATTI(0):tiltInclination'].values)
                    UAV_tilt_direction = float(df2.iloc[row]['IMU_ATTI(0):tiltDirection'].values)
                GPSLongitude = metadata['Composite:GPSLongitude']
                GPSLatitude = metadata['Composite:GPSLatitude']
                GPSAltitude = metadata['Composite:GPSAltitude']
                #Pitch = metadata['MakerNotes:CameraPitch'] + 90
                
                if (metadata['MakerNotes:CameraYaw']>=0):
                    CameraYaw = metadata['MakerNotes:CameraYaw']
                else:
                    CameraYaw = 360 + metadata['MakerNotes:CameraYaw']
                
                CameraRoll = metadata['MakerNotes:CameraRoll']            
                
                df = df.append(pd.Series([filename[:-3] + 'tiff', Timestamp, GPSLatitude, GPSLongitude, GPSAltitude,
                                      UAV_pitch, UAV_roll, UAV_tilt, UAV_tilt_direction, CameraRoll, CameraYaw, downwelling, upwelling, albedo], index=df.columns), ignore_index = True)

    df.set_index('Image ID', inplace = True)
    df.to_csv(path_to_output)
                            
    return (path_to_output)

#raw_to_tiff()

#read_EXIF(path_to_raw_images, os.path.join(path_to_tiffs, 'imageData50m.csv'))

