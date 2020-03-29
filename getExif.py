# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 08:49:03 2020

@author: aman3
"""
import os
import exifread
import pandas as pd


path_to_images = 'C:/Users/aman3/Documents/GradSchool/testing/data/'
path_to_output = 'C:/Users/aman3/Documents/GradSchool/testing/output/'

df = pd.DataFrame(columns = ['Image ID', 'GPSLatitude', 'GPSLongitude', 'GPSAltitude', 'Lens Model', 
                         ])


def LatLong_to_decimal(Latitude, LatitudeRef, Longitude, LongitudeRef):
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
        GPSAltitudeRef = int(str(tags['GPS GPSAltitudeRef']))
        
        LensModel = str(tags['EXIF LensModel'])
        
        
        
        
        #LensMake = str(tags['EXIF LensMake'])
        #LensSpecification = str(tags['EXIF LensSpecification'])
        FocalLength = str(tags['EXIF FocalLength'])
        #FocalLength35 = str(tags['EXIF FocalLengthIn35mmFilm'])
        #ShutterSpd = str(tags['EXIF ShutterSpeedValue'])
        #Aperture = str(tags['EXIF ApertureValue'])
        #ISO = str(tags['EXIF FNumber'])
        
        df = df.append(pd.Series([filename, GPSLatitude, GPSLongitude, GPSAltitude,
                                  LensModel], index=df.columns), ignore_index = True)
        df.set_index('Image ID', inplace = True)
        df.to_csv(path_to_output + 'EXIF.csv')



