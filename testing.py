# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:37:46 2020

@author: x51b783
"""
import exifread
import piexif
from PIL import Image

from GPSPhoto import gpsphoto

def convert(s):
    try:
        return int(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)

def split(s):
    try:
        return int(s)
    except ValueError:
        num, denom = s.split('/')
        return (int(num), int(denom))



raw = 'C:/Masters/DroneAlbedoProject/Field_Data/test/DJI_0525.DNG'
tiff = 'C:/Masters/DroneAlbedoProject/Field_Data/test/DJI_0525_avg.tiff'
new_tiff = 'C:/Masters/DroneAlbedoProject/Field_Data/test/DJI_0525_avg_geo.tiff'

test_jpeg = 'C:/Masters/DroneAlbedoProject/Field_Data/test/test_jpeg.jpeg'

raw_open = open(raw, 'rb')
tags = exifread.process_file(raw_open)


Latitude = str(tags['GPS GPSLatitude'].values).strip('][').split(', ')
Latitude = ((int(Latitude[0]),1), (int(Latitude[1]),1), split(Latitude[2]))
LatitudeRef = str(tags['GPS GPSLatitudeRef'])


Longitude = str(tags['GPS GPSLongitude'].values).strip('][').split(', ')
Longitude = ((int(Longitude[0]),1), (int(Longitude[1]),1), split(Longitude[2]))
LongitudeRef = str(tags['GPS GPSLongitudeRef'])


Altitude = split(str(tags['GPS GPSAltitude']))
AltitudeRef = int(str(tags['GPS GPSAltitudeRef']))


im_tiff = Image.open(tiff)
#im_raw = Image.open(raw)
#exif_dict_raw = piexif.load(raw)
exif_dict_tiff = piexif.load(tiff)
#exif_bytes = piexif.dump(exif_dict_raw)



#exif_dict_tiff["GPS"][piexif.GPSIFD.GPSLongitude] = list(Longitude);
#exif_dict_tiff["GPS"][piexif.GPSIFD.GPSLongitudeRef] = LongitudeRef;

#exif_dict_tiff["GPS"][piexif.GPSIFD.GPSLatitude] = list(Latitude);
#exif_dict_tiff["GPS"][piexif.GPSIFD.GPSLatitudeRef] = LatitudeRef;

exif_dict_tiff["GPS"][piexif.GPSIFD.GPSAltitude] = list(Altitude);
#exif_dict_tiff["GPS"][piexif.GPSIFD.GPSAltitudeRef] = AltitudeRef;

exif_dict_tiff["0th"][piexif.ImageIFD.Artist] = "Andrew";

exif_bytes = piexif.dump(exif_dict_tiff)


test = Image.open(test_jpeg)
test.save(test_jpeg, exif=exif_bytes)

new = Image.open(test_jpeg)
exif_dict_new = piexif.load(test_jpeg)



#photo = gpsphoto.GPSPhoto(tiff)
#lat = gpsphoto.coord2decimal(Lattitude, LattitudeRef)
#long = gpsphoto.coord2decimal(Longitude, LongitudeRef)

#info = gpsphoto.GPSInfo([lat, long], int(alt))

#exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = [(120, 1), (37,1), (429996, 10000)];




