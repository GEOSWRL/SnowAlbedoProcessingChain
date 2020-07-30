# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:20:59 2020

@author: aman3
"""

"""
topographic correction

"""

"""
This script utilizes py6s, a python wrapper for the 6s radiative transfer function, which is written in FORTRAN
    #pip install Py6s
    
"""
import math
import gdal
from Py6S import *
import pandas as pd
import dateutil
from pytz import timezone
#import pysolar
import os
import numpy as np
import sys
from osgeo import gdal, gdalconst
import struct
from osgeo.gdalnumeric import *

path_to_dem =  'C:/Users/aman3/Documents/GradSchool/testing/50mDEM.tif'
path_to_output = 'C:/Users/aman3/Documents/GradSchool/testing/50m'
path_to_orthophotos = 'C:/Users/aman3/Documents/GradSchool/testing/ortho/'
path_to_warped = 'C:/Users/aman3/Documents/GradSchool/testing/ortho/warped/'
path_to_corrected = 'C:/Users/aman3/Documents/GradSchool/testing/ortho/corrected/'
path_to_log = 'C:/Masters/DroneAlbedoProject/Field_Data/BART/BART20200702/processing/Fly164_imageData.csv'
path_to_slope = 'C:/Users/aman3/Documents/GradSchool/testing/50mSlope.tif'
path_to_aspect = 'C:/Users/aman3/Documents/GradSchool/testing/50mAspect.tif'

df = pd.read_csv(path_to_log)
df.set_index('Image ID', inplace = True)


def get_incidence_angle(topocentric_zenith_angle, slope, slope_orientation, topocentric_azimuth_angle):
    tza_rad = math.radians(topocentric_zenith_angle)
    slope_rad = math.radians(slope)
    so_rad = math.radians(slope_orientation)
    taa_rad = math.radians(topocentric_azimuth_angle)
    return math.degrees(math.acos(math.cos(tza_rad) * math.cos(slope_rad) + math.sin(slope_rad) * math.sin(tza_rad) * math.cos(taa_rad - so_rad)))

def process_DEM(path_to_dem, path_to_output):
    gdal.DEMProcessing(path_to_output +'slope.tif', path_to_dem, 'slope')
    gdal.DEMProcessing(path_to_output + 'aspect.tif', path_to_dem, 'aspect')

def run_radiative_transfer(df):
    
    DIP = []
    solar_zenith = []
    solar_azimuth = []
    global_downwelling = []

    for index, row in df.iterrows():
        lat = row['GPSLatitude']
        lon = row['GPSLongitude']
        alt = row['GPSAltitude']/1000
        
        dt = row['Timestamp']
        dt = dateutil.parser.parse(dt, dayfirst=True)
        dt = str(dt.astimezone(timezone('gmt')))
    
        s = SixS()
        
        s.geometry.from_time_and_location(lat, lon, dt, 0, 0)
        
        s.altitudes.set_target_custom_altitude(alt - 0.05)
        
        s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
        
        #s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)
        s.atmos_profile = AtmosProfile.FromLatitudeAndDate(lat, dt)
        
        s.wavelength = Wavelength(0.31, 2.7)
        
        s.visibility = None
        
        s.aot550 = 0.235
        
        s.run()

        DIP.append(s.outputs.percent_direct_solar_irradiance)
        solar_zenith.append(s.outputs.solar_z)
        solar_azimuth.append(s.outputs.solar_a)
        #s.outputs.int_solar_spectrum
        print(s.outputs.direct_solar_irradiance)
        print('6s downwelling irradiance: ' + str((s.outputs.direct_solar_irradiance + s.outputs.diffuse_solar_irradiance + s.outputs.environmental_irradiance)*2.39) + ', pyranometer downwelling irradiance: ' + str(row['Downwelling Irradiance']))
        global_downwelling.append((s.outputs.direct_solar_irradiance + s.outputs.diffuse_solar_irradiance + s.outputs.environmental_irradiance)*2.39)
        
    df['Direct_Irradiance_Proportion'] = DIP
    df['Solar_Zenith_Angle'] = solar_zenith
    df['Solar_Azimuth_Angle'] = solar_azimuth
    df['6s_Global_Downwelling_Irradiance'] = global_downwelling
    df.to_csv(path_to_log)

def prep_calc(filename, path_to_ortho, path_to_slope, path_to_aspect, path_to_output):
    #open orthophoto and extract information
    ortho = gdal.Open(path_to_ortho, gdalconst.GA_ReadOnly)
    ortho_proj = ortho.GetProjection() 

    #########################################################################################################################

    #open aspect raster and extract information      
    aspect = gdal.Open(path_to_aspect, gdalconst.GA_ReadOnly)
    aspect_XSize = aspect.RasterXSize
    aspect_YSize = aspect.RasterYSize
    aspect_proj = aspect.GetProjection()
    aspect_geotransform = aspect.GetGeoTransform() 

    ###########################################################################################################################

    #Preform resampling on orthophoto to match extent and resolution of aspect and slope rasters

    # Create Output / destination
    dst_filename = path_to_output + filename[:-4] + '_warped.tif'
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, aspect_XSize, aspect_YSize, 1, gdalconst.GDT_Float32)
    dst.SetGeoTransform(aspect_geotransform)
    dst.SetProjection(aspect_proj)

    # Do the work
    gdal.ReprojectImage(ortho, dst, ortho_proj, aspect_proj)
    
    del dst
    del aspect
    del ortho  # Flush
    
    return dst_filename
    
def run_correction(ortho_dir, path_to_slope, path_to_aspect, output_dir):
    df = pd.read_csv(path_to_log)
    df.set_index('Image ID', inplace = True)
    
    aspect = gdal.Open(path_to_aspect, gdalconst.GA_ReadOnly)
    aspect_XSize = aspect.RasterXSize
    aspect_YSize = aspect.RasterYSize
    aspect_band = aspect.GetRasterBand(1)
    
    slope = gdal.Open(path_to_slope, gdalconst.GA_ReadOnly)
    slope_band = slope.GetRasterBand(1)
    
    
    for filename in os.listdir(ortho_dir):
            if (filename.endswith('.tif')):
                
                warped_ortho = prep_calc(filename, ortho_dir + filename, path_to_slope, path_to_aspect, output_dir)
                
                solar_zenith_angle = df.loc[filename[:-4]+'.DNG']['Solar_Zenith_Angle']
                solar_azimuth_angle = df.loc[filename[:-4]+'.DNG']['Solar_Azimuth_Angle']
                direct_proportion = df.loc[filename[:-4]+'.DNG']['Direct_Irradiance_Proportion']
                incoming_irradiance = 65535
                
                outFile = output_dir + filename [:-4] + '_topographic.tif'

                #Open the warped orthophoto
                ds1 = gdal.Open(warped_ortho, gdalconst.GA_ReadOnly)
                warped_ortho_band = ds1.GetRasterBand(1)

                #Read the data into numpy arrays
                
                
                
                warped_ortho_array = BandReadAsArray(warped_ortho_band)
                aspect_array = BandReadAsArray(aspect_band)
                slope_array = BandReadAsArray(slope_band)

                #Apply correction
                slope_rad = np.radians(slope_array)
                aspect_rad = np.radians(aspect_array)
                

                solar_zenith_rad = np.radians(solar_zenith_angle)
                solar_azimuth_rad = np.radians(solar_azimuth_angle)

                correction_factor = np.divide( np.cos( slope_rad ) * np.cos( solar_zenith_rad ) + np.multiply(np.sin( solar_zenith_rad ) * np.sin( slope_rad ), np.cos(solar_azimuth_rad - aspect_rad)), np.cos(solar_zenith_rad))

                topo_correction = np.divide(warped_ortho_array, (np.multiply (correction_factor, direct_proportion*incoming_irradiance) + (1 - direct_proportion)*incoming_irradiance))

                topo_correction[topo_correction > 1] = 0
                topo_correction[topo_correction < 0] = 0



                #Write the out file
                driver = gdal.GetDriverByName("GTiff")
                dsOut = driver.Create(outFile, aspect_XSize, aspect_YSize, 1, gdalconst.GDT_Float32)
                CopyDatasetInfo(ds1,dsOut)
                bandOut = dsOut.GetRasterBand(1)
                BandWriteArray(bandOut, topo_correction)
                
                #Close the datasets
                dsOut = None
                
run_radiative_transfer(df)
#run_correction(path_to_orthophotos, path_to_slope, path_to_aspect, path_to_corrected)

