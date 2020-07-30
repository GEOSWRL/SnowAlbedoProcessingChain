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


def process_DEM(path_to_dem, path_to_slope, path_to_aspect):
    """
    Creates slope and aspect rasters given a DEM

    Parameters
    ----------
    path_to_dem : string
        path to dem raster, must be a .tif
        
    path_to_slope : string
        desired output path for slope raster, must end in .tif
    
    path_to_aspect : string
        desired output path for aspect raster, must end in .tif
    
    Returns
    -------
    result : null
        creates files at specified paths

    """
    
    gdal.DEMProcessing(path_to_slope, path_to_dem, 'slope')
    gdal.DEMProcessing(path_to_aspect + 'aspect.tif', path_to_dem, 'aspect')
    return
    
def run_radiative_transfer(path_to_log):
    """
    Runs 6s radiative transfer model to calculate solar zenith, azimuth, and proportion direct irradiance 
    given the latitude, longitude, and altitude of the UAV.

    Parameters
    ----------
    path_to_log : string
        path to image_data .csv file
    
    Returns
    -------
    result : null
        creates new data entries in image data .csv file

    """
    
    df = pd.read_csv(path_to_log)
    df.set_index('Image ID', inplace = True)
    
    DIP = [] # direct irradiance proportion
    solar_zenith = []
    solar_azimuth = []
<<<<<<< HEAD
    global_downwelling = []
=======
    et_irr = []
>>>>>>> be9843e835fc953e2048e3f4db6895c107ac2cfe

    for index, row in df.iterrows():
        lat = row['GPSLatitude']
        lon = row['GPSLongitude']
        alt = row['GPSAltitude']/1000
        
        dt = row['Timestamp']
        dt = dateutil.parser.parse(dt, dayfirst=True)
        dt_str = str(dt.astimezone(timezone('gmt')))
        
        
        s = SixS()
<<<<<<< HEAD
        
        s.geometry.from_time_and_location(lat, lon, dt, 0, 0)
        
        s.altitudes.set_target_custom_altitude(alt - 0.05)
        
=======
        s.wavelength = Wavelength(0.31, 2.7) #Kipp and Zonen Pr1 pyranometer bandwidth
        s.altitudes.set_target_custom_altitude(alt)
        s.geometry.from_time_and_location(lat, lon, dt_str, 0, 0)
>>>>>>> be9843e835fc953e2048e3f4db6895c107ac2cfe
        s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
        
        #s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)
        s.atmos_profile = AtmosProfile.FromLatitudeAndDate(lat, dt)
        
        s.wavelength = Wavelength(0.31, 2.7)
        
        s.visibility = None
        
        s.aot550 = 0.235
        
        s.run()
        
        et = pvlib.irradiance.get_extra_radiation(dt, solar_constant=1366.1, method='spencer', epoch_year=2020)
        
        et_irr.append(et)        
        DIP.append(s.outputs.percent_direct_solar_irradiance)
        solar_zenith.append(s.outputs.solar_z)
        solar_azimuth.append(s.outputs.solar_a)
<<<<<<< HEAD
        #s.outputs.int_solar_spectrum
        print(s.outputs.direct_solar_irradiance)
        print('6s downwelling irradiance: ' + str((s.outputs.direct_solar_irradiance + s.outputs.diffuse_solar_irradiance + s.outputs.environmental_irradiance)*2.39) + ', pyranometer downwelling irradiance: ' + str(row['Downwelling Irradiance']))
        global_downwelling.append((s.outputs.direct_solar_irradiance + s.outputs.diffuse_solar_irradiance + s.outputs.environmental_irradiance)*2.39)
        
=======
        
    df['Extraterrestrial_irradiance'] = et_irr
>>>>>>> be9843e835fc953e2048e3f4db6895c107ac2cfe
    df['Direct_Irradiance_Proportion'] = DIP
    df['Solar_Zenith_Angle'] = solar_zenith
    df['Solar_Azimuth_Angle'] = solar_azimuth
    df['6s_Global_Downwelling_Irradiance'] = global_downwelling
    df.to_csv(path_to_log)
    
    return 

def prep_calc(filename, path_to_ortho, path_to_slope, path_to_aspect, path_to_output):
    """
    Creates slope and aspect rasters given a DEM

    Parameters
    ----------
    path_to_dem : string
        path to dem raster, must be a .tif
        
    path_to_slope : string
        desired output path for slope raster, must end in .tif
    
    path_to_aspect : string
        desired output path for aspect raster, must end in .tif
    
    Returns
    -------
    result : null
        creates files at specified paths

    """
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
    """
    Creates slope and aspect rasters given a DEM

    Parameters
    ----------
    path_to_dem : string
        path to dem raster, must be a .tif
        
    path_to_slope : string
        desired output path for slope raster, must end in .tif
    
    path_to_aspect : string
        desired output path for aspect raster, must end in .tif
    
    Returns
    -------
    result : null
        creates files at specified paths

    """
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
                print('warping ' + filename)
                warped_ortho = prep_calc(filename, ortho_dir + filename, path_to_slope, path_to_aspect, output_dir)
                
                print('extracting values from image data file')
                #extract pertinent information from image data csv file
                solar_zenith_angle = df.loc[filename[:-4]+'.DNG']['Solar_Zenith_Angle']
                solar_azimuth_angle = df.loc[filename[:-4]+'.DNG']['Solar_Azimuth_Angle']
                direct_proportion = df.loc[filename[:-4]+'.DNG']['Direct_Irradiance_Proportion']
                extraterrestrial_irradiance = df.loc[filename[:-4]+'.DNG']['Extraterrestrial_irradiance']
                
                
                incoming_irradiance = 65535
                
                print('calculating parameters')
                
                incoming_diffuse = incoming_irradiance * (1-direct_proportion)
                
                outFile = output_dir + filename [:-4] + '_topographic.tif' #output filepath

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
                solar_altitude_rad = np.pi/2 - solar_zenith_rad
                
                
                
                incidence_angle = np.arccos(np.cos( slope_rad ) * np.cos( solar_zenith_rad ) + np.multiply(np.sin( solar_zenith_rad ) * np.sin( slope_rad ), np.cos(solar_azimuth_rad - aspect_rad)))
                
                a = np.maximum(0, np.cos(incidence_angle))
                b = np.maximum(0.087, np.cos(solar_zenith_rad))
                
                relative_oa = np.divide(1, np.sin(solar_altitude_rad) + np.power(0.15*(solar_altitude_rad + 3.885), 1.253))
                delta = np.multiply(incoming_diffuse, np.divide(relative_oa, 70000))
                F1 = np.add(np.add(0.568, np.multiply(0.187, delta)), np.multiply(-0.295, solar_zenith_rad))
                F2 = np.add(np.add(0.109, np.multiply(-.152, delta)), np.multiply(-0.014, solar_zenith_rad))
                #F2 = 0.109 + (-.152) * delta + (-0.014 *solar_zenith_rad)
                
                print('running diffuse correction')
                diffuse_corrected = np.multiply(incoming_diffuse, np.add(np.multiply((np.subtract(1,F1)), np.divide(np.add(1, np.cos(slope_rad)),2)), np.add(np.multiply(F1, np.divide(a,b)), np.multiply(F2, np.sin(slope_rad)))))
                
                correction_factor = np.divide( np.cos( slope_rad ) * np.cos( solar_zenith_rad ) + np.multiply(np.sin( solar_zenith_rad ) * np.sin( slope_rad ), np.cos(solar_azimuth_rad - aspect_rad)), np.cos(solar_zenith_rad))

                print('applying direct correction')
                topo_correction = np.divide(warped_ortho_array, (np.multiply (correction_factor, direct_proportion*incoming_irradiance) + diffuse_corrected))
                
                """
                diffuse_correction
                X = Xh[(1-F1)(1+cos(S))/2 + F1a/b +F2sinS]
                
                X = irradiance recieved
                #F1 and F2 are coefficients expressing the degree of circumsolar and horizon/zenith anisotropy
                delta = diffuse_horizontal * relative optical airmass / extraterrestrial irradiance
                relative optical airmass = 1/[sin (solar_altitude_rad) + a (solar_altitude_rad + b) ^ -c] in which a = 0.1500, b = 3.885, c = 1.253
                solar_altidtude = math.PI/2 - solar_zenith_rad
                extraterrestrial_irradiance = df.loc[filename[:-4]+'.DNG']['Extraterrestrial_irradiance']
                assuming bin 4 from Perez et al.
                F1 = 0.568 + 0.187 * delta + (-0.295 * zenith angle)
                F2 = 0.109 + (-.152) * delta + (-0.014 * zenith angle) 
                S = slope
                a = max(0, cos(incidence_angle))
                b = max(0.087, cosZ)
                z = zenith_angle
                
                
                """
                #filter erroneous results from raster matrix
                topo_correction[topo_correction > 1] = 0
                topo_correction[topo_correction < 0] = 0



                #Write the out file
                driver = gdal.GetDriverByName("GTiff")
                dsOut = driver.Create(outFile, aspect_XSize, aspect_YSize, 1, gdalconst.GDT_Float32)
                CopyDatasetInfo(ds1,dsOut)
                bandOut = dsOut.GetRasterBand(1)
                BandWriteArray(bandOut, topo_correction)
                bandOut.SetNoDataValue(0)
                
                #Close the datasets
                dsOut = None
<<<<<<< HEAD
                
run_radiative_transfer(df)
#run_correction(path_to_orthophotos, path_to_slope, path_to_aspect, path_to_corrected)
=======
                print('corrected')
#run_radiative_transfer(path_to_log)
run_correction(path_to_orthophotos, path_to_slope, path_to_aspect, path_to_corrected)
>>>>>>> be9843e835fc953e2048e3f4db6895c107ac2cfe

