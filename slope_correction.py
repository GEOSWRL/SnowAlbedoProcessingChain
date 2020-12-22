# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:20:59 2020

@author: aman3
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
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import cm as CM
import earthpy as et


wd_path = os.path.join(et.io.HOME, 'Documents', 'SnowAlbedoProcessingChain', 'working_directory')

print(et.io.HOME)
path_to_dem =  os.path.join(wd_path, 'surface_models', 'elevation', os.listdir(os.path.join(wd_path, 'surface_models', 'elevation'))[0])
path_to_output = et.io.HOME
path_to_orthophotos = et.io.HOME
path_to_tiffs = os.path.join(wd_path, 'imagery', 'TIFF')
#path_to_warped = 'C:/Users/aman3/Documents/GradSchool/testing/ortho/warped/'
#path_to_corrected = 'C:/Users/aman3/Documents/GradSchool/testing/ortho/corrected/'
path_to_log = os.path.join(path_to_tiffs, 'imageData50m.csv')
#path_to_slope = et.io.HOME
#path_to_aspect = et.io.HOME


df = pd.read_csv(path_to_log)
df.set_index('Image ID', inplace = True)


def get_incidence_angle(topocentric_zenith_angle, slope, slope_orientation, topocentric_azimuth_angle):
    tza_rad = math.radians(topocentric_zenith_angle)
    slope_rad = math.radians(slope)
    so_rad = math.radians(slope_orientation)
    taa_rad = math.radians(topocentric_azimuth_angle)
    return math.degrees(math.acos(math.cos(tza_rad) * math.cos(slope_rad) + math.sin(slope_rad) * math.sin(tza_rad) * math.cos(taa_rad - so_rad)))

def process_DEM(path_to_dem, path_to_slope = None, path_to_aspect = None):
    #reproject DEM to UTM
    
    processing_options = gdal.DEMProcessingOptions(alg = 'ZevenbergenThorne', slopeFormat = 'degree')
    
    if path_to_slope == None and path_to_aspect == None:
        return
    
    elif path_to_slope != None and path_to_aspect == None:
        gdal.DEMProcessing(path_to_slope, path_to_dem, 'slope', options = processing_options)
        return
    
    elif path_to_slope == None and path_to_aspect != None:
        gdal.DEMProcessing(path_to_aspect, path_to_dem, 'aspect', options = processing_options)
        return
    else:   
        gdal.DEMProcessing(path_to_slope, path_to_dem, 'slope', options = processing_options)
        gdal.DEMProcessing(path_to_aspect, path_to_dem, 'slope', options = processing_options)
        return

def run_radiative_transfer(df, GPS_latitude_fname, GPS_longitude_fname, GPSAltitude_fname, datetime_fname, start_wavelength, end_wavelength):
    #.31, 2.7 for PR1 pyranometers
    bandwidth = end_wavelength-start_wavelength
    DIP = []
    solar_zenith = []
    solar_azimuth = []
    global_downwelling = []
    et_irr = []

    for index, row in df.iterrows():
        
        #gather row data from flight log entry
        lat = row[GPS_latitude_fname]
        lon = row[GPS_longitude_fname]
        alt = row[GPSAltitude_fname]/1000
        dt = index
        #dt = dateutil.parser.parse(dt, dayfirst=True)
        dt = str(dt.astimezone(timezone('gmt')))
    
        #initiate 6s, set parameters, and run
        s = SixS()
        
        
        s.wavelength = Wavelength(start_wavelength, end_wavelength)
        s.altitudes.set_target_custom_altitude(alt - 0.05)
        s.geometry.from_time_and_location(lat, lon, dt, 0, 0)
        s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
        s.atmos_profile = AtmosProfile.FromLatitudeAndDate(lat, dt)
        s.visibility = None
        s.aot550 = 0.235
        
        s.run()

        #append radiative transfer results to corresponding storage arrays
           
        global_downwelling.append((s.outputs.direct_solar_irradiance + s.outputs.diffuse_solar_irradiance + s.outputs.environmental_irradiance)*bandwidth)
        DIP.append(s.outputs.percent_direct_solar_irradiance)
        solar_zenith.append(s.outputs.solar_z)
        solar_azimuth.append(s.outputs.solar_a)
        et_irr.append(s.outputs.int_solar_spectrum)
        
    #create new dataframe columns from radiative transfer storage arrays
    df['6s_Extraterrestrial_irradiance'] = et_irr
    df['6s_Direct_Irradiance_Proportion'] = DIP
    df['6s_Solar_Zenith_Angle'] = solar_zenith
    df['6s_Solar_Azimuth_Angle'] = solar_azimuth
    df['6s_modeled_global_irradiance'] = global_downwelling
    
    return df

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

def init_C_correction(incidence_flat, ortho_flat):
    print('calculating empirical coefficients')
    #incidence_angle_tiff = gdal.Open(path_to_aspect, gdalconst.GA_ReadOnly)
    #incidence_XSize = incidence_angle_tiff.RasterXSize
    #incidence_YSize = incidence_angle_tiff.RasterYSize
    #incidence_angle_arr = incidence_angle_tiff.GetRasterBand(1)
    #incidence_XSize = np.shape(incidence_angle_arr)[1]
    #incidence_YSize = np.shape(incidence_angle_arr)[0]
    
    #print("incidence_xsize: ", incidence_XSize, ", incidence_ysize: ", incidence_YSize)
    #ortho_tiff = gdal.Open(path_to_aspect, gdalconst.GA_ReadOnly)
    #ortho_XSize = ortho_tiff.RasterXSize
    #ortho_YSize = ortho_tiff.RasterYSize
    #ortho_arr = ortho_tiff.GetRasterBand(1)
    #ortho_XSize = np.shape(ortho_arr)[1]
    #ortho_YSize = np.shape(ortho_arr)[0]
    #print("ortho_xsize: ", str(ortho_XSize), ", ortho_ysize: ", str(ortho_YSize))

    #if(incidence_XSize == ortho_XSize and incidence_YSize == ortho_YSize):
     #   print("file extents match, running regression")
    
    

    print("fitting linear model")
    model = LinearRegression().fit(incidence_flat, ortho_flat)



    intercept = model.intercept_
    slope = model.coef_[0]
    r_sq = model.score(incidence_flat, ortho_flat)


    print('intercept:', intercept)
    print('slope:', slope)
    print('coefficient of determination:', r_sq)

    x = np.arange(0.3, 0.8, 0.1)
    y = slope*x + intercept


    ################################
    
    fig, ax = plt.subplots()
    ax.plot(x, y, c = '#782020')
    
    
    ax.scatter(incidence_flat, ortho_flat, c = '#0a0303', alpha=0.01, s = 0.05)

    ax.set(xlabel='Cosine of Incidence Angle', ylabel='Pixel DN',
           title='Incidence Angle vs Pixel DN')
    ax.grid()
    
    plt.show()
    #################################
    
    #gridsize=30
    #plt.subplot(111)

    # if 'bins=None', then color of each hexagon corresponds directly to its count
    # 'C' is optional--it maps values to x-y coordinates; if 'C' is None (default) then 
    # the result is a pure 2D histogram 
    """
    plt.hexbin(incidence_flat, ortho_flat, gridsize=gridsize, cmap=CM.jet, bins=None)
    plt.axis([incidence_flat.min(), incidence_flat.max(), ortho_flat.min(), ortho_flat.max()])

    cb = plt.colorbar()
    cb.set_label('hhhhhhh') 
    
    plt.show()
    """
    #plt.savefig(path_to_output + 'Incidence_vs_DN.png')
    return slope, intercept






    
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
                
                print('warping ' + filename)
                warped_ortho = prep_calc(filename, ortho_dir + filename, path_to_slope, path_to_aspect, output_dir)
                
                print('extracting values from image data file')
                #extract pertinent information from image data csv file
                #solar_zenith_angle = df.loc[filename[:-4]+'.DNG']['Solar_Zenith_Angle']
                #solar_azimuth_angle = df.loc[filename[:-4]+'.DNG']['Solar_Azimuth_Angle']
                #direct_proportion = df.loc[filename[:-4]+'.DNG']['Direct_Irradiance_Proportion']
                
                solar_zenith_angle = 65.14814815
                solar_azimuth_angle = 139.1407407
                direct_proportion  = 0.684

                #extraterrestrial_irradiance = df.loc[filename[:-4]+'.DNG']['Extraterrestrial_irradiance']
                
                
                #incoming_irradiance = 50000
                #incoming_diffuse = incoming_irradiance * (1-direct_proportion)
                
                
                print('calculating parameters')
                
                
                
                outFile = output_dir + filename [:-4] + '_topographic.tif' #output filepath
                incidence_outFile = output_dir + filename [:-4] + '_incidence.tif'
                
                
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
                
                #solar_altitude_rad = np.pi/2 - solar_zenith_rad
                
                
                incidence_angle = np.arccos(np.cos( slope_rad ) * np.cos( solar_zenith_rad ) + np.multiply(np.sin( solar_zenith_rad ) * np.sin( slope_rad ), np.cos(solar_azimuth_rad - aspect_rad)))
                
                """
                print(np.degrees(incidence_angle[0]))
                print(incidence_angle.shape)
                #Write the out file for incidence angle
                driver = gdal.GetDriverByName("GTiff")
                incidenceOut = driver.Create(incidence_outFile, aspect_XSize, aspect_YSize, 1, gdalconst.GDT_Float32)
                CopyDatasetInfo(aspect,incidenceOut)
                bandOut = incidenceOut.GetRasterBand(1)
                BandWriteArray(bandOut, np.degrees(incidence_angle))
                bandOut.SetNoDataValue(0)
                incidenceOut = None
                """                

                print('applying direct correction')
                
                ####No Corrrection
                topo_correction= warped_ortho_array
                
                
                ####Cosine correction
                #correction_factor = np.divide( np.cos(incidence_angle), np.cos(solar_zenith_rad))
                #topo_correction = np.divide(warped_ortho_array, (np.multiply (correction_factor, direct_proportion*incoming_irradiance) + ((1-direct_proportion)*incoming_irradiance)))
                #topo_correction = np.divide(warped_ortho_array, (np.multiply (correction_factor, incoming_irradiance)))
                
                #cosine for optical dn values, not albedo
                #correction_factor = np.divide(np.cos(solar_zenith_rad), np.cos(incidence_angle))
                #topo_correction = np.multiply (correction_factor, np.add(np.multiply(direct_proportion,warped_ortho_array), (np.multiply((1-direct_proportion),warped_ortho_array))))
                
                
                ####C-corrrection
                
                incidence_flat = incidence_angle.flatten()
                incidence_flat = np.cos(incidence_flat)
    
                ortho_flat = warped_ortho_array.flatten()
                
                ortho_bad_index = np.where(ortho_flat < 12000)[0]
                incidence_bad_index = np.where(incidence_flat > 0.7)[0]
                drop_index = np.unique(np.concatenate((ortho_bad_index, incidence_bad_index)))
                
                ortho_flat = np.delete(ortho_flat, drop_index)
                incidence_flat = np.delete(incidence_flat, drop_index).reshape((-1,1))
                
                m, b = init_C_correction(incidence_flat, ortho_flat)
                c = b/m
                #correction_factor = np.divide( np.cos( slope_rad ) * np.cos( solar_zenith_rad ) + np.multiply( np.sin( solar_zenith_rad ) * np.sin( slope_rad), np.cos(solar_azimuth_rad - aspect_rad)) + c, np.cos(solar_zenith_rad) + c)
                #topo_correction = np.divide(warped_ortho_array, (np.multiply (correction_factor, incoming_irradiance)))
                
                #c-correction for optical dn values, not albedo
                correction_factor = np.divide(np.cos(solar_zenith_rad) + c, np.cos(incidence_angle) + c)
                topo_correction = np.multiply(warped_ortho_array, correction_factor)
                
                
    
                topo_flat = topo_correction.flatten()
                topo_flat = np.delete(topo_flat, drop_index)
               
                """
                """
                plt.scatter(incidence_flat, topo_flat, c = '#0a0303', alpha=0.01, s = 0.05)
                plt.xlabel('Cosine of Incidence Angle')
                plt.ylabel('Pixel DN')
                plt.title('Incidence Angle vs C-corrected Pixel DN')
               
                plt.grid()
                
                plt.show()
                
                
                
                #filter erroneous results from raster matrix
                #topo_correction[topo_correction > 1] = 0
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
process_DEM(path_to_dem, path_to_slope = os.path.join(wd_path, 'surface_models', 'slope', 'slope.tiff'), path_to_aspect=(os.path.join(wd_path, 'surface_models', 'aspect', 'aspect.tiff')))         
#run_radiative_transfer(df)
#run_radiative_transfer(df, 'GPSLatitude', 'GPSLongitude', 'GPSAltitude', 'Timestamp')


#run_correction(path_to_orthophotos, path_to_slope, path_to_aspect, path_to_output)

