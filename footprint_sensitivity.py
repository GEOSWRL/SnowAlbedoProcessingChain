# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:00:12 2021

@author: x51b783
"""
import os
os.environ['PROJ_LIB'] = 'C:\\Users\\x51b783\\.conda\\envs\\py37\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\x51b783\\.conda\\envs\py37\\Library\\share'
from osgeo import gdal, gdalconst
import math
import numpy as np
import topo_correction_util as tcu
import pytz
import geopandas
import pandas as pd
import angles
import seaborn as sns
from matplotlib import pyplot as plt

#Desired paths to slope and aspect rasters that are created in the "prep_rasters" function. Must end in '.tif'
path_to_dem = 'C:/Temp/0311/sfm/elevation/elevation.tif'
path_to_slope = 'C:/Temp/0311/sfm/slope/slope.tif'
path_to_aspect = 'C:/Temp/0311/sfm/aspect/aspect.tif'
path_to_coordinate_array_x = 'C:/Temp/0311/sfm/coords/3DEP_xcoords_res_UTM.tif'
path_to_coordinate_array_y = 'C:/Temp/0311/sfm/coords/3DEP_ycoords_res_UTM.tif'

#path to csv file
#csv file must contain latitude, longitude, altitude, pitch, roll, and yaw
csv_path = 'C:/Temp/footprint_sensitivity/footprint_sensitivityB.csv'

#csv field names
datetime_fname = 'datetime'
GPS_latitude_fname = 'GPS:Lat'
GPS_longitude_fname  = 'GPS:Long'
GPSAltitude_fname = 'GPS:heightMSL'
pitch_fname = 'IMU_ATTI(0):pitch:C'
roll_fname = 'IMU_ATTI(0):roll:C'
yaw_fname = 'IMU_ATTI(0):yaw:C'
tilt_fname = 'IMU_ATTI(0):tiltInclination:C'
tilt_dir_fname = 'IMU_ATTI(0):tiltDirectionBodyFrame:C'
albedo_meas_fname = 'albedo'
 
source_epsg = 'EPSG:4326' #EPSG that the point data is initially stored in
dest_epsg = 'EPSG:32612' #UTM Zone 12-N

print("loading surface models")

elev_source = gdal.Open(path_to_dem, gdal.GA_ReadOnly) # open reprojected elevation GeoTiff 
slope_source = gdal.Open(path_to_slope, gdal.GA_ReadOnly)
aspect_source = gdal.Open(path_to_aspect, gdal.GA_ReadOnly)
coordinate_x_source = gdal.Open(path_to_coordinate_array_x, gdal.GA_ReadOnly)
coordinate_y_source = gdal.Open(path_to_coordinate_array_y, gdal.GA_ReadOnly)
        
elev_arr = tcu.get_band_array(elev_source)
slope_arr = tcu.get_band_array(slope_source)
aspect_arr = tcu.get_band_array(aspect_source)
coordinate_array_x_arr = tcu.get_band_array(coordinate_x_source)
coordinate_array_y_arr = tcu.get_band_array(coordinate_y_source)

print("surface models loaded")

df = pd.read_csv(csv_path, parse_dates = True, index_col = datetime_fname)
    
if df.index.tzinfo is None:
    df.index = df.index.tz_localize('US/Mountain')
        
df.index = df.index.tz_convert(pytz.timezone('gmt'))
    

#df = run_radiative_transfer(row)

gdf = geopandas.GeoDataFrame(df, geometry = geopandas.points_from_xy(df[GPS_longitude_fname], df[GPS_latitude_fname])) # create geodataframe and specify coordinate geometry
gdf = gdf.set_crs(source_epsg) # set CRS to WGS 84
gdf = gdf.to_crs(dest_epsg) # project geometry to UTM Zone 12N

gdf['cos_avg_slope_ss'] = np.zeros(gdf.shape[0])
gdf['cos_avg_aspect_ss'] = np.zeros(gdf.shape[0])
gdf['pitch_radians'] = np.radians(gdf[pitch_fname])
gdf['roll_radians'] = np.radians(gdf[roll_fname])
gdf['yaw_radians'] = np.radians(gdf[yaw_fname])
        
def calc_terrain_parameters(elev_array, slope_array, aspect_array, coordinate_array_x, coordinate_array_y, gdf, row, index, sensor_half_FOV_rad):
   
    point_lon = row['geometry'].centroid.x # centroid.x is x coordinate of point geometry feature in UTM
    point_lat = row['geometry'].centroid.y # centroid.y is y coordinate of point geometry feature in UTM
    point_elev = row[GPSAltitude_fname]

    elev_diff = point_elev - elev_array # calculate elevation difference
    
    elev_diff[elev_diff<=0]=np.nan # points above the downward-facing sensor should be masked out as well
    
    #print(np.nanmax(elev_diff))
    #print(np.nanmin(elev_diff))

    # now that we have both coordinate arrays, we can turn them into distance arrays by subtracting the pixel coordinates from the 
    # coordinates of the measurement point. Since everything is in UTM this distance is in meters.

    dist_x = coordinate_array_y - point_lat
    dist_y = coordinate_array_x - point_lon
        
    #print(np.nanmax(dist_x))
    #print(np.nanmin(dist_x))
    #print(np.nanmax(dist_y))
    #print(np.nanmin(dist_y))
    #notice that x and y are switched here. This is because we need the coordinate system to be in the form of 
    #North = x axis, East = y axis, Up/down = z axis

    

    surface_normal = [[0],[0],[-1]]
    
    pitch = row['pitch_radians']
    roll = row['roll_radians']
    yaw = row['yaw_radians']
    
    rot_matrix = [
        [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(roll)*np.sin(pitch)-np.cos(roll)*np.sin(yaw), np.sin(roll)*np.sin(yaw)+np.cos(roll)*np.cos(yaw)*np.sin(pitch)],
        [np.cos(pitch)*np.sin(yaw), np.cos(roll)*np.cos(yaw)+np.sin(roll)*np.sin(yaw)*np.sin(pitch), np.cos(roll)*np.sin(yaw)*np.sin(pitch)-np.cos(yaw)*np.sin(roll)],
        [-1*np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(roll)*np.cos(pitch)],
        ]
    
    surface_normal = np.dot(rot_matrix, surface_normal)
    
    
    
    angle = np.arcsin(np.abs((surface_normal[0][0] * dist_x + 
                              surface_normal[1][0] * dist_y + 
                              surface_normal[2][0] * -1 * elev_diff) /
                             (np.sqrt(np.square(dist_x)+np.square(dist_y)+np.square(elev_diff)) *
                              np.sqrt(np.square(surface_normal[0][0])+np.square(surface_normal[1][0])+np.square(surface_normal[2][0])
                                      ))
                             ))

    angle[angle<=(math.pi/2)-sensor_half_FOV_rad]=np.nan
    
    #note that the maximum angle should never greater than 90 degrees (1.5708 rad)
    #and the minimum angle should never be less than 90-HFOV (.349066 rad for HFOV = 70 degrees)

    cosine_incidence = np.cos((math.pi/2)-angle)
    cos_sum = np.nansum(cosine_incidence)
    weighting = cosine_incidence/cos_sum

    
    # calculate cosine wighted average
    aspect_arr_weighted = weighting * aspect_array
    weighted_aspect = np.nansum(aspect_arr_weighted)
    #print('cosine-weighted mean aspect: ' + str(weighted_aspect))
    
    #cos_avg_aspect = cos_avg_aspect.append(weighted_aspect)
    '''
    print('aspect: ' + str(weighted_aspect))
    print('tilt_dir: ' + str(row['tilt_dir']))
    '''
    
    # calculate cosine wighted average
    slope_arr_weighted = weighting * slope_array
    weighted_slope = np.nansum(slope_arr_weighted)
    #print('cosine-weighted mean slope: ' + str(weighted_slope))
    return weighted_slope, weighted_aspect

#set sensor specifications
sensor_bandwidth = [0.31, 2.7] #in micrometers
sensor_half_FOV = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85] #in degrees
sensor_half_FOV_rad = np.radians(sensor_half_FOV)
slope = np.array([])
aspect = np.array([])

df = pd.DataFrame()
df['half_FOV'] = sensor_half_FOV
for half_FOV in sensor_half_FOV_rad: 
    print('Half FOV: ' + str(np.degrees(half_FOV)) + " degrees")
    for index, row in gdf.iterrows():
        print('----------------------row-------------------------')
        ss_slope, ss_aspect = calc_terrain_parameters(elev_arr, slope_arr, aspect_arr, coordinate_array_x_arr, coordinate_array_y_arr, gdf, row, index, half_FOV)
        print('slope: ' + str(ss_slope) + ', aspect: ' + str(ss_aspect))
        slope = np.append(slope, ss_slope)
        aspect = np.append(aspect,ss_aspect)
        
df['slope'] = slope
df['aspect'] = aspect

a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)

sns.scatterplot(y=df['slope'], x=df['half_FOV'], s=200, palette = 'hls')
#sns.scatterplot(y=df['corrected_albedo_ss'], x=df.index, s=200, palette = 'hls')
   
elev_source = None
slope_source = None
aspect_source = None
coordinate_x_source = None
coordinate_y_source = None