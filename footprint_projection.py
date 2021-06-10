# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 12:47:10 2021

@author: x51b783
"""

import os
import matplotlib.pyplot as plt
import numpy as np
#import geopandas
import pandas as pd
from osgeo import gdal, gdalconst, osr
import math
from mpl_toolkits.mplot3d import Axes3D
import pytz
from Py6S import *
import topo_correction_util as tcu
import angles
from scipy import ndimage
import pyproj
os.environ['PROJ_LIB'] = 'C:\\Users\\x51b783\\.conda\\envs\\gdal\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\x51b783\\.conda\\envs\gdal\\Library\\share'
################################################ Set Parameters ################################################

#path to DEM
elev_path = 'C:/Temp/3DEP/3DEP_clipped.tif'
temp_elev = 'C:/Temp/temp_elev.tif'
usgs_slope = ''
usgs_aspect = ''
usgs_xcoords = ''
usgs_ycoords = ''
#Desired path to UTM projected DEM that is created in the "prep_rasters" function. Must end in '.tif'
elev_UTM_path = 'C:/Temp/0311/sfm/elevation/elevation.tif'

#Desired paths to slope and aspect rasters that are created in the "prep_rasters" function. Must end in '.tif'

path_to_slope = 'C:/Temp/0311/sfm/slope/slope.tif'
path_to_aspect = 'C:/Temp/0311/sfm/aspect/aspect.tif'
path_to_coordinate_array_x = 'C:/Temp/0311/sfm/coords/xcoords.tif'
path_to_coordinate_array_y = 'C:/Temp/0311/sfm/coords/ycoords.tif'


#path to csv file
#csv file must contain latitude, longitude, altitude, pitch, roll, and yaw
csv_path = 'C:/Temp/IMU/test/3-11-2021 11-48-35 AM.csv'


#csv field names
datetime_fname = 'datetime'
GPS_latitude_fname = 'latitude'
GPS_longitude_fname  = 'longitude'
GPSAltitude_fname = 'elevation'
pitch_fname = 'AngleY：'
roll_fname = 'AngleX：'
yaw_fname = 'AngleZ：'
tilt_fname = 'tilt'
tilt_dir_fname = 'tilt_dir'
albedo_meas_fname = 'albedo'
 
source_epsg = 'EPSG:4326' #EPSG that the point data is initially stored in
dest_epsg = 'EPSG:32612' #UTM Zone 12-N

wgs_epsg = 4326
utm_epsg = 32612

#set sensor specifications
sensor_bandwidth = [0.31, 2.7] #in micrometers
sensor_half_FOV = 85 #in degrees
sensor_half_FOV_rad = np.radians(sensor_half_FOV)
geotransform = []

ss_elev = gdal.Open(elev_UTM_path, gdal.GA_ReadOnly)
geotransform = ss_elev.GetGeoTransform()

ss_elev = None

#enable or disable specific processes
prepare_rasters = False

prepare_point_data = True
run_radiative_transfer = True
calculate_terrain_parameters = True
run_topo_correction = True

#open up landsat 8 file and read as array
#ls8 = gdal.Open('C:/Temp/0311/ls8/LC08_038029_20210311.tif', gdal.GA_ReadOnly)
#ls8_array = tcu.get_band_array(ls8)
#ls8 = None


################################################### Code Body ##################################################

def radiative_transfer(gdf, row, index):
    print('running 6s radiative transfer')
    #.31, 2.7 for PR1 pyranometers
    bandwidth = sensor_bandwidth[1]-sensor_bandwidth[0]
    
    #print("running 6s radiative transfer for measurement taken at " + str(index))
    #gather row data from flight log entry
    lat = row[GPS_latitude_fname]
    lon = row[GPS_longitude_fname]
    alt_km = row[GPSAltitude_fname]/1000
    dt = str(index)
        
    #initiate 6s
    s = SixS()
    
    #set 6s parameters
    s.wavelength = Wavelength(sensor_bandwidth[0], sensor_bandwidth[1])
    s.altitudes.set_target_custom_altitude(alt_km)
    s.geometry.from_time_and_location(lat, lon, dt, 0, 0)
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    s.atmos_profile = AtmosProfile.FromLatitudeAndDate(lat, dt)
    #s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeWinter)
    #s.visibility = None
    #s.aot550 = 0.1
        
    #run 6s
    s.run()

    #get radiative transfer outputs
    global_irradiance = (s.outputs.direct_solar_irradiance + s.outputs.diffuse_solar_irradiance + s.outputs.environmental_irradiance)*bandwidth
    p_dir = (s.outputs.percent_direct_solar_irradiance) #direct proportion of irradiance
    p_diff = 1-p_dir #diffuse proportion of irradiance
    solar_zenith = (s.outputs.solar_z)
    solar_azimuth = (s.outputs.solar_a)  
    print('6s_global: ' + str(global_irradiance))
    gdf.loc[index, '6s_modeled_global_irradiance'] = global_irradiance
    gdf.loc[index, '6s_Direct_Irradiance_Proportion'] = p_dir
    gdf.loc[index, '6s_Diffuse_Irradiance_Proportion'] = p_diff
    gdf.loc[index, '6s_Solar_Zenith_Angle'] = solar_zenith
    gdf.loc[index, '6s_Solar_Azimuth_Angle'] = solar_azimuth
        
    return gdf

def make_coordinate_array(geotransform, image_shape):
    print(geotransform)
    coordinate_array_x = np.zeros(image_shape)
    coordinate_array_y = np.zeros(image_shape)

    pixel_y = geotransform[3]
    pixel_x = geotransform[0]

    for row in range(0, image_shape[0]):
    
        pixel_x = geotransform[0]
        pixel_y += geotransform[5]
    
        for col in range(0, image_shape[1]):
            pixel_x += geotransform[1]
        
            coordinate_array_x[row][col] = pixel_x
            coordinate_array_y[row][col] = pixel_y
            
    tcu.write_geotiff(path_to_coordinate_array_x, image_shape, geotransform, 32612, coordinate_array_x)
    tcu.write_geotiff(path_to_coordinate_array_y, image_shape, geotransform, 32612, coordinate_array_y)
    
    return coordinate_array_x, coordinate_array_y
    
def prep_rasters():
    
    print('preparing raster data')
    
    #elev = gdal.Open(elev_path, gdal.GA_ReadOnly) # open GeoTiff in read only mode
    #warp = gdal.Warp(elev_UTM_path,elev,dstSRS = dest_epsg) # reproject elevation GeoTiff from WGS to UTM

    # close files
    #warp = None
    #elev = None
    
    elev_UTM = gdal.Open(elev_UTM_path, gdal.GA_ReadOnly) # open reprojected elevation GeoTiff 
    
    #use gdal DEM processing to create slope and aspect rasters
    #processing_options = gdal.DEMProcessingOptions(alg = 'ZevenbergenThorne', format = 'GTiff')
    #slope_UTM = gdal.DEMProcessing(path_to_slope, elev_UTM_path, 'slope')
    #aspect_UTM = gdal.DEMProcessing(path_to_aspect, elev_UTM_path, 'aspect')
    slope_UTM = gdal.Open(path_to_slope, gdal.GA_ReadOnly)
    aspect_UTM = gdal.Open(path_to_aspect, gdal.GA_ReadOnly)
    
    
    #represent the rasters as numpy arrays
    elev_array = tcu.get_band_array(elev_UTM)
    slope_array = tcu.get_band_array(slope_UTM)
    aspect_array = tcu.get_band_array(aspect_UTM)
    
    # the raster geotransform tells the precise x,y location of the upper left corner of the upper left pixel, as well as pixel size
    geotransform = elev_UTM.GetGeoTransform()
    image_shape = np.shape(elev_array)
    
    
    coordinate_array_x, coordinate_array_y = make_coordinate_array(geotransform, image_shape)
    
    
    elev_UTM = None
    slope_UTM = None
    aspect_UTM = None
    
    #return rasters as arrays
    return elev_array, slope_array, aspect_array, coordinate_array_x, coordinate_array_y

def convert_coordinates(lon, lat):
    # Define the Rijksdriehoek projection system (EPSG 28992)
    wgs84=pyproj.CRS(source_epsg) # LatLon with WGS84 datum used by GPS units and Google Earth 
    utm_12N=pyproj.CRS(dest_epsg) # UK Ordnance Survey, 1936 datum 

    lon, lat = pyproj.transform(wgs84, utm_12N, lat, lon)
    
    return lon, lat

def prep_point_data():
    
    print('preparing point data')
    
    gdf = pd.read_csv(csv_path, parse_dates = True, index_col = datetime_fname)
    
    if gdf.index.tzinfo is None:
        gdf.index = gdf.index.tz_localize('US/Mountain')
        
    gdf.index = gdf.index.tz_convert(pytz.timezone('gmt'))
    
    
    #df = run_radiative_transfer(row)
    projected_lon, projected_lat = convert_coordinates(gdf[GPS_longitude_fname], gdf[GPS_latitude_fname])
    gdf['lon_utm'] = projected_lon
    gdf['lat_utm'] = projected_lat
    

    gdf['pitch_radians'] = np.radians(gdf[pitch_fname])
    gdf['roll_radians'] = np.radians(gdf[roll_fname])
    gdf['yaw_radians'] = np.radians(gdf[yaw_fname])

    if(calculate_terrain_parameters==True):
        #gdf['mean_slope'] = np.zeros(gdf.shape[0])
        #gdf['mean_aspect'] = np.zeros(gdf.shape[0])
        
        gdf['cos_avg_slope_ss'] = np.zeros(gdf.shape[0])
        gdf['cos_avg_aspect_ss'] = np.zeros(gdf.shape[0])
        
        '''
        gdf['cos_avg_slope_bg'] = np.zeros(gdf.shape[0])
        gdf['cos_avg_aspect_bg'] = np.zeros(gdf.shape[0])
        gdf['cos_avg_slope_usgs'] = np.zeros(gdf.shape[0])
        gdf['cos_avg_aspect_usgs'] = np.zeros(gdf.shape[0])
        '''
    #create new dataframe columns from radiative transfer storage arrays
    if(run_radiative_transfer==True):
        gdf['6s_Direct_Irradiance_Proportion'] = np.zeros(gdf.shape[0])
        gdf['6s_Diffuse_Irradiance_Proportion'] = np.zeros(gdf.shape[0])
        gdf['6s_Solar_Zenith_Angle'] = np.zeros(gdf.shape[0])
        gdf['6s_Solar_Azimuth_Angle'] = np.zeros(gdf.shape[0])
        gdf['6s_modeled_global_irradiance'] = np.zeros(gdf.shape[0])
        
    if(run_topo_correction):
        gdf['corrected_albedo_ss'] = np.zeros(gdf.shape[0]) #cosine corrected
        gdf['cos_avg_ls8_ss'] = np.zeros(gdf.shape[0])
        #gdf['arithmetic_avg_corrected_albedo'] = np.zeros(gdf.shape[0])
        
        '''
        gdf['corrected_albedo_bg'] = np.zeros(gdf.shape[0]) #cosine corrected
        gdf['cos_avg_ls8_bg'] = np.zeros(gdf.shape[0])
        
        gdf['corrected_albedo_usgs'] = np.zeros(gdf.shape[0]) #cosine corrected
        gdf['cos_avg_ls8_usgs'] = np.zeros(gdf.shape[0])
        '''
        
        #gdf['sensor_incidence_angle'] = np.zeros(gdf.shape[0])
        #gdf['slope_incidence_angle'] = np.zeros(gdf.shape[0])
    
    
    
    return gdf
    
def run_viewshed(elev_band, point_lon, point_lat):
    viewshed = gdal.ViewshedGenerate(elev_band, 'GTiff', 
                                 'C:/Temp/3DEP/viewshed_test.tif',
                                 creationOptions=None,
                                 observerX=point_lon, 
                                 observerY=point_lat, 
                                 observerHeight=1.5, 
                                 targetHeight = 0,
                                 visibleVal=1,
                                 invisibleVal=0,
                                 outOfRangeVal=0,
                                 noDataVal=np.nan,
                                 dfCurvCoeff=1,
                                 mode=1,
                                 maxDistance=10000000000)

    #viewshed_arr = viewshed.ReadAsArray()
    
    #viewshed = None
    
    return viewshed

def reduce_array_extent(array, geotransform):
    ulx = geotransform[0]
    uly = geotransform[3]
    
    xres = geotransform[1]
    yres = geotransform[5]

    #return indices of non_null pixels
    arr_non_null= np.argwhere(array>0)

    #split in to 2 arrays of y coords, xcoords
    coords = np.hsplit(arr_non_null,2)
    
    #find max and min coordinates to form new bounds
    ymax=np.max(coords[0])
    ymin=np.min(coords[0])

    xmax=np.max(coords[1])
    xmin=np.min(coords[1])

    clipped_arr = array[ymin:ymax+1, xmin:xmax+1]

    new_ulx = ulx + xres*xmin
    new_uly = uly + yres*ymin
    
    new_geotransform = [new_ulx, xres, 0, new_uly, 0, yres]
    
    return clipped_arr, new_geotransform

def calc_terrain_parameters(elev_array, slope_array, aspect_array, coordinate_array_x, coordinate_array_y, gdf, row, index):
    
    
    
    print('calculating terrain parameters')
        
    #point_lon = row['geometry'].centroid.x # centroid.x is x coordinate of point geometry feature in UTM
    #point_lat = row['geometry'].centroid.y # centroid.y is y coordinate of point geometry feature in UTM
    point_lon = row['lon_utm']
    point_lat = row['lat_utm']
    
    point_elev = row[GPSAltitude_fname]
    
    
    usgs_gt = usgs_source.GetGeoTransform()
    usgs_proj = usgs_source.GetProjection()
    usgs_cols = usgs_source.RasterXSize
    usgs_rows = usgs_source.RasterYSize
    
    
    minX = usgs_gt[0]
    maxX = usgs_gt[0] + (usgs_cols * usgs_gt[1])

    minY = usgs_gt[3] + (usgs_rows * usgs_gt[5])
    maxY = usgs_gt[3]
    
    viewshed = run_viewshed(elev_utm_band, point_lon, point_lat)
    
    #resample viewshed array to original resolution
    warp_options = gdal.WarpOptions(format = 'GTiff', outputBounds = [minX, minY, maxX, maxY], 
                                width = new_cols, height = new_rows,
                                srcSRS = utm_epsg, dstSRS = utm_epsg)
    resampled_viewshed = gdal.Warp(resampled_viewshed_path, viewshed, options = warp_options)
    
    
    not_visible = np.where(viewshed_arr_resampled==0)
    
    temp_elev_viewshed = None
    reduced_elev_band = None
    elev_array[not_visible] = np.nan
    
    
    
    elev_diff = point_elev - elev_array # calculate elevation difference
    elev_diff[elev_diff<=0]=np.nan # points above the downward-facing sensor should be masked out as well
    
    
    #print(np.nanmax(elev_diff))
    #print(np.nanmin(elev_diff))

    # now that we have both coordinate arrays, we can turn them into distance arrays by subtracting the pixel coordinates from the 
    # coordinates of the measurement point. Since everything is in UTM this distance is in meters.

    dist_y = coordinate_array_y - point_lat
    #tcu.write_geotiff('C:/Temp/IMU/distx.tif', np.shape(dist_x), geotransform, 32612, dist_x)
    
    dist_x = coordinate_array_x - point_lon
    
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

    '''
    rot_matrix = [
        [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(roll)*np.sin(pitch)-np.cos(roll)*np.sin(yaw), np.sin(roll)*np.sin(yaw)+np.cos(roll)*np.cos(yaw)*np.sin(pitch)],
        [np.cos(pitch)*np.sin(yaw), np.cos(roll)*np.cos(yaw)+np.sin(roll)*np.sin(yaw)*np.sin(pitch), np.cos(roll)*np.sin(yaw)*np.sin(pitch)-np.cos(yaw)*np.sin(roll)],
        [-1*np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(roll)*np.cos(pitch)],]
    
    surface_normal = np.dot(rot_matrix, surface_normal)
    '''
    #surface_normal = angles.rotate_normals(surface_normal, pitch, roll, yaw)
    
    
    #calculate incidence angle between radiating pixel and sensor
    angle = np.arcsin(np.abs((surface_normal[0][0] * dist_x + 
                              surface_normal[1][0] * dist_y + 
                              surface_normal[2][0] * -1 * elev_diff) /
                             (np.sqrt(np.square(dist_x)+np.square(dist_y)+np.square(elev_diff)) *
                              np.sqrt(np.square(surface_normal[0][0])+np.square(surface_normal[1][0])+np.square(surface_normal[2][0])
                                      ))
                             ))
    
    
    #plt.imshow(np.degrees(angle))
    #tcu.write_geotiff('C:/Temp/angle.tif', np.shape(angle), geotransform, utm_epsg, np.degrees(angle))
    
    #filter pixels based on FOV of sensor
    angle[angle<=(math.pi/2)-sensor_half_FOV_rad]=np.nan
    nan_indices = np.where(np.isnan(angle))
    
    #set pixels to nan in other arrays
    elev_array[nan_indices] = np.nan
    
    #plt.imshow(elev_array)
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
    
    print('aspect: ' + str(weighted_aspect))
    print('tilt_dir: ' + str(row['tilt_dir']))
    
    
    # calculate cosine wighted average
    slope_arr_weighted = weighting * slope_array
    weighted_slope = np.nansum(slope_arr_weighted)
    #print('cosine-weighted mean slope: ' + str(weighted_slope))
    
    #cos_avg_slope = cos_avg_slope.append(weighted_slope)

    '''
    # now calculate arithmetic mean
    aspect_arr_masked = aspect_arr_weighted/weighting
    arth_mean_aspect = np.nanmean(aspect_arr_masked)
    #print('arithmetic mean aspect: ' + str(arth_mean_aspect))
    gdf.loc[index, 'mean_aspect'] = arth_mean_aspect
    #mean_aspect = mean_aspect.append(arth_mean_aspect)
    
    # now calculate arithmetic mean
    slope_arr_masked = slope_arr_weighted/weighting
    arth_mean_slope = np.nanmean(slope_arr_masked)
    #print('arithmetic mean slope: ' + str(arth_mean_slope))
    #mean_slope = mean_slope.append(arth_mean_slope)
    gdf.loc[index,'mean_slope'] = arth_mean_slope
    '''

    """
    #calculate cosine averaged ls8 albedo value
    ls8_arr_weighted = weighting * ls8_array
    weighted_ls8 = np.nansum(ls8_arr_weighted)
    #print('cosine-weighted mean aspect: ' + str(weighted_aspect))
    gdf.loc[index, 'cos_avg_ls8'] = weighted_ls8
    """
    
    
    return weighted_slope, weighted_aspect

def topo_correction(gdf, row, index, slope_field, aspect_field):
    
    measured_albedo = row[albedo_meas_fname] 
    tilt = row[tilt_fname] 
    tilt_dir = row[tilt_dir_fname] 
    #tilt_dir = row['mean_aspect']
    #slope_mean = row['mean_slope'] 
    slope_cos = row[slope_field]
    #aspect_mean = row['mean_aspect'] 
    aspect_cos = row[aspect_field]
    solar_zenith = row['6s_Solar_Zenith_Angle']
    solar_azimuth = row['6s_Solar_Azimuth_Angle'] 
    p_dir = row['6s_Direct_Irradiance_Proportion'] 
    p_diff = row['6s_Diffuse_Irradiance_Proportion']
    
    #print('zenith_angle: ' + str(solar_zenith))
    solar_zenith_rad = math.radians(solar_zenith)
    
    cos_pyranometer_incidence = tcu.get_cos_incidence_angle(solar_zenith, tilt, tilt_dir, solar_azimuth)
    
    #print('pyranometer incidence angle: ' + str(math.degrees(math.acos(cos_pyranometer_incidence))))
    
    #cos_mean_slope_incidence = tcu.get_cos_incidence_angle(solar_zenith, slope_mean, aspect_mean, solar_azimuth)
    
    cos_cos_slope_incidence = tcu.get_cos_incidence_angle(solar_zenith, slope_cos, aspect_cos, solar_azimuth)
    
    #print('slope incidence angle: ' + str(math.degrees(math.acos(cos_slope_incidence))))
    
    #arithmetic_avg_corrected_albedo = measured_albedo * ((p_diff * np.cos(solar_zenith_rad) + p_dir * cos_pyranometer_incidence) / 
    #                                   (p_diff * np.cos(solar_zenith_rad) + p_dir * cos_mean_slope_incidence))
    
    cosine_avg_corrected_albedo = measured_albedo * ((p_diff * np.cos(solar_zenith_rad) + p_dir * cos_pyranometer_incidence) / 
                                       (p_diff * np.cos(solar_zenith_rad) + p_dir * cos_cos_slope_incidence))
    
    gdf.loc[index, 'cos_sensor_incidence_angle'] = cos_pyranometer_incidence
    gdf.loc[index, 'cos_slope_incidence_angle'] = cos_cos_slope_incidence
    
    #gdf.loc[index, 'arithmetic_avg_corrected_albedo'] = arithmetic_avg_corrected_albedo
    
    return cosine_avg_corrected_albedo

def process_handler():
    gdf = pd.DataFrame()
    elev_array = []
    slope_array = []
    aspect_array = []
    coordinate_array_x = []
    coordinate_array_y = []
    
    if prepare_rasters == False:
        print('loading rasters')
        ss_elev = gdal.Open(elev_UTM_path, gdal.GA_ReadOnly) # open reprojected elevation GeoTiff 
        ss_slope = gdal.Open(path_to_slope, gdal.GA_ReadOnly)
        ss_aspect = gdal.Open(path_to_aspect, gdal.GA_ReadOnly)
        ss_coordinate_x = gdal.Open(path_to_coordinate_array_x, gdal.GA_ReadOnly)
        ss_coordinate_y = gdal.Open(path_to_coordinate_array_y, gdal.GA_ReadOnly)
        
        
        '''
        usgs_elev = gdal.Open(elev_UTM_path, gdal.GA_ReadOnly) # open reprojected elevation GeoTiff 
        usgs_slope = gdal.Open(path_to_slope, gdal.GA_ReadOnly)
        usgs_aspect = gdal.Open(path_to_aspect, gdal.GA_ReadOnly)
        usgs_coordinate_x = gdal.Open(path_to_coordinate_array_x, gdal.GA_ReadOnly)
        usgs_coordinate_y = gdal.Open(path_to_coordinate_array_y, gdal.GA_ReadOnly)
        
        bg_elev = gdal.Open(elev_UTM_path, gdal.GA_ReadOnly) # open reprojected elevation GeoTiff 
        bg_slope = gdal.Open(path_to_slope, gdal.GA_ReadOnly)
        bg_aspect = gdal.Open(path_to_aspect, gdal.GA_ReadOnly)
        bg_coordinate_x = gdal.Open(path_to_coordinate_array_x, gdal.GA_ReadOnly)
        bg_coordinate_y = gdal.Open(path_to_coordinate_array_y, gdal.GA_ReadOnly)
        '''
        ss_elev_arr = tcu.get_band_array(ss_elev)
        ss_slope_arr = tcu.get_band_array(ss_slope)
        ss_aspect_arr = tcu.get_band_array(ss_aspect)
        ss_coordinate_array_x_arr = tcu.get_band_array(ss_coordinate_x)
        ss_coordinate_array_y_arr = tcu.get_band_array(ss_coordinate_y)
        '''
        usgs_elev_arr = tcu.get_band_array(elev_UTM)
        usgs_slope_arr = tcu.get_band_array(slope_UTM)
        usgs_aspect_arr = tcu.get_band_array(aspect_UTM)
        usgs_coordinate_array_x_arr = tcu.get_band_array(coordinate_x)
        usgs_coordinate_array_y_arr = tcu.get_band_array(coordinate_y)
        
        bg_elev_arr = tcu.get_band_array(elev_UTM)
        bg_slope_arr = tcu.get_band_array(slope_UTM)
        bg_aspect_arr = tcu.get_band_array(aspect_UTM)
        bg_coordinate_array_x_arr = tcu.get_band_array(coordinate_x)
        bg_coordinate_array_y_arr = tcu.get_band_array(coordinate_y)
        '''
        print('rasters loaded')
        ss_elev = None 
        ss_slope = None
        ss_aspect = None
        ss_coordinate_x = None
        ss_coordinate_y = None
        
    else:
        
        elev_array, slope_array, aspect_array, coordinate_array_x, coordinate_array_y = prep_rasters()
    print('prepping point data')  
    gdf = prep_point_data()
    print('point data prepped')  
    for index, row in gdf.iterrows():
        print('-----------------------Row-------------------------')
        print('measured incoming: ' + str(row['incoming (W/m^2)']))
        
        if calculate_terrain_parameters:
            ss_slope, ss_aspect = calc_terrain_parameters(ss_elev_arr, ss_slope_arr, ss_aspect_arr, ss_coordinate_array_x_arr, ss_coordinate_array_y_arr, gdf, row, index)
            gdf.loc[index, 'cos_avg_slope_ss'] = ss_slope
            gdf.loc[index, 'cos_avg_aspect_ss'] = ss_aspect
            
            '''
            bg_slope, bg_aspect = calc_terrain_parameters(elev_array, slope_array, aspect_array, coordinate_array_x, coordinate_array_y, gdf, row, index)
            gdf.loc[index, 'cos_avg_slope_bg'] = bg_slope
            gdf.loc[index, 'cos_avg_aspect_bg'] = bg_aspect
            
            usgs_slope, usgs_aspect = calc_terrain_parameters(elev_array, slope_array, aspect_array, coordinate_array_x, coordinate_array_y, gdf, row, index)
            gdf.loc[index, 'cos_avg_slope_usgs'] = usgs_slope
            gdf.loc[index, 'cos_avg_aspect_usgs'] = usgs_aspect
            '''
            
        if run_radiative_transfer:
            gdf = radiative_transfer(gdf, row, index)
        
    if run_topo_correction:
        print('running topographic correction')
        
        for index, row in gdf.iterrows(): 
            cosine_avg_corrected_albedo_ss = topo_correction(gdf, row, index, 'cos_avg_slope_ss', 'cos_avg_aspect_ss')
            gdf.loc[index, 'corrected_albedo_ss'] = cosine_avg_corrected_albedo_ss
        
            '''
            cosine_avg_corrected_albedo_bg = topo_correction(gdf, row, index, 'cos_avg_slope_bg', 'cos_avg_aspect_bg')
            gdf.loc[index, 'corrected_albedo_bg'] = cosine_avg_corrected_albedo_bg
            
            cosine_avg_corrected_albedo_usgs = topo_correction(gdf, row, index, 'cos_avg_slope_usgs', 'cos_avg_aspect_usgs')
            gdf.loc[index, 'corrected_albedo_usgs'] = cosine_avg_corrected_albedo_usgs
            '''
            
    gdf.to_csv(csv_path)
    
def main():
    process_handler()

if __name__ == "__main__":
    #prep_rasters()
    main()
    print('hi')





