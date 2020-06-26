# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:20:59 2020

@author: aman3
"""

"""
topographic correction

"""

"""
steps


4: For each orthophoto from orthomosaic
    a: overlay on slope - azimuth map to generate slope - azimuth of orthophoto
    b: take synchronous radiometer measurements to generate slope corrections
    
"""
import math
import gdal
from Py6S import *
import pandas as pd
import dateutil
from pytz import timezone
#import pysolar
import os

path_to_dem =  'C:/Users/aman3/Documents/GradSchool/testing/50mDEM.tif'
path_to_output = 'C:/Users/aman3/Documents/GradSchool/testing/50m'
path_to_orthophotos = 'C:/Users/aman3/Documents/GradSchool/testing/ortho/DJI_0726.tif'
path_to_log = 'C:/Users/aman3/Documents/GradSchool/testing/data/imageData.csv'

df = pd.read_csv(path_to_log)
df.set_index('Image ID', inplace = True)


def get_incidence_angle(topocentric_zenith_angle, slope, slope_orientation, topocentric_azimuth_angle):
    tza_rad = math.radians(topocentric_zenith_angle)
    slope_rad = math.radians(slope)
    so_rad = math.radians(slope_orientation)
    taa_rad = math.radians(topocentric_azimuth_angle)
    return math.degrees(math.acos(math.cos(tza_rad) * math.cos(slope_rad) + math.sin(slope_rad) * math.sin(tza_rad) * math.cos(taa_rad - so_rad)))

def process_DEM(path_to_dem, path_to_output):
    gdal.DEMProcessing(path_to_output+'slope.tif', path_to_dem, 'slope')
    gdal.DEMProcessing(path_to_output + 'aspect.tif', path_to_dem, 'aspect')

def run_radiative_transfer(df):
    
    DIP = []
    solar_zenith = []
    solar_azimuth = []

    for index, row in df.iterrows():
        lat = row['GPSLatitude']
        lon = row['GPSLongitude']
        alt = row['GPSAltitude']/1000
        dt = row['Timestamp']
        dt = dateutil.parser.parse(dt, dayfirst=True)
        dt = str(dt.astimezone(timezone('gmt')))
    
        s = SixS()
        s.wavelength = Wavelength(0.31, 2.7)
        s.altitudes.set_sensor_custom_altitude(alt)
        s.geometry.from_time_and_location(lat, lon, dt, 0, 0)
        s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
        s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeWinter)
        s.run()

        DIP.append(s.outputs.percent_direct_solar_irradiance)
        solar_zenith.append(s.outputs.solar_z)
        solar_azimuth.append(s.outputs.solar_a)
    
    df['Direct_Irradiance_Proportion'] = DIP
    df['Solar_Zenith_Angle'] = solar_zenith
    df['Solar_Azimuth_Angle'] = solar_azimuth
    df.to_csv(path_to_log)
    
def run_correction(ortho_dir, path_to_slope, path_to_aspect):
    df = pd.read_csv(path_to_log)
    df.set_index('Image ID', inplace = True)
    
    for filename in os.listdir(ortho_dir):
            if (filename.endswith('.tiff')):
                tza = df.loc[filename[:-5]]['Solar_Zenith_Angle']
                taa = df.loc[filename[:-5]]['Solar_Azimuth_Angle']
                direct_proportion = df.loc[filename[:-5]]['Direct_Irradiance_Proportion']
                
                """
                raster calculator
                
                corrected = tiff/(direct_proportion*incoming_DN*(cos(incidence_angle)/cos(tza)) + (1-direct_proportion)*incoming_DN)
                """
    
process_DEM(path_to_dem, path_to_output)

