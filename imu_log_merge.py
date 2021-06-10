# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 00:14:18 2021

@author: x51b783
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:50:13 2020

@author: Andrew Mullen


conda install -c conda-forge earthpy
"""

import pandas as pd
import pytz
import os
import earthpy as ep
import numpy as np
import topo_correction_util as tcu

#set working directory
imu_path = 'C:/Temp/imu_logs/'
meteon_path = 'C:/Temp/meteon/validation/'
if not (os.path.exists(imu_path)):
    print('working directory does not exist')

#set paths to logfiles    
paths_to_IMU = [os.path.join(imu_path, f) for f in os.listdir(os.path.join(imu_path))]
#paths_to_Meteon = os.path.join(meteon_path, 'logfiles', 'meteon', os.listdir(os.path.join(meteon_path, 'logfiles', 'meteon'))[0])


angle_tolerance = 5 #degree threshold of tilt and roll, angles greater than this will be filtered out
height_tolerance = 0.3 #records below this relative altitude(m) will be filtered out

timezone = 'US/Mountain'



def tilt_direction_to_azimuth(a):
    """
    Creates new Geotiff from numpy array

    Parameters
    ----------
    a : integer angle from -180 to 180 degrees
    
    Returns
    -------
    a : angle corrected to 360 degree azimuth (clockwise from 0 degrees at North)

    """
    
    if(a<0):
        a+=360
        return a
    return a

def parse_IMU(path):
    """
    Creates new flight log file that is filtered, timestamps converted, and ready to be merged with irradiance data

    Parameters
    ----------
    path_to_DJI : string
        path to the directory containing DJI logfiles (in .csv format)
    
    Returns
    -------
    dfs : list of Pandas Dataframes
        DataFrames for filetered DJI flight logs file ready to be merged
    
    filenames : String
        Filenames corresponding to DJI flight logs

    """
        
    #extract desired columns
    df = pd.read_csv(path, usecols=['Record Time', 'AngleX', 'AngleY', 'AngleZ'], header=1, skiprows=0)

    #filtering out altitudes <5m and pitch and roll >5 degrees
    #df = df.loc[(df['General:relativeHeight']>=height_tolerance) & (df['IMU_ATTI(0):roll']<=angle_tolerance) & (df['IMU_ATTI(0):roll']>=-angle_tolerance) & (df['IMU_ATTI(0):pitch']<=angle_tolerance) & (df['IMU_ATTI(0):pitch']>=-angle_tolerance)]

    #convert to mountain time
    df['Record Time'] = pd.DatetimeIndex(df['Record Time']).tz_localize(timezone)
        
    
    #flight logs collect on milliseconds, so we must average values over each second
    df = df.groupby(df['Record Time']).mean()
        
    return df
    

def parse_Meteon(path_to_Meteon):
    """
    Creates new logfile with unnecesary columns removed, ready to be merged with DJI flight log

    Parameters
    ----------
    path_to_DJI : string
        path to the directory containing Meteon logfiles (in .csv format)
    
    Returns
    -------
    result : Pandas Dataframe
        filetered Meteon log file ready to be merged

    """    
    df = pd.read_csv(path_to_Meteon, usecols=[0,2,5], skiprows=9, names = ["Time", "incoming (W/m^2)", "reflected (W/m^2)"], parse_dates=True)
    #df = pd.read_csv(path_to_Meteon, usecols=[0,5,2], skiprows=9, names = ["Time", "reflected (W/m^2)", "incoming (W/m^2)"], parse_dates=True)
    
    df.insert(3,'albedo', df['reflected (W/m^2)'].div(df['incoming (W/m^2)'])) #calculate albedo
    
    df = df.loc[(df['albedo']<=1)] #albedo cannot be > 1
    
    df['Time'] = pd.DatetimeIndex(df['Time']).tz_localize(None)
    df['Time'] = pd.DatetimeIndex(df['Time']).tz_localize(pytz.timezone(timezone))
    df.set_index('Time', inplace=True)
    return df
    

def merge_tables(IMU_parsed, Meteon_parsed):
    """
    Creates a new .csv file from merging the DJI flight log and Meteon data based on timestamp

    Parameters
    ----------
    DJI_parsed : string
        path to the directory containing parsed DJI logfile
    Meteon_parsed : string
        path to the directory containing parsed Meteon ligfile

    Returns
    -------
    merged: list of Pandas DataFrame objects
        DataFrame objects for each merged DJI flight log
        
    result : writes file in same directory as other files
        filtered and merged table containing all irradiance measurements with associated flight log

    """
    
        
    #take intersection of two dataframes based on index
    merged = pd.concat([IMU_parsed, Meteon_parsed], axis=1, join='inner')
    print(merged.head())
    #add to list of dataframes
    
    return merged


def process_and_merge():
    imu_all = pd.DataFrame()

    for p in os.listdir(imu_path):
        imu_parsed = parse_IMU(imu_path+p)
        imu_all = imu_all.append(imu_parsed)
    
    for m in os.listdir(meteon_path):
        
        meteon_parsed = parse_Meteon(meteon_path + m)
        
        meteon_merged = merge_tables(imu_all, meteon_parsed)
    
        meteon_merged.to_csv(meteon_path + m)
        
def calc_tilt_and_tilt_dir():
    for m in os.listdir(meteon_path):
        df = pd.read_csv(meteon_path+m)
        df['tilt'] = np.zeros(df.shape[0])
        df['tilt_dir'] = np.zeros(df.shape[0])
        for index, row in df.iterrows():
            tilt, tilt_dir = tcu.get_tilt(row['AngleY'], row['AngleX'], row['AngleZ'])
            df.loc[index, 'tilt'] = tilt
            df.loc[index, 'tilt_dir'] = tilt_dir
        df.to_csv(meteon_path + m)
        
        
calc_tilt_and_tilt_dir()
            
        
    
    
    
    
