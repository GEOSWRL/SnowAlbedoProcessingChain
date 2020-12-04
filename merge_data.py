# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:50:13 2020

@author: Andrew Mullen
"""

import pandas as pd
import pytz
import slope_correction


paths_to_DJI = ['C:/Masters/DroneAlbedoProject/Field_Data/tarp/09132020_parkingLot/flight_logs/FLY218.csv']
path_to_Meteon = 'C:/Masters/DroneAlbedoProject/Field_Data/tarp/09132020_parkingLot/pyranometers/20200913_pyranometers.csv'

angle_tolerance = 3 #degree threshold of tilt and roll, angles greater than this will be filtered out
height_tolerance = 0.3 #records below this relative altitude(m) will be filtered out

def correct_tilt_direction(a):
    if(a<0):
        a+=360
        return a
    return a

def parse_DJI(paths_to_DJI):
    """
    Creates new flight log file that is filtered, timestamps converted, and ready to be merged with irradiance data

    Parameters
    ----------
    path_to_DJI : string
        path to the directory containing DJI logfiles (in .csv format)
    
    Returns
    -------
    result : Pandas Dataframe
        filetered DJI flight log file ready to be merged

    """
    dfs = []
    
    for path in (paths_to_DJI):
        
        #extract desired columns
        df = pd.read_csv(path, usecols=[10,22,25,26,27,28,29,42,43,44,45,46,47,48,49,50,51,52], header=0)

        #filtering out altitudes <5m and pitch and roll >5 degrees
        df = df.loc[(df['General:relativeHeight']>=height_tolerance) & (df['IMU_ATTI(0):roll']<=angle_tolerance) & (df['IMU_ATTI(0):roll']>=-angle_tolerance) & (df['IMU_ATTI(0):pitch']<=angle_tolerance) & (df['IMU_ATTI(0):pitch']>=-angle_tolerance)]

        #convert to mountain time
        mountain = pytz.timezone('US/Mountain')
        df['GPS:dateTimeStamp'] = pd.DatetimeIndex(df['GPS:dateTimeStamp']).tz_localize(None)
        df['GPS:dateTimeStamp'] = pd.DatetimeIndex(df['GPS:dateTimeStamp']).tz_localize('Zulu')
        df['GPS:dateTimeStamp'] = pd.DatetimeIndex(df['GPS:dateTimeStamp']).tz_convert(mountain)
    
        #flight logs collect on milliseconds, so we must average values over each second
        df = df.groupby(df['GPS:dateTimeStamp']).mean()
        
        #adjust tilt_Direction to 360 deg. azimuth
        df['IMU_ATTI(0):tiltDirection'] = df['IMU_ATTI(0):tiltDirection'].apply(correct_tilt_direction)
        
        #run py6s radiative transfer
        
        df = slope_correction.run_radiative_transfer(df, 'GPS(0):Lat', 'GPS(0):Long', 'GPS(0):heightMSL', 'GPS:dateTimeStamp')
        
        dfs.append(df)
        
        #new dataframe with stabilized measurements
        
        
        
    

    return dfs
    

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
    mountain = pytz.timezone('US/Mountain')
    df = pd.read_csv(path_to_Meteon, usecols=[0,2,5], skiprows=9, names = ["Time", "incoming (W/m^2)", "reflected (W/m^2)"], parse_dates=True)
    #df = pd.read_csv(path_to_Meteon, usecols=[0,5,2], skiprows=9, names = ["Time", "reflected (W/m^2)", "incoming (W/m^2)"], parse_dates=True)
    
    df.insert(3,'albedo', df['reflected (W/m^2)'].div(df['incoming (W/m^2)'])) #calculate albedo
    
    df = df.loc[(df['albedo']<=1)] #albedo cannot be > 1
    
    df['Time'] = pd.DatetimeIndex(df['Time']).tz_localize(mountain)
    df.set_index('Time', inplace=True)
    return df
    
    

def merge_tables(DJI_parsed, Meteon_parsed):
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
    result : writes file in same directory as other files
        filtered and merged table containing all irradiance measurements with associated flight log

    """
    
    x=0
    merged = []
    for df in DJI_parsed:
        m = pd.merge(DJI_parsed[x], Meteon_parsed, left_on = DJI_parsed[x].index, right_on = Meteon_parsed.index, how='inner')
        
        merged.append(m)
        m.to_csv(paths_to_DJI[x][:-4] + "_merged.csv")
        
        x+=1
        
    return merged


DJI_parsed = parse_DJI(paths_to_DJI)
Meteon_parsed = parse_Meteon(path_to_Meteon)
merged = merge_tables(DJI_parsed, Meteon_parsed)

    
