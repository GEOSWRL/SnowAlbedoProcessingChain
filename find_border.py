# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:35:38 2021

@author: x51b783
"""

import numpy as np
import os
os.environ['PROJ_LIB'] = 'C:\\Users\\x51b783\\.conda\\envs\\py37\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\x51b783\\.conda\\envs\py37\\Library\\share'
from osgeo import gdal, gdalconst
import topo_correction_util as tcu
from matplotlib import pyplot as plt

footprint_path = 'C:/Temp/weighting_footprint.tif'
footprint_source = gdal.Open(footprint_path, gdalconst.GA_ReadOnly)
footprint_gt = footprint_source.GetGeoTransform()
footprint_proj = footprint_source.GetProjection()
footprint_cols = footprint_source.RasterXSize
footprint_rows = footprint_source.RasterYSize
footprint_arr = footprint_source.GetRasterBand(1).ReadAsArray()


arr = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan,   7,      8,      1,    np.nan],
                [np.nan,   6,      9,      2,    np.nan],
                [np.nan,   5,      4,      3,    np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan]])

arr2 = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan,   7,      8,      1,    np.nan],
                [np.nan,   6,      9,      2,    np.nan],
                [np.nan,   5,      4,      np.nan,    np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan]])

arr3 = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                 
                 [np.nan, np.nan,    1,      2,      3,      4,      5,      6,     7,    np.nan],
                 
                 [np.nan, np.nan, np.nan,   35,      1,      11,      10,    9,     8,    np.nan],
                 
                 [np.nan,   32,     33,     34,      1,      12,    np.nan, np.nan,np.nan, np.nan],
                 
                 [np.nan,   31,     30,     29,     28,      13,      14,   15,    10,    np.nan],
                 
                 [np.nan, np.nan,  np.nan, np.nan,  27,      1,      1,      1,    11,    np.nan],
                 
                 [np.nan,   23,     24,     25,     26,      1,      1,      1,    12,    np.nan],
                 
                 [np.nan,   22,      1,      1,      1,      1,      1,      14,   13,    np.nan],
                 
                 [np.nan,   np.nan, 20,     19,     18,     17,     16,      15,   np.nan, np.nan],
                 
                 [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])


origin = ([315],[306])

#opx = origin[0][0]
#opy = origin[1][0]

lpx = origin[0][0]
lpy = origin[1][0]

cpx = origin[0][0]
cpy = origin[1][0]

border_pix = []


def get_ur(cpx, cpy, arr):
        
    return arr[[cpy-1], [cpx+1]][0]

def get_r(cpx, cpy, arr):
        
    return arr[[cpy], [cpx+1]][0]

def get_br(cpx, cpy, arr):
        
    return arr[[cpy+1], [cpx+1]][0]

def get_b(cpx, cpy, arr):
        
    return arr[[cpy+1], [cpx]][0]

def get_bl(cpx, cpy, arr):
        
    return arr[[cpy+1], [cpx-1]][0]

def get_l(cpx, cpy, arr):
        
    return arr[[cpy], [cpx-1]][0]

def get_ul(cpx, cpy, arr):
        
    return arr[[cpy-1], [cpx-1]][0]

def get_u(cpx, cpy, arr):
        
    return arr[[cpy-1], [cpx]][0]

def get_cp(cpx, cpy, arr):
    
    return arr[[cpy], [cpx]][0]

def check_borders_nan(cpx, cpy, arr):

    if np.isnan(get_u(cpx, cpy, arr)):
        #print('up is nan')
        return True
    
    if np.isnan(get_ur(cpx, cpy, arr)):
        #print('up is nan')
        return True
    
    if np.isnan(get_r(cpx, cpy, arr)):
        #print('right is nan')   
        return True
    
    if np.isnan(get_br(cpx, cpy, arr)):
        #print('up is nan')
        return True
    
    if np.isnan(get_b(cpx, cpy, arr)):
        #print('down is nan')
        return True
    
    if np.isnan(get_bl(cpx, cpy, arr)):
        #print('up is nan')
        return True
    
    if np.isnan(get_l(cpx, cpy, arr)):
        #print('left is nan')
        return True
    
    if np.isnan(get_ul(cpx, cpy, arr)):
        #print('up is nan')
        return True
    
    return False

def move(cpx, cpy, arr):
    
    #try right
    cpx_n = cpx + 1
    cpy_n = cpy
    #print('trying right')
    if not np.isnan(get_cp(cpx_n, cpy_n, arr)) and check_borders_nan(cpx_n, cpy_n, arr) and not [cpx_n, cpy_n] in border_pix:
        #print(get_cp(cpx_n, cpy_n, arr))
        #print('cpx = ' + str(cpx))
        return cpx_n, cpy_n
    
    #try down
    cpx_n = cpx
    cpy_n = cpy + 1
    #print('trying down')
    if not np.isnan(get_cp(cpx_n, cpy_n, arr)) and check_borders_nan(cpx_n, cpy_n, arr) and not [cpx_n, cpy_n] in border_pix:
        #print(get_cp(cpx_n, cpy_n, arr))
        #print('cpx = ' + str(cpx))
        return cpx_n, cpy_n
    
    #try left
    cpx_n = cpx -1
    cpy_n = cpy
    #print('trying left')
    if not np.isnan(get_cp(cpx_n, cpy_n, arr)) and check_borders_nan(cpx_n, cpy_n, arr) and not [cpx_n, cpy_n] in border_pix:
        #print(get_cp(cpx_n, cpy_n, arr))
        #print('cpx = ' + str(cpx))
        return cpx_n, cpy_n
    
    #try up
    cpx_n = cpx
    cpy_n = cpy - 1
    #print('trying up')
    if not np.isnan(get_cp(cpx_n, cpy_n, arr)) and check_borders_nan(cpx_n, cpy_n, arr) and not [cpx_n, cpy_n] in border_pix:
        #print(get_cp(cpx_n, cpy_n, arr))
        #print('cpx = ' + str(cpx))
        return cpx_n, cpy_n
    
    #try up-right
    cpx_n = cpx + 1
    cpy_n = cpy - 1
    #print('trying up-right')
    if not np.isnan(get_cp(cpx_n, cpy_n, arr)) and check_borders_nan(cpx_n, cpy_n, arr) and not [cpx_n, cpy_n] in border_pix:
        #print(get_cp(cpx_n, cpy_n, arr))
        #print('cpx = ' + str(cpx))
        return cpx_n, cpy_n
    
    #try down-right
    cpx_n = cpx + 1
    cpy_n = cpy +1
    #print('trying down-right')
    if not np.isnan(get_cp(cpx_n, cpy_n, arr)) and check_borders_nan(cpx_n, cpy_n, arr) and not [cpx_n, cpy_n] in border_pix:
        #print(get_cp(cpx_n, cpy_n, arr))
        #print('cpx = ' + str(cpx))
        return cpx_n, cpy_n
    
    #try down-left
    cpx_n = cpx - 1
    cpy_n = cpy +1
    #print('trying down-left')
    if not np.isnan(get_cp(cpx_n, cpy_n, arr)) and check_borders_nan(cpx_n, cpy_n, arr) and not [cpx_n, cpy_n] in border_pix:
        #print(get_cp(cpx_n, cpy_n, arr))
        #print('cpx = ' + str(cpx))
        return cpx_n, cpy_n
    
    #try up-left
    cpx_n = cpx - 1
    cpy_n = cpy - 1
    #print('trying up-left')
    if not np.isnan(get_cp(cpx_n, cpy_n, arr)) and check_borders_nan(cpx_n, cpy_n, arr) and not [cpx_n, cpy_n] in border_pix:
        #print(get_cp(cpx_n, cpy_n, arr))
        #print('cpx = ' + str(cpx))
        return cpx_n, cpy_n
    
    return -10, -10
    
while not np.isnan ( get_u(cpx, cpy, footprint_arr) ):
    cpy-=1


pos = []

while True:
    cpx, cpy = move(cpx,cpy,footprint_arr)
    
    if cpx == -10:
        
        if pos == 1:
            break
        
        cpx, cpy = move(border_pix[pos-1][0], border_pix[pos-1][1], footprint_arr)
        pos-=1
        
    
    else:
        border_pix.append([cpx,cpy])
        pos = len(border_pix) - 1
        
    print('value = ' + str(get_cp(cpx, cpy, footprint_arr)))
    print('cpx = ' + str(cpx))
    print('cpy = ' + str(cpy))
    print('---------------------------------------------------')

    
    
    
border_pix = np.array(border_pix) 
border_pix = np.hsplit(border_pix,2)

footprint_arr[border_pix[1], border_pix[0]] =-100

plt.imshow(footprint_arr)
tcu.write_geotiff('C:/Temp/weighting_footprint_outlined.tif', np.shape(footprint_arr), footprint_gt, 32612, footprint_arr)
print(np.nanmax(footprint_arr))

print(np.nanmin(footprint_arr))


#move(cpx, cpy, arr)


    

