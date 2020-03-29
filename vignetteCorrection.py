# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:19:00 2020

@author: aman3
"""

"""
Step 1 :
    calculate average image for R,G,B respectively
    



"""
import os
import rasterio
import separate_RGB

path_to_raw_files = 'C:/Users/aman3/Documents/GradSchool/testing/data/'
path_to_output = 'C:/Users/aman3/Documents/GradSchool/testing/output/'
r = []
g = []
b = []
numFiles = 0

for filename in os.listdir(path_to_raw_files):
    if filename.endswith(".DNG"):
        separate_RGB.raw_to_tiff(filename)

for filename in os.listdir(path_to_output):
    
    if filename.endswith(".tiff"):
        print(filename)
        tiff = path_to_output+filename
        #split tiff into R, G, and B tiffs
        red, green, blue = separate_RGB.split_rgb(tiff)
        if numFiles == 0:
            r = red
            g = green
            b = blue
        
        else:
            r += red
            g += green
            b += blue
        numFiles+=1
        
   
#take average
avg_R = (r/numFiles).astype('uint16')
avg_G = (g/numFiles).astype('uint16')
avg_B = (b/numFiles).astype('uint16')
    
    
#create avg red tiff
new_dataset = rasterio.open(
path_to_output + 'avg_red.tiff',
'w',
driver='GTiff',
height=avg_R.shape[0],
width=avg_R.shape[1],
count=1,
dtype=avg_R.dtype
)
new_dataset.write(avg_R,1)
new_dataset.close()

#create avg green tiff
new_dataset = rasterio.open(
path_to_output + 'avg_green.tiff',
'w',
driver='GTiff',
height=avg_G.shape[0],
width=avg_G.shape[1],
count=1,
dtype=avg_G.dtype
)
new_dataset.write(avg_G,1)
new_dataset.close()

#create avg blue tiff
new_dataset = rasterio.open(
path_to_output + 'avg_blue.tiff',
'w',
driver='GTiff',
height=avg_B.shape[0],
width=avg_B.shape[1],
count=1,
dtype=avg_B.dtype
)
new_dataset.write(avg_B,1)
new_dataset.close()