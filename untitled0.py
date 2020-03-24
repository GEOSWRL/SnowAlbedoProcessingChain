# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:50:13 2020

@author: x51b783
"""
import gdal
from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry

original_tif = 'C:/Masters/DroneAlbedoProject/Field_Data/YC/YC20200127/Agisoft/YC20200127_Ortho.tif'
red = 'C:/Masters/DroneAlbedoProject/Field_Data/YC/YC20200127/Agisoft/YC20200127_Ortho_R.tif'
green = 'C:/Masters/DroneAlbedoProject/Field_Data/YC/YC20200127/Agisoft/YC20200127_Ortho_G.tif'
blue = 'C:/Masters/DroneAlbedoProject/Field_Data/YC/YC20200127/Agisoft/YC20200127_Ortho_B.tif'

gdal.Translate(red, original_tif, bandList=[1])
gdal.Translate(green, original_tif, bandList=[2])
gdal.Translate(blue, original_tif, bandList=[3])

calc = QgsRasterCalculator( 'boh@1/10000', 
                        'E:/data/abc.tif', 
                        'GTiff',
                        original_tif.extent(), 
                        original_tif.width(), 
                        original_tif.height(), 
                        )

gdal.calc
