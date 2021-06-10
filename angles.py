# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:43:26 2021

@author: x51b783
"""

import topo_correction_util as tcu
from scipy.spatial.transform import Rotation as R
import numpy as np

#aspect = 97
dip = 29
dip_dir = 295

pitch = 27.9

roll = 0.29

yaw = 94

'''
pitch = -10

roll = 0

yaw = -90
'''

#56
#63
#yaw_offset = 63

#yaw -= yaw_offset


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    tilt_dir_deg = np.degrees(angle)
    
    '''
    if(v2[0] >=0 and v2[1] >=0):
        return tilt_dir_deg
    
    if(v2[0] >=0 and v2[1] <=0):
        return tilt_dir_deg
    
    if(v2[0] <=0 and v2[1] <=0):
        return 360-tilt_dir_deg
    
    if(v2[0] <=0 and v2[1] >=0):
        return 360-tilt_dir_deg
    '''
    
    return tilt_dir_deg

def get_tilt_dir_dji(surface_normal):
    
    vector_north = [1, 0]
    
    angle = angle_between(vector_north, surface_normal[:2])
    
    if(surface_normal[0] >=0 and surface_normal[1] >=0):
        return angle
    
    if(surface_normal[0] >=0 and surface_normal[1] <=0):
        return -1*angle
    
    if(surface_normal[0] <=0 and surface_normal[1] <=0):
        return -1*angle
    
    if(surface_normal[0] <=0 and surface_normal[1] >=0):
        return angle
    
    return angle
    
def rotate_normals(surface_normal, pitch_rad, roll_rad, yaw_rad):
    
    rot_matrix = [
        [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(roll)*np.sin(pitch)-np.cos(roll)*np.sin(yaw), np.sin(roll)*np.sin(yaw)+np.cos(roll)*np.cos(yaw)*np.sin(pitch)],
        [np.cos(pitch)*np.sin(yaw), np.cos(roll)*np.cos(yaw)+np.sin(roll)*np.sin(yaw)*np.sin(pitch), np.cos(roll)*np.sin(yaw)*np.sin(pitch)-np.cos(yaw)*np.sin(roll)],
        [-1*np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(roll)*np.cos(pitch)],]
    
    new_surface_normal = np.dot(rot_matrix, surface_normal)
    
    enu_surface_normal = np.array([new_surface_normal[1], new_surface_normal[0], new_surface_normal[2]])
    
    return enu_surface_normal



def get_tilt_witmotion(pitch, roll, yaw):
    #pitch*=-1
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    yaw_rad = np.radians(yaw)

    surface_normal = np.array([0,0,1])

    
    r = R.from_euler('ZYX', [yaw_rad, roll_rad, pitch_rad], degrees=False)
    
    
    new_surface_normal = r.apply(surface_normal)
   
    print(new_surface_normal)
    
    angle = np.arccos(np.abs((surface_normal[0] * new_surface_normal[0] + 
                              surface_normal[1] * new_surface_normal[1] + 
                              surface_normal[2] * new_surface_normal[2]) /
                             (np.sqrt(np.square(new_surface_normal[0])+np.square(new_surface_normal[1])+np.square(new_surface_normal[2])) *
                              np.sqrt(np.square(surface_normal[0])+np.square(surface_normal[1])+np.square(surface_normal[2])
                                      ))
                             ))
    tilt_deg = np.degrees(angle)
    tilt_dir_deg = angle_between([0,1], new_surface_normal[:2])
    
    return tilt_deg, tilt_dir_deg






def get_tilt_dji(pitch, roll, yaw):
    #pitch*=-1
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    yaw = np.radians(yaw)
    
    
    surface_normal = np.array([0,0,-1])
    
    
    r = R.from_euler('ZYX', [yaw, pitch, roll], degrees=False)
    print(r.as_matrix())
    new_surface_normal = r.apply(surface_normal)
    
    print(new_surface_normal)
    
    
    
    angle = np.arccos(np.abs((surface_normal[0] * new_surface_normal[0] + 
                              surface_normal[1] * new_surface_normal[1] + 
                              surface_normal[2] * new_surface_normal[2]) /
                             (np.sqrt(np.square(new_surface_normal[0])+np.square(new_surface_normal[1])+np.square(new_surface_normal[2])) *
                              np.sqrt(np.square(surface_normal[0])+np.square(surface_normal[1])+np.square(surface_normal[2])
                                      ))
                             ))
    tilt_deg = np.degrees(angle)
    
    tilt_dir_deg = get_tilt_dir_dji(new_surface_normal)
    
    return tilt_deg, tilt_dir_deg



tilt, tilt_dir = get_tilt_witmotion(pitch, roll, yaw)

#slope = 6.3

#diff = tilt_dir - aspect
