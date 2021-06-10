# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 20:50:41 2021

@author: Andrew Mullen
"""
from scipy.spatial.transform import Rotation as R
import numpy as np

# define rotation angles in degrees. These are also known as "Euler angles"
pitch = 27.9 #pitch = Y axis rotation
roll = 0.29 #roll = X axis rotation
yaw = 94 #yaw = Z axis rotation

def unit_vector(vector):
    """
    Creates unit vector for a given vector of any size
    
    Parameters
    ----------
    vector : array of integers
        1-dimensional array of any length
    
    Returns
    -------
    result : array of integers
        unit vector of input vector (unit vector is a vector with the same direction and a magnitude of 1)
    """
    
    return vector / np.linalg.norm(vector)

def angle_between_vectors(v1, v2):
    """
    Calculates angle between two vectors
    
    Parameters
    ----------
    v1 : array of integers
        1-dimensional array of any length
        
    v2 : array of integers
        1-dimensional array of any length
    
    Returns
    -------
    angle_between : float
        angle (in degrees) between vectors v1 and v2
    """
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    angle_between = np.degrees(angle)
    
    return angle_between

def get_tilt_witmotion(pitch, roll, yaw):
    """
    Calculates tilt and tilt direction of Witmotion (shitmotion) IMU
    
    Parameters
    ----------
    pitch : float
        pitch angle of IMU (degrees)
        
    roll : float
        roll angle of IMU (degrees)
        
    yaw: float
        yaw angle of IMU (degrees)
    
    Returns
    -------
    tilt_angle_deg: float
        tilt angle (degrees) of IMU, can be thought of as slope angle
        
    tilt_dir_deg: float
        tilt direction (degrees) of IMU, can be thought of as aspect angle relative to True North
    """
    
    #convert angles to radians
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    yaw_rad = np.radians(yaw)

    #surface normal vector in (x,y,z) of a horizontal plane
    surface_normal = np.array([0,0,1])

    #create Rotation object r from pitch, roll and yaw
    #scipy Rotation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    #website about Euler angles: http://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
    r = R.from_euler('ZYX', [yaw_rad, roll_rad, pitch_rad], degrees=False)
    
    # now we will apply rotation r to the surface normal of horizontal plane
    # to get the (x,y,z) of the normal vector to the tilted surface
    new_surface_normal = r.apply(surface_normal) 
    print(new_surface_normal)
    
    # Now to get the tilt (slope), we need to calculate the angle between the normal vector
    # of the horizontal plane and the normal vector of the tilted surface (new_surface_normal)
    tilt_angle_deg = angle_between_vectors(surface_normal, new_surface_normal)
    
    # Now to get the tilt direction (aspect), we need to calculate the angle between the vector that represents North (x,y) = (0,1)
    # of the horizontal plane and the normal vector of the tilted surface (new_surface_normal)
    tilt_dir_deg = angle_between_vectors([0,1], new_surface_normal[:2])
    
    # Since the angle_between function returns the acute angle between vectors, if the (x,y) of the rotated surface normal
    # falls into either of the two left quadrants, we need to multiply the angle by -1
    if(new_surface_normal[0] <=0 and new_surface_normal[1] <=0):
        tilt_dir_deg*=-1
    
    if(new_surface_normal[0] <=0 and new_surface_normal[1] >=0):
        tilt_dir_deg*=-1
    
    return tilt_angle_deg, tilt_dir_deg

tilt, tilt_dir = get_tilt_witmotion(pitch, roll, yaw)


