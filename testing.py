# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:37:46 2020

@author: x51b783
"""
import exiftool
import os
import earthpy as ep

def myfunct(x, hello = 'hello', hello2 = 'hello2'):
    print(x)
    print(hello)
    print(hello2)
    
myfunct(12, hello2 = 'hhhhhh')