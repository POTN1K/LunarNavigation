# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 13:48:45 2023

@author: M. Dinescu
"""

'''
Verification and Validation of the ADCS
written by M. Dinescu
'''
import numpy as np


def test_array(func, width, height):
    '''Checks if array has correct size'''
    
    test_w = np.shape(func)[1] == width
    test_h = np.shape(func)[0] == height
    return test_h, test_w
    
def test_pos(func):
    '''Checks if output is positive'''
    
    t_mag = func > 0
    return t_mag

def test_calc(func, value):
    '''Checks if signs of outputs the same,
    then check if output is in pm 5 % margin'''
    
    t_mag = (func > 0 and value > 0) or (func < 0 and value < 0)
    t_calc = (np.abs(func) < 1.05*np.abs(value)) and (np.abs(func) > 0.95*np.abs(value))
    return t_mag, t_calc

def assign_Zero():
    ''' Assigns a zero value to desired parameter
        Make sure to assign after initial designation '''
    
    a = 0
    return a

def test_sensitivity(start, end, step):
    ''' varies requested parameter
        Make sure to assign after initial designation'''
    
    R = np.arange(start, end + step, step)
    return R

def test_planes(planea, planeb):
    ''' Checks the two planes considered are not the same '''
    
    test_p = planea != planeb
    return test_p
        
