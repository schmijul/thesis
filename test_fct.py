import deepkriging as dk
import interpolation 
import data_preparation as dp
import numpy as np
import pandas as pd
import tensorflow as tf
import trainNN
import os

import pytest 

"""
        This Script functions as a quick unit test for : 
                                                        data_preparation.py
                                                        interpolation.py
                                                        trainNN.py
                                                        deepkriging.py

"""
    
    
    
    
whole_map = pd.read_csv('WholeMap_Rounds_40_to_17.csv')
"""
    Testing data_preparation.py functions

"""    
                
def test_stack_map():
    excepted = (len(whole_map)*(len(whole_map.columns)-1),3)
    assert dp.stack_map(whole_map).shape == excepted
    
Map, StackedMap = dp.prepare_map(whole_map)

start_point = 0
length = 100


def test_cut_map_len():
    excepted = (start_point+length) * 12
    assert dp.cut_map_len(StackedMap, start_point, length).y.max() == excepted



sampling_distance_x = 2
sampling_distance_y = 24
known_points, unknown_points = dp.resample(StackedMap, sampling_distance_x, sampling_distance_y)


def test_randomsampling():
    expected = len(known_points)
    assert len(dp.randomsampling(StackedMap, len(known_points))) == expected
    
"""
    Testing deepkriging.py functions
"""

def test_findWorkingNumBasis():
    N =  900 
    H = 4
    expected =  [10**2,19**2,37**2,73**2]
    assert dk.findWorkingNumBasis(N,H) == expected


def test_wendlandkernel():
    
    # Testing wendland kernel with a known Phi taken from : https://github.com/aleksada/DeepKriging/blob/master/Nonstationary_function-2d.ipynb
    
    expected = pd.read_csv('wendland_from_paper.csv').drop(columns = ['Unnamed: 0'])
    print(expected.shape)
    n = 30 
    N = int(n**2) ## sample size
    M = 1 ## Number of replicate
    coord1 = np.linspace(0,1,n)
    coord2 = np.linspace(0,1,n)
    s1,s2 = np.meshgrid(coord1,coord2)
    s = np.vstack((s1.flatten(),s2.flatten())).T
    print(s.shape)
    
    print(s.shape)
    print(np.vstack(s).shape)
    points = pd.DataFrame(s, columns = ['x','y'])
    
    
    numBasis = [10**2,19**2,37**2]
    
    
    phi = dk.wendlandkernel(points,numBasis)
    assert phi.equals(expected) == False

test_wendlandkernel()