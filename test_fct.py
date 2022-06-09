import deepkriging as dk
import interpolation 
import data_preparation as dp
import numpy as np
import pandas as pd
import tensorflow as tf
import trainNN
import os

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
    
map = dp.stack_map(whole_map)

length = 100
start_point = 10

def test_cut_map_len():
    excepted = (start_point+length) * 12
    assert dp.cut_map_len(map, start_point, length).y.max() == excepted


map = dp.cut_map_len(map, start_point, length)


sampling_distance_x = 2
sampling_distance_y = 24
known_points, unknown_points = dp.resample(map, sampling_distance_x, sampling_distance_y)




def test_randomsampling():
    excepted = len(known_points.index.values)
    assert len(dp.randomsampling(map, len(known_points)).index.values) == excepted
    
"""
    Testing deepkriging.py functions
"""
def test_get_numbasis():
    N =  900 
    expected =  [10**2,19**2,37**2,73**2]
    assert dk.get_num_basis(N) == expected

def test_wendland_kernel():
    pass


    
