import pandas as pd

import deepkriging as dk
import datapreparation as dp


whole_map = pd.read_csv('reference.csv')


def test_stack_map():
    """
    check if the map is stacked correctly
    """
    excepted = (len(whole_map)*(len(whole_map.columns)-1),3)
    assert dp.stackmap(whole_map).shape == excepted

stackedmap = dp.preparemap(whole_map)[1]

STARTPOINT = 0
LENGTH = 100


def test_cut_map_len():
    """
    check if the map is cut correctly
    """
    excepted = (STARTPOINT+LENGTH)
    assert dp.cutmap(stackedmap, STARTPOINT, LENGTH).y.max() == excepted



SAMPLINGDIST_X = 2
SAMPLINGDIST_Y = 24
known_points, unknown_points = dp.resample(stackedmap, SAMPLINGDIST_X, SAMPLINGDIST_Y)


def test_randomsampling():
    """
    check if randomly sampled data hast the correct shape
    """
    expected = len(known_points)
    assert len(dp.randomsampling(stackedmap, len(known_points))) == expected


def test_findworkingnumbasis():
    numcoords =  900
    numelementsbas = 4
    expected =  [10**2,19**2,37**2,73**2]
    assert dk.findworkingnumbasis(numcoords,numelementsbas) == expected
