import numpy as np
import pandas as pd

def stack_map(map):

        """

        make sure all values are float values and not strings
        
        Assumption :    first column represents the y axis and the other columns represent the x-axis
                        There fore the first column ( y-axis) will be used as an index
        """
       
        map = map.astype(float).set_index('Unnamed: 0')
        map = map.reset_index().drop(columns=['Unnamed: 0'])
        map.columns = [x for x in range(len(map.columns))]

        
        map_stacked = map.stack().reset_index()
        map_stacked.columns = ['y', 'x', 'z']
        map_stacked = map_stacked[['x', 'y', 'z']]

        return map_stacked

def cut_map_len(map,start_point,length):

        """

        map is supposed to be a stacked dataframe with columns x, y, z
        start_point : smallest y val
        length is the wanted length of the map in cm

        12 steps on y axis = 1 cm

        if length = 700 (7m), then 12*700 cm are required to get to 7m 
        therefore the length * 0.12 = the length in cm 
        
        """
        
        # Cheking inputs Args
        
        if not (len(map >= start_point+length)):
                print('length is longer than the map length')
                return False
        
        
        
        # Fct begins here
        
        
        end_point = start_point+length # Find endpoint
        
        max_y = end_point *12 # Calc to cm

        map = map[map['y'] <= max_y]
        
        map = map[map['y'] >= start_point*12]
        
        return map

def prepare_map(wholeMap,start_point=0, length=None):
        
        """
        _summary_
                Args:
                        wholeMap (pandas DataFrame) : dataframe where first column represents the y axis and the other columns represent the x-axis
        
                        start_point (int) : smallest y val
                        length (int) : length of the map in cm, if length is None, then the whole map is used

                Returns:
                        map (pandas DataFrame) : dataframe with reasampled values in format x, y, z
                        
                        wholeMap (pandas DataFrame) : dataframe where first column represents the y axis and the other columns represent the x-axis
        

       
        """
        
        map = stack_map(wholeMap)
        
        if length:
                map = cut_map_len(map,start_point,length)
        
        wholeMap = map.pivot_table(index='y', columns='x', values='z')
        
        return wholeMap, map


def resample(map, sampling_distance_x, sampling_distance_y, verbose=False):
    
        """

        map : stacked dataframe with columns x, y, z
        sampling_distance_x : distance between two known points in x direction
        sampling_distance_y : distance between two known points in y direction
        
        known_points : known points
        unknown_points : known points

        if unknown points are outside the field of known points, then they are ignored

        """
        

        data = map[map['y' ]%(sampling_distance_y/2) == 0]
        # resampling on y axis
        known_points = data[data['y'] % int(sampling_distance_y) == 0]
        unknown_points = data[data['y'] % sampling_distance_y != 0]


        # resampling on x axis
        
        if sampling_distance_x != 1:
                


                known_points = known_points[known_points['x'] % int(sampling_distance_x) == 0]

                unknown_points = unknown_points[unknown_points['x'] % int(sampling_distance_x/2)  == 0 ]
                unknown_points =unknown_points[ unknown_points['x'] % int(sampling_distance_x) != 0]

        if verbose:
                print('cheking if unknown points are outside the field of known points')
                print('..')
                print(' ')

        if unknown_points['x'].max() > known_points['x'].max():
                unknown_points = unknown_points[unknown_points['x'] != unknown_points['x'].max()]

                if verbose:
                        print('there are unknown points are outside the field of known points on the x axis')

        if unknown_points['y'].max() > known_points['y'].max(): 
                unknown_points = unknown_points[unknown_points['y'] != unknown_points['y'].max()]

                if verbose:
                        print('there are unknown points are outside the field of known points on the y axis')

        return known_points, unknown_points


def randomsampling(map, len_sample):

        """
        _summary_
                Args:
                        map (pandas DataFrame) :  _description_ dataframe with all values in format x, y, z
                        len_sample (int) :  _description_ how many rows should the resampled data contain
                Returns:
                        map (pandas DataFrame) : dataframe with reasampled values in format x, y, z
        """
        
        
        # Checking inputs Args
        
        if not (len(map) >= len_sample):
                print('sample length is longer than the map length')
                return False
        
        
        # Fct begins here
        
        return map.loc[np.random.choice(map.index, size=len_sample)]
