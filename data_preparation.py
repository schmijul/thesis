import numpy as np
import pandas as pd

def stack_map(map):

        """

        make sure all values are float values and not strings
        Assumption : first column is y, all other columns are x

        """
       
        map = map.astype(float).set_index('Unnamed: 0')
        map = map.reset_index().drop(columns=['Unnamed: 0'])
        map.columns = [x for x in range(len(map.columns))]

        
        map_stacked = map.stack().reset_index()
        map_stacked.columns = ['y', 'x', 'z']
        map_stacked = map_stacked[['x', 'y', 'z']]

        return map_stacked

def cut_map_len(map,tart_point,length):

        """

        map is supposed to be a stacked dataframe with columns x, y, z
        start_point : smallest y val
        length is the wanted length of the map in cm

        12 steps on y axis = 1 cm

        if length = 700 (7m), then 12*700 cm are required to get to 7m 
        therefore the length * 0.12 = the length in cm 
        
        """
        length = length * 0.1
        max_y = length *12

        map = map[map['y'] <= max_y]
        
        map = map[map['y']> start_point]
        return

def resample(map, sampling_distance_x, sampling_distance_y, verbose=False):
    
        """

        map : stacked dataframe with columns x, y, z
        sampling_distance_x : distance between two known points in x direction
        sampling_distance_y : distance between two known points in y direction
        
        data_i : known points
        data_o : known points

        if unknown points are outside the field of known points, then they are ignored

        """
        

        data = map[map['y' ]%(sampling_distance_y/2) == 0]
        # resampling on y axis
        data_i = data[data['y'] % int(sampling_distance_y) == 0]
        data_o = data[data['y'] % sampling_distance_y != 0]


        # resampling on x axis
        
        if sampling_distance_x != 1:
                


                data_i = data_i[data_i['x'] % int(sampling_distance_x) == 0]

                data_o = data_o[data_o['x'] % int(sampling_distance_x/2)  == 0 ]
                data_o =data_o[ data_o['x'] % int(sampling_distance_x) != 0]

        if verbose:
                print('cheking if unknown points are outside the field of known points')
                print('..')
                print(' ')

        if data_o['x'].max() > data_i['x'].max():
                data_o = data_o[data_o['x'] != data_o['x'].max()]

                if verbose:
                        print('there are unknown points are outside the field of known points on the x axis')

        if data_o['y'].max() > data_i['y'].max(): 
                data_o = data_o[data_o['y'] != data_o['y'].max()]

                if verbose:
                        print('there are unknown points are outside the field of known points on the y axis')

        return data_i, data_o


def randomsampling(map,min_x, max_x, min_y, max_y, sampling_distance_x, sampling_distance_y, data_o,verbose=False):

        """

        map : dataframe with all values in format x, y, z
        min_x : min  x value  coordinates are allowed to take
        max_x : max  x values  coordinates are allowed to take
        min_y : min  y value  coordinates are allowed to take
        max_y : max  y values  coordinates are allowed to takedata_o : length reference
        sampling_distance_x : serves as a x-stepsize indicator to calculate the wished amout of total rows for a comparisson with NonrandomResampling
        sampling_distance_y : serves as a y-stepsize indicator to calculate the wished amout of total rows for a comparisson with NonrandomResampling

        This is still lazy code as the while loop is very inefficient, but for now it works

        """
        
        
        
        amount_x_vals = len([x for x in range(int(sampling_distance_x/2),max_x+1,sampling_distance_x)])
       
        amount_y_vals = len([x for x in range(int(sampling_distance_y/2),max_y+1,sampling_distance_y)])
       
        numrows = amount_x_vals * amount_y_vals
        
        x_vals = np.random.randint(min_x, max_x,int(amount_x_vals))

        y_vals = np.random.randint(min_y, max_y,int(amount_y_vals))
        
        
        sample_map = map.copy().query("x in @x_vals")
        sample_map = sample_map.query("y in @y_vals")
       
        while len(sample_map) <= numrows:
                
                if verbose:
                        
                        print(f'len sample map != numrows')
                        print(f'len sample map: {len(sample_map)}')
                        print(f'numrows: {numrows}')
                        
                
                x_vals = np.random.randint(min_x, max_x,int(1))

                y_vals = np.random.randint(min_y, max_y,int(1))
        
                next_sample_map = map.copy().query("x in @x_vals")
                next_sample_map = next_sample_map.query("y in @y_vals")
                
                sample_map = pd.concat([sample_map,next_sample_map]).drop_duplicates(keep='first')
        
        
        
        while len(sample_map) <= numrows:
                
                if verbose:
                        
                        print(f'len sample map != numrows')
                        print(f'len sample map: {len(sample_map)}')
                        print(f'numrows: {numrows}')
                        
                
                x_vals = np.random.randint(min_x, max_x,int(1))

                y_vals = np.random.randint(min_y, max_y,int(1))
        
                next_sample_map = map.copy().query("x in @x_vals")
                next_sample_map = next_sample_map.query("y in @y_vals")
                
                sample_map = pd.concat([sample_map,next_sample_map]).drop_duplicates(keep='first')
        
        if len(sample_map) > numrows:

                sample_map=sample_map.iloc[:numrows]
        return sample_map

