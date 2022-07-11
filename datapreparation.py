import numpy as np
import pandas as pd

def stackmap(wholemap):

    """
    _summary_
    Args: map (pandas DataFrame)

    Returns: mapStacked (pandas DataFrame) : format x, y, z

    make sure all values are float values and not strings

    """

    unstackedmap = wholemap.astype(float)
    unstackedmap = unstackedmap.set_index('Unnamed: 0').reset_index().drop(columns=['Unnamed: 0'])
    unstackedmap.columns = list(range(len(unstackedmap.columns)))
    stackedmap = unstackedmap.stack().reset_index()
    stackedmap.columns = ['y', 'x', 'z']
    stackedmap = stackedmap[['x', 'y', 'z']]

    return stackedmap

def cutmap(uncutmap,start_point,length):

    """
    _summary_
    Args:
        uncutmap (pandas DataFrame) : dataframe with all values in format x, y, z
        start_point (int) : smallest y val
        length (int) : length of the map in cm
    Returns:
                cut map
    _description
        12 steps on y axis = 1 cm

        if length = 700 (7m), then 12*700 cm are required to get to 7m
        therefore the length * 0.12 = the length in cm

    """

    # Cheking inputs Args

    if not len(uncutmap) >= start_point+length:
        print('length is longer than the map length')
        return False



    # Fct begins here


    end_point = start_point+length # Find endpoint

    maxy = end_point
    map_cut = uncutmap[uncutmap['y'] <= maxy]
    map_cut = map_cut[map_cut['y'] >= start_point*12]

    return map_cut

def preparemap(wholemap,start_point=0, length=None):

    """

    _summary_
    Args:
        wholemap (pandas DataFrame)

        start_point (int) : smallest y val
        length (int) : length of the map in cm, if length is None, then the whole map is used

    Returns:
        stackedmap (pandas DataFrame) : format x, y, z
        wholemap (pandas DataFrame)

    """

    stackedmap = stackmap(wholemap)

    if length:
        stackedmap = cutmap(stackedmap,start_point,length)


    return wholemap, stackedmap


def resample(entiremap, sampling_distance_x, sampling_distance_y, verbose=False):

    """

    _summary_

    Args:
        entiremap (pandas DataFrame) : dataframe with all values in format x, y, z
        sampling_distance_x (int) :distance between two samples in x-axis
        sampling_distance_y (int) :distance between two samples in y-axis

    Returns:
        known_points (pandas DataFrame) : dataframe with reasampled values in format x, y, z
        unknown_points (pandas DataFrame) : dataframe with reasampled values in format x, y, z

    _description_

        if unknown points are outside the field of known points, then they are ignored

    """

    data = entiremap[entiremap['y' ]%(sampling_distance_y/2) == 0]

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


def randomsampling(uniformmap, len_sample):

    """
    _summary_

        Args:
            stackedmap (pandas DataFrame) : format x, y, z
            len_sample (int) : len rows
        Returns:
            map (pandas DataFrame) : dataframe with reasampled values in format x, y, z
    """

    # Checking inputs Args

    if not len(uniformmap) >= len_sample:
        print('sample length is longer than the map length')
        return False

    # Fct begins here

    return uniformmap.loc[np.random.choice(uniformmap.index, size=len_sample)]


if __name__ == '__main__':

    whole_map = pd.read_csv(('testdata/testMap.csv'))

    map_stacked = preparemap(whole_map, start_point=0, length=700)[1]

    if map_stacked.equals(pd.read_csv('testdata/stackedmap.csv')[['x', 'y', 'z']]):
        print('from datapreparation.py: preparemap: OK')
