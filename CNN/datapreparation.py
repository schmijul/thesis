import math
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


def randomsampling(uniformmap, len_sample, dist, include_corners=True):

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
    
    # Set np random seed to dist:
    print(len_sample)
    np.random.seed(dist)

    if include_corners:

        # First collect all the corner points
        corner_point_1 = uniformmap[uniformmap.x == uniformmap.x.min() ][ uniformmap.y ==  uniformmap.y.min()]
        corner_point_2 = uniformmap[uniformmap.x == uniformmap.x.min() ][ uniformmap.y ==  uniformmap.y.max()]
        corner_point_3 = uniformmap[uniformmap.x == uniformmap.x.max() ][ uniformmap.y ==  uniformmap.y.min()]
        corner_point_4 = uniformmap[uniformmap.x == uniformmap.x.max() ][ uniformmap.y ==  uniformmap.y.max()]

        corner_points = pd.concat([corner_point_1,
                           corner_point_2,
                           corner_point_3,
                           corner_point_4])

        # Drop corner_points from uniformmap
        uniformmap = uniformmap.drop(corner_points.index)
        print(len_sample)

        # Randomsampling on rest of uniformmap
        randommap = uniformmap.loc[np.random.choice(uniformmap.index, size=len_sample-4)]

        # Add corner_points to randommap

        randommap= randommap.merge(corner_points, how='outer')

    else:
        randommap = uniformmap.loc[np.random.choice(uniformmap.index, size=len_sample)]
        print(randommap.index)
        print(uniformmap.index)
    return randommap



def minmax(array, array_max, array_min):

    """
    _summary_
    Args:
        array ([np.array or pandas df]):
        array_max ([float]):maxval
        array_min ([float]): minval

    Returns:
        [np.array or pandas df]: 0-1, minmax normalized array
    """

    return (array-array_min)/(array_max-array_min)

def reminmax(array, array_max, array_min):

    """
    _summary_
    Args:
        array ([np.array or pandas df]):
        array_max ([float]):maxval
        array_min ([float]): minval

    Returns:
        [np.array or pandas df]: 0-1, minmax renormalized array
    """

    return array * (array_max-array_min) + array_min

def normalize_data(map_not_normalized, known_points, unknown_points):

    """

    _summary_

        Args:
            map_not_normalized (pandas DataFrame) : map with all points in x,y,z format

        Returns:
            map minmax normalized
            normvals

    """

    maxvals = {'x': map_not_normalized['x'].max(),
               'y': map_not_normalized['y'].max(),
               'z': map_not_normalized['z'].max()}

    minvals = {'x': map_not_normalized['x'].min(),
               'y': map_not_normalized['y'].min(),
               'z': map_not_normalized['z'].min()}

    map_normalized = pd.DataFrame()
    for col in known_points.columns:

        known_points[col] = minmax(known_points[col], maxvals[col], minvals[col])
        unknown_points[col] = minmax(unknown_points[col], maxvals[col], minvals[col])
        map_normalized[col] = minmax(map_not_normalized[col], maxvals[col], minvals[col])

    return map_normalized,known_points, unknown_points, maxvals, minvals


def calc_h_for_num_basis(number_points, n_dimensions=2, verbose=False):


    """
        _summary_

        Args:
            number_points (int) : number of points
            n_dimensions (int, optional) : Dimensions of points  Defaults to 2 (x,y).
            verbose(bool): show H

        Returns:
            h_for_num_basis (float

        _description_
            This Function calculates the Number of Elements in num_basis)

    """

    h_for_num_basis = 1 + (np.log2( number_points**(1/n_dimensions) / 10 ))

    if verbose:
        print("H: ", h_for_num_basis)

    return math.ceil(h_for_num_basis)


def get_numbasis(num_elements , n_dimensions=2, verbose=False):

    """
      _summary_

            Args:
                N (int) : number of points
                num_elements (float) : ( Number of Elements in num_basis)
                n_dimensions: number of dimensions of the coordinates

            Returns:
                num_basis (list): list of number of basis functions for each dimension

    _description_

            This function returns a list of number of basis functions for each dimension
            as descriped in : " https://arxiv.org/pdf/2007.11972.pdf " on page 12



    """


    numbasis = []

    for i in range(1,int(num_elements+1)):
        k = (9 * 2**(i-1) + 1 )
        numbasis.append(int(k)**n_dimensions)

    if verbose:
        print(f"amount base fct: {sum(numbasis)}")
    return numbasis


def findworkingnumbasis(len_data, num_elements, n_dimensions=2, verbose=False):

    """
    _summary_

        Args:
            len_data (int) : number of points
            num_elements (float) : ( Number of Elements in num_basis)
            n_dimensions (int, optional) : Dimensions of points  Defaults to 2 (x,y).
            verbose (bool): show info about recursion

        Returns:
            numbasis (list): list of number of basis functions for each dimension

    _description_
            Because the number of basis functions will become very high for a hugh Data set,
            this function will check recursively,
            if the number of basis functions is too high
            this fct will decrease the number of basis functions
            until it is below the maximum number of basis functions.

    """

    # Create ArrayMemoryError class to catch a custom exception
    # ArrayMemoryError is a numpy specific exception)
    class ArrayMemoryError(Exception):
        pass

    try:
        numbasis = get_numbasis( num_elements, n_dimensions, verbose)

        testvariable = np.zeros((len_data, int(sum(numbasis))),dtype=np.float16)
        del testvariable    # delete testvariable to free up memory



    except np.core._exceptions._ArrayMemoryError:
        if verbose:
            print('Error : Not enough memory to create basis functions')
            print('try to reduce H by 1')
            
            print(f"exception : { np.core._exceptions._ArrayMemoryError}")

            numbasis = findworkingnumbasis(len_data, (num_elements-1) , n_dimensions=2)

    return numbasis

def wendlandkernel(points, numbasis):

    """
    _summary_

        Args:
            points (pandas DataFrame) : cordinnates in format x, y
            numbasis (int): number of basis functions
        Returns:
            phi (pandas DataFrame): matrix of shape N x number_of_basis_functions

    _description_

        This funkction applies the wendlandkernel to a set of points
        and returns a matrix of shape Nxnumber_of_basis_functions
        typicalls x and y represented all points in the entire map

    """

    # Fct begins here




    knots_1dx = [np.linspace(0,1,int(np.sqrt(i))) for i in numbasis]
    knots_1dy = [np.linspace(0,1,int(np.sqrt(i))) for i in numbasis]

    ##Wendland kernel

    basis_size = 0

    # Create matrix of shape N x number_of_basis_functions
    # Use np.float32 to save memory

    phi = np.zeros((len(points), int(sum(numbasis))),dtype=np.float32)



    for res in range(len(numbasis)):

        theta = 1/np.sqrt(numbasis[res])*2.5
        knots_x, knots_y = np.meshgrid(knots_1dx[res],knots_1dy[res])
        knots = np.column_stack((knots_x.flatten(),knots_y.flatten()))

        for i in range(int(numbasis[res])):

            d = np.linalg.norm(np.vstack((points.x, points.y)).T-knots[i,:],axis=1)/theta

            for j in range(len(d)):

                if d[j] >= 0 and d[j] <= 1:

                    phi[j,i + basis_size] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3

                else:

                    phi[j,i + basis_size] = 0

        basis_size = basis_size + numbasis[res]

    return pd.DataFrame(phi)
