
import numpy as np
import pandas as pd
import skgstat as skg
from scipy.interpolate import griddata





def gridinterpolation(knownpoints, unknownpoints, method='linear', verbose=False):

    """
    _summary_
        Args:
            knownpoints (pandas DataFrame) : known points in format x, y, z
            unknownpoints (pandas DataFrame) : unknown points in format x, y, z
            method (str) : interpolation method [linear, cubic, etc ..]


    _description_

            This function will execute scipy.interpolate.griddata
            on the knownpoints and unknownpoints dataframes.

    """

    ###########################
    ### Checking input Args ###
    ###########################


    if not( (knownpoints.columns.tolist() == ['x', 'y', 'z']) or (unknownpoints.columns.tolist() == ['x', 'y', 'z']) ):
        print('Error: knownpoints and unknownpoints must have columns x, y, z')
        return False

    ##############################
    ### starting interpolation ###
    ##############################

    x_values = unknownpoints['x']
    y_values = unknownpoints['y']

    if verbose:
        print(f' unique x vals : {x_values.unique()}')
        print(' ')
        print(f' unqique y vals : {y_values.unique()}')

    stepsize_x = x_values.unique()[1] - x_values.unique()[0]
    stepsize_y = y_values.unique()[1] - y_values.unique()[0]

    if verbose:
        print('stepsize_x = ', stepsize_x)
        print('stepsize_y = ', stepsize_y)

    xx_grid, yy_grid = np.mgrid[x_values.min():x_values.max()+1:stepsize_x,
                      y_values.min():y_values.max()+1:stepsize_y]

    if verbose:
        print(f'created grid with shape {xx_grid.shape}')
        print(f'xx_grid min = {xx_grid.min()}')
        print(f'xx_grid max = {xx_grid.max()}')
        print(f'stepsize_x = {stepsize_x}')
        print(f'yy_grid min = {yy_grid.min()}')
        print(f'yy_grid max = {yy_grid.max()}')
        print(f'stepsize_y = {stepsize_y}')

    lin_interpol = griddata(knownpoints[['x', 'y']], # Coordinates we know
                    knownpoints['z'], # Values we know
                    (xx_grid.T, yy_grid.T), # Points to interpolate
                    method=method)

    if verbose:
        print(f'interpolated grid with shape {lin_interpol.shape}')

    lin_interpol = pd.DataFrame(lin_interpol)


    lin_interpol.columns = unknownpoints.pivot_table(index='y', columns='x', values='z').columns

    lin_interpol.index = unknownpoints.pivot_table(index='y', columns='x', values='z').index

    if verbose:
        print(f'interpolated grid with shape {lin_interpol.shape}')
        print(f'lin interpol stacked shape : {lin_interpol.stack().reset_index().shape}')
        print(f'unknownpoints shape : {unknownpoints.shape}')

    lin_interpol_stacked = lin_interpol.stack(dropna=False).reset_index()
    lin_interpol_stacked.index = unknownpoints.index

    lin_interpol_stacked.columns = ['y', 'x', 'z']
    lin_interpol_stacked = lin_interpol_stacked[['x', 'y', 'z']]

    # stack lin_interpol array

    return lin_interpol_stacked#lin_interpol[lin_interpol.columns[2]].to_numpy()

def kriging(knownpoints, unknownpoints, maxpoints=10, test_maxpoints=False, verbose=False):

    """
    _summary_
    Args:
        knownpoints (pandas DataFrame) : known points in format x, y, z

        unknownpoints (pandas DataFrame) : unknown points in format x, y, z

        maxpoints (int) :   max number of points the kriging model
                            is allowed to consider for interpolation of one point

        test_maxpoints (bool) : if True, the kriging model is tested
                                with a range of maxpoints values and the
                                best one is chosen

    Returns:

        skg_stacked (pandas DataFrame) :    stacked dataframe with interpolated
                                            values and coordinates in format x, y, z
    """



    # Checking input Args

    if not((knownpoints.columns.tolist() == ['x', 'y', 'z']) or (unknownpoints.columns.tolist() == ['x', 'y', 'z'])):
        print('Error: knownpoints and unknownpoints must have columns x, y, z')
        return False

    variogram = skg.Variogram(knownpoints[['x', 'y']], knownpoints['z'], n_lags=10)

    if test_maxpoints:

        if verbose:
            print('testing maxpoints')
            mse = np.Inf

        for maxpoints in [8, 10, 20, 30, 40, 50, 80]:
            ordinarykriging_obj = skg.OrdinaryKriging(variogram, mode='exact', max_points=maxpoints)

            # Kriging interpolation
            skg_result = ordinarykriging_obj.transform(unknownpoints[['x', 'y']])

            # create a DataFrame
            skg_result= pd.DataFrame([skg_result,unknownpoints['x'].to_numpy(),
                                      unknownpoints['y'].to_numpy()]).transpose()
            skg_result.columns = ['z','x','y'] # Fix column names
            skg_result  = skg_result[['x','y','z']]
            skg_result.index = unknownpoints.index # Fix index

            # See how max points paramter affects the interpolation
            mse_mp = np.mean((skg_result.z - unknownpoints.z)**2)

            if  mse_mp < mse : # Find best interpolation
                mse = mse_mp
                skg_stacked = skg
    else:
        ordinarykriging_obj = skg.OrdinaryKriging(variogram, mode='exact', max_points=maxpoints)

        # Kriging interpolation
        skg_result= ordinarykriging_obj.transform(unknownpoints[['x', 'y']])

        # create a DataFrame
        skg_result= pd.DataFrame([skg_result,unknownpoints['x'].to_numpy(),
                                  unknownpoints['y'].to_numpy()]).transpose()

        skg_result.columns = ['z','x','y'] # Fix column names
        skg_result  = skg_result[['x','y','z']]
        skg_result.index = unknownpoints.index # Fix index
        skg_stacked = skg_result

    return  skg_stacked
