import re
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from pykrige.uk import UniversalKriging
from pykrige.ok import OrdinaryKriging
import os
import skgstat as skg



def grid_interpolation(data_i, data_o,method='linear', verbose=False):

        """
        
        data_i : known points
        data_o : unknown points
        method  ['linear', 'nearest', 'cubic']
        fillnans : if True, then NaN values are replaced with 0

        """

        x = data_o['x']
        y = data_o['y']
        if verbose:
            print(f' unique x vals : {x.unique()}')
            print(' ')
            print(f' unqique y vals : {y.unique()}')
        stepsize_x = x.unique()[1] - x.unique()[0]
        stepsize_y = y.unique()[1] - y.unique()[0]
        
        if verbose:
                print('stepsize_x = ', stepsize_x)
                print('stepsize_y = ', stepsize_y)
                
        xx, yy = np.mgrid[x.min():x.max()+1:stepsize_x, y.min():y.max()+1:stepsize_y]

        if verbose:
                print(f'created grid with shape {xx.shape}')
                print(f'grid : xx min = {xx.min()}, xx max = {xx.max()}, stepsize_x = {stepsize_x}')
                print(f'grid : yy min = {yy.min()}, yy max = {yy.max()}, stepsize_y = {stepsize_y}')

        
        lin_interpol = griddata(data_i[['x', 'y']], # Points we know
                        data_i['z'], # Values we know
                        (xx.T, yy.T), # Points to interpolate
                        method=method)

        if verbose:
                print(f'interpolated grid with shape {lin_interpol.shape}')
        lin_interpol = pd.DataFrame(lin_interpol)

        
        
        lin_interpol.columns = data_o.pivot_table(index='y', columns='x', values='z').columns

        lin_interpol.index = data_o.pivot_table(index='y', columns='x', values='z').index
        
        if verbose:
                print(f'interpolated grid with shape {lin_interpol.shape}')
                print(f'lin interpol stacked shape : {lin_interpol.stack().reset_index().shape}')
                print(f'data_o shape : {data_o.shape}')

        lin_interpol_stacked = lin_interpol.stack(dropna=False).reset_index()
        lin_interpol_stacked.index = data_o.index
        
        lin_interpol_stacked.columns = ['y', 'x', 'z']
        lin_interpol_stacked = lin_interpol_stacked[['x', 'y', 'z']]

        return lin_interpol, lin_interpol_stacked

def kriging_pykrige(data_i, data_o, type='UK', variogram_model='linear', verbose=False):
    
            """
            
            data_i : known points
            data_o : unknown points
            type : 'UK' or 'OK'
                    UK : UniversalKriging 
                    OK : OrdinaryKriging 
            method : Variogram type

            """

            if type == 'UK':
                K = UniversalKriging(
                                        data_i['x'].astype(float),
                                        data_i['y'],
                                        data_i['z'],
                                        variogram_model=variogram_model,
                                        drift_terms=["regional_linear"],
                                        verbose=verbose
          
                                  )
            if type == 'OK':    
                K = OrdinaryKriging(
                                        data_i['x'].astype(float),
                                        data_i['y'],
                                        data_i['z'],
                                        variogram_model=variogram_model,
                                        verbose=verbose

                                    )


            kriging_interpol, kriging_variance = K.execute('points', data_o['x'].astype(float), data_o['y'])

            kriging_interpol = pd.DataFrame(kriging_interpol) # convert numpy array to pandas dataframe
            kriging_interpol.index = data_o.index # match idx for merging
            kriging_interpol_stacked = pd.merge(data_o[['x','y']], kriging_interpol, how='inner', left_index=True, right_index=True)
            kriging_interpol_stacked.columns = ['x', 'y', 'z']
            kriging_interpol = kriging_interpol_stacked.pivot_table(index='y', columns='x', values='z')
            if verbose:
                print(f' kriging_matrix shape : {kriging_interpol.shape}')
            return kriging_interpol, kriging_interpol_stacked

def kriging_skg(data_i, data_o, mpoints, test_max_points=False, verbose=False):
        
        """
        
        data_i : known points
        data_o : unknown points
        mpoints : maximum number of points to use in the kriging
        test_max_points : if True, then the maximum number of points is tested and the interpolation with best max pints param is returned
        
        """

        V = skg.Variogram(data_i[['x', 'y']], data_i['z'], n_lags=10)
        
        
        
        if test_max_points:
                
                if verbose:
                        print('testing max_points')
                mse = np.Inf
                
                for max_points in [8, 10, 20, 30, 40, 50, 80]:
                        ok = skg.OrdinaryKriging(V, mode='exact', max_points=max_points)  
                        
                        skg = ok.transform(data_o[['x', 'y']]) # Kriging interpolation
        
        
        
                        skg = pd.DataFrame([skg,data_o['x'].to_numpy(),data_o['y'].to_numpy()]).transpose() # create a DataFrame 
                        skg.columns = ['z','x','y'] # Fix column names
                        skg   = skg_stacked[['x','y','z']]
                        skg.index = data_o.index # Fix index
                        
                        mse_mp = np.mean((skg.z - data_o.z)**2) # calc mse to see how max points paramter affects the interpolation
                        
                        if  mse_mp < mse : # Find best interpolation
                                mse = mse_mp
                                skg_stacked = skg
        else:
                ok = skg.OrdinaryKriging(V, mode='exact', max_points=mpoints)  
                        
                skg = ok.transform(data_o[['x', 'y']]) # Kriging interpolation   
                skg = pd.DataFrame([skg,data_o['x'].to_numpy(),data_o['y'].to_numpy()]).transpose() # create a DataFrame 
                skg.columns = ['z','x','y'] # Fix column names
                skg   = skg_stacked[['x','y','z']]
                skg.index = data_o.index # Fix index
                skg_stacked = skg
                
                
                       
        skg_matrix = skg_stacked.pivot_table(index='y', columns='x', values='z')
        
        if verbose:
            print(f' kriging_matrix shape : {skg_estimate.shape}')
            
        return skg_matrix, skg_stacked


def plot_interpolpoints_on_whole_map(map,stacked_interpolation_matrix):
    
    """
    
    map : pandas dataframe with columns x,y,z and all points
    stacked_interpolation_matrix : pandas dataframe with columns x,y,z and all interpolated points
    
    """
    
    
    map_for_whole_plot = map.copy() # Use a copy of the map to avoid overwritting the original
    map_for_whole_plot['z'] = 0 # Set all z values to 0
    map_for_whole_plot.loc[stacked_interpolation_matrix.index] = np.nan

    stacked_interpolation_matrix_with_whole_map_stacked = map_for_whole_plot.combine_first(stacked_interpolation_matrix)
    
    stacked_interpolation_matrix_with_whole_map = stacked_interpolation_matrix_with_whole_map_stacked.pivot_table(index='y', columns='x', values='z')
    
    return stacked_interpolation_matrix_with_whole_map, stacked_interpolation_matrix_with_whole_map_stacked
        

