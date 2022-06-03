
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf
from data_preparation import *


def create_callback(trainedModelPath,verbose=False):
    
        """
        quick function to create callbacks and or overwrite existing callbacks
        
        """
        
        return  [
            tf.keras.callbacks.ModelCheckpoint(
                trainedModelPath+"/best_model.h5", save_best_only=True, monitor="val_loss", verbose=verbose
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            )
            
        ]
        
def minmax(x, x_max, x_min):
    return (x-x_min)/(x_max-x_min)

def reminmax(x, x_max, x_min):
    return x * (x_max-x_min) + x_min

def normalize_data(map):
    
    """
    _summary_

        Args:
            map (pandas DataFrame) : map with all points in x,y,z format
            
        Returns:
            map minmax normalized
            normvals
            
    """
    
    maxvals = {'x': map['x'].max(), 'y': map['y'].max(), 'z': map['z'].max()}
    minvals = {'x': map['x'].min(), 'y': map['y'].min(), 'z': map['z'].min()}
    
    map.x = minmax(map.x, maxvals['x'], minvals['x'])
    map.y = minmax(map.y, maxvals['y'], minvals['y'])
    map.z = minmax(map.z, maxvals['z'], minvals['z'])
    
    return map, maxvals, minvals
    

    
def get_num_basis(N, n_dimensions=2):
      
    """
      _summary_
      
        Args:
            N: number of points in the map
            n_dimensions: number of dimensions of the coordinates
        
        Returns:
            num_basis (list): list of number of basis functions for each dimension
      
    _description_
    
        This function returns a list of number of basis functions for each dimension
        as descriped in : " https://arxiv.org/pdf/2007.11972.pdf " on page 12
        
    """
      
      
    H = 1 + (np.log2( N**(1/n_dimensions) / 10 )) 
    num_basis = []
    for h in range(int(H)):
            Kh = (9 * 2*+(h-1) + 1 )**n_dimensions
            num_basis.append(Kh**n_dimensions)
        
    return num_basis

def wendlandkernel(x, y, num_basis, verbose=False):
    
    """
    _summary_
    
        Args:
            x (array): x-coordinates of point
            y (array): y-coordinates of point
            num_bais (int): number of basis functions
        Returns:
            phi (pandas DataFrame): matrix of shape N x number_of_basis_functions
        
    _description_
    
        This funkction applies the wendlandkernel to a set of points and returns a matrix of shape Nxnumber_of_basis_functions
        typicalls x and y represented all points in the entire map
        
        the code for the wendland kernel was taken from : " https://github.com/aleksada/DeepKriging/blob/master/PM25_application.py "
        
    """
    
    knots_1dx = [np.linspace(0,1,int(np.sqrt(i))) for i in num_basis]
    knots_1dy = [np.linspace(0,1,int(np.sqrt(i))) for i in num_basis]
    ##Wendland kernel
    basis_size = 0
    phi = np.zeros((N, sum(num_basis)))
    for res in range(len(num_basis)):
        theta = 1/np.sqrt(num_basis[res])*2.5
        knots_x, knots_y = np.meshgrid(knots_1dx[res],knots_1dy[res])
        knots = np.column_stack((knots_x.flatten(),knots_y.flatten()))
        for i in range(num_basis[res]):
            d = np.linalg.norm(np.vstack((y,x)).T-knots[i,:],axis=1)/theta
            for j in range(len(d)):
                if d[j] >= 0 and d[j] <= 1:
                    phi[j,i + basis_size] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
                else:
                    phi[j,i + basis_size] = 0
        basis_size = basis_size + num_basis[res]

    return pd.DataFrame(phi)





def train_val_split(phi, known_points, unknown_points, verbose=False):
    
    """
    _summary_

        Args:
            phi (pandas DataFrame): matrix of shape N x number_of_basis_functions prepared with the wendland kernel
            known_points (pandas DataFrame): array of shape N x 3 -> x, y and z values of known points
            unknown_points (pandas DataFrame): array of shape N x 3 -> x, y and vlues of unknown points
            verbose (bool): if True, print more information 
        
        Returns:
            in and output for training and validation data
    
    _description_

        This function uses all points from the map prepared by the wendland kernel.
        It will then uses the index from known and unknown points to split the data into training and validation data.
        
    """
    
    if verbose:
        print(f'max phi index : {phi.index.max()}')
        print(f'max known points index : {known_points.index.max()}')
        print(f'max unknown points index : {unknown_points.index.max()}')
        
    train_idx = known_points.loc[known_points.index < phi.index.max()].index 
    val_idx = unknown_points.iloc[unknown_points.index < phi.index.max()].index
    
    x_train = phi.loc[train_idx].to_numpy()
    y_train = known_points.z.loc[train_idx].to_numpy()
    x_val = phi.loc[val_idx].to_numpy()
    y_val = unknown_points.z.loc[val_idx].to_numpy()

    return x_train, y_train, x_val, y_val

def build_model(input_dim, verbose=False):
    
    """
    _summary_

        Args:
            input_dim (int): number of input dimensions which is the number of basis funtions -> phi.shape[1]
            verbose (bool): if True, print model summary
    
        Returns:
            Keras model ready for training
    
    """
    
    model = Sequential()
    model.add(Dense(100, input_dim = input_dim,  kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    
    
   

    model.compile(optimizer='adam'
                        , loss='mse'
                        , metrics=['mse','mae'])
    
    if verbose:
        print(model.summary())
        
    return model
 
def train_model(model, x_train, y_train, x_val, y_val, name,epochs, batch_size, verbose=False):
     
    """
     
     _summary_

        Args: 
            model: compiled Keras model object 
            x_train, x_val (pandas DataFrame): training and validation coordinates
            y_train, y_val (pandas DataFrame): training and validation values at specfific coordinates
            name (str): name of scenario -> to save model in correct directory
            epochs (int): number of epochs to train for
            batch_size (int): batch size for training data
            
            verbose (bool): if True, print more information
        
        Returns: 
            trained model
            train-history as pandas DataFrame
    
    """
    trainedModelPath = f'trainedModels/deepkriging/{name}/'
     
     
    if not os.path.exists(trainedModelPath):
                                os.makedirs(trainedModelPath)
                    
        
        
    history = model.fit(x_train,y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        callbacks = create_callback(trainedModelPath),
                        verbose=verbose)
    
    return model, pd.DataFrame(history.history)

     
     

def predict(name, x_val):
    
    
    """
    _summary_

        Args:
            name (str): name of scenario this is used to load the best model for the choosen scenario
            x_val (pandas DataFrame): validation coordinates
            
            Returns:
                predicted values at validation coordinates
        
    """
    model = tf.keras.models.load_model(f'trainedModels/deepkriging/{name}/best_model.h5')


    return model.predict(x_val)
    
    
    
if __name__ == '__main__':
    
    print(' Test run')
    print(' ')
    
    sampling_distance_y = 12 *4
    sampling_distance_x =  4
    whole_map = pd.read_csv('WholeMap_Rounds_40_to_17.csv')
    map = stack_map(whole_map) 
    
    map, maxvals, minvals  = normalize_data(map)
    print(map.head())
    
    N = len(map)
    num_basis= get_num_basis(N)
    
    phi = wendlandkernel(map.x, map.y, num_basis)
    
    print(phi.shape)
    