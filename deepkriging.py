
import pandas as pd
import numpy as np

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

def normalize_data(map, known_points, unknown_points):
    
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
                                     
    known_points['x'] = minmax(known_points['x'], maxvals['x'], minvals['x'])
    known_points['y'] = minmax(known_points['y'], maxvals['y'], minvals['y'])
    known_points['z'] = minmax(known_points['z'], maxvals['z'], minvals['z'])
    unknown_points['x'] = minmax(unknown_points['x'], maxvals['x'], minvals['x'])
    unknown_points['y'] = minmax(unknown_points['y'], maxvals['y'], minvals['y'])
    unknown_points['z'] = minmax(unknown_points['z'], maxvals['z'], minvals['z'])
   
    map.x = minmax(map.x, maxvals['x'], minvals['x'])
    map.y = minmax(map.y, maxvals['y'], minvals['y'])
    map.z = minmax(map.z, maxvals['z'], minvals['z'])
    
    return map,known_points, unknown_points, maxvals, minvals
    

    
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
    for h in range(1,int(H)+1):
            Kh = (9 * 2**(h-1) + 1 )**n_dimensions
            num_basis.append(int(Kh)**n_dimensions)
        
    #return num_basis
    return [10**2,19**2,37**2,73**2, 145**2]


def wendlandkernel(known_points, unknown_points, num_basis, verbose=False):
    
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
    
    

    # Fct begins here
    points = pd.concat([known_points, unknown_points])[['x','y']]

    N = len(points)


    x = points.x
    y = points.y
    
    
    knots_1dx = [np.linspace(0,1,int(np.sqrt(i))) for i in num_basis]
    knots_1dy = [np.linspace(0,1,int(np.sqrt(i))) for i in num_basis]


    ##Wendland kernel
   
    knots_1dx = [np.linspace(0,1,int(np.sqrt(i))) for i in num_basis]
    knots_1dy = [np.linspace(0,1,int(np.sqrt(i))) for i in num_basis]
    ##Wendland kernel
    basis_size = 0
    phi = np.zeros((N, int(sum(num_basis))))
    for res in range(len(num_basis)):
        theta = 1/np.sqrt(num_basis[res])*2.5
        knots_x, knots_y = np.meshgrid(knots_1dx[res],knots_1dy[res])
        knots = np.column_stack((knots_x.flatten(),knots_y.flatten()))
        for i in range(int(num_basis[res])):
            d = np.linalg.norm(np.vstack((y,x)).T-knots[i,:],axis=1)/theta
            for j in range(len(d)):
                if d[j] >= 0 and d[j] <= 1:
                    phi[j,i + basis_size] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
                else:
                    phi[j,i + basis_size] = 0
        basis_size = basis_size + num_basis[res]

    return pd.DataFrame(phi)





def train_val_split(phi, known_points, unknown_points, map,verbose=False):
    
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
   # Checking Input Args
    if not(len(phi) > len(known_points) or len(phi) > len(unknown_points)):
        print('Error : more known or unkown points than total points')
        return False
    
    """
    if not(len(phi) == len(map)):
        print('Error : len phi does not match len map')
        return False
    """
    

    # Fct begins here
    if verbose:
        print(f'max phi index : {phi.index.max()}')
        print(f'max known points index : {known_points.index.max()}')
        print(f'max unknown points index : {unknown_points.index.max()}')

    start_index = np.min((known_points.index[0],unknown_points.index[0]))
   

    train_idx = known_points.loc[known_points.index < phi.index.max()].index 
    val_idx = unknown_points.loc[unknown_points.index < phi.index.max()].index
    
    """
        Since phi creation startet with a concatenation of known and unknown points and the index was not reset/sorted the order was still : 
        first all known points, second all unknown points, there fore first taking the idx for known, then for second can be used here
    """
    train_idx = np.asarray([x for x in range(len(known_points))])
    val_idx = np.asarray([x+len(known_points) for x in range(len(unknown_points))])
    
    x_train = phi.loc[train_idx].to_numpy()
    y_train = known_points.z.to_numpy()
    x_val = phi.loc[val_idx].to_numpy()
    y_val = unknown_points.z.to_numpy()

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
    
    """_summary_

    Returns:
        _type_: _description_
    """    
    # Checking input Args
    if not((len(x_train) == len(y_train)) and (len(x_val) == len(y_val))):
        print('Error x and y shapes do not match')
        return False
    
    if not ((len(x_train) != 0) or (len(y_train != 0))):
        print('Error empty Data ')
        return False
    
     
    
    # Fct begins here
    
    
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
    # Checking input Args
    
    
    if not (len(x_val != 0)):
        print('Error : x-val empty')
        return False

    
   # Fct begins here 
    
    

    model = tf.keras.models.load_model(f'trainedModels/deepkriging/{name}/best_model.h5')


    return model.predict(x_val)
    
    
    
if __name__ == '__main__':
    
    """This works as a test run for debugging
    """
    verbose = True
    epochs = 100
    sampling_distance_y = 12 * 2
    sampling_distance_x =  2
    length = 150
    start_point = 0
    whole_map = pd.read_csv('WholeMap_Rounds_40_to_17.csv')
    map = stack_map(whole_map)
    map = map.loc[:10000]

    #map = cut_map_len(map,start_point,length) # cut the map to the length of the map
    
    known_points, unknown_points = resample(map, sampling_distance_x, sampling_distance_y) # resample the map
                                  
    map,known_points, unknown_points, maxvals, minvals  = normalize_data(map,known_points, unknown_points)
    
   
    N = len(known_points) + len(unknown_points) 
    num_basis = get_num_basis(N)
    
    
    phi = wendlandkernel(known_points,unknown_points, num_basis)
    
    
    
    x_train,y_train, x_val, y_val = train_val_split(phi, known_points, unknown_points,map, verbose=verbose)
    
    
    dk_model =  build_model(phi.shape[1], verbose=True)
    
    name = 'testrun'


    
    
    dk_model, dk_hist = train_model(dk_model, x_train, y_train, x_val, y_val, name,epochs, batch_size=100, verbose=verbose)
    
    dk_prediction = predict(name, x_val)[:,0]
    
    dk_prediction = reminmax(dk_prediction, maxvals['z'], minvals['z'])
    
    dk_prediction = pd.DataFrame(dk_prediction, unknown_points['x'].to_numpy(), unknown_points['y'].to_numpy())
    dk_prediction.columns = ['z', 'x', 'y']
    dk_prediction = dk_prediction[['x','y','z']]
    dk_prediction.index = unknown_points.index()
    dk_prediction.to_csv('deepkriging_prediction.csv')
    
    print(np.mean(np.abs(dk_prediction['z'] - unknown_points['z'])))