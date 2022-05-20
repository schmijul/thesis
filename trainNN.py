
import seaborn as sns
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error 
debug = True
import os
import pandas as pd
import tensorflow as tf
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.gaussian_process import GaussianProcess
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from data_preparation import *
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6)
import pylab 
import numpy as np
import pandas as pd
import skgstat as skg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from skgstat.models import spherical
from scipy.linalg import solve
from sklearn.metrics import mean_squared_error as mse

def rel_error(pred, true):
    return np.abs(pred - true) / np.abs(true)


def create_callback(trainedModelPath,verbose=False):

    """
    quick function to create callbacks and or overwrite existing callbacks
    
    """
    
    return  [
        keras.callbacks.ModelCheckpoint(
            trainedModelPath+"/best_model.h5", save_best_only=True, monitor="val_loss", verbose=verbose
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        )
        
    ]
def build_model(x_train, verbose=False):

    """

    lazy Code to build Sequential model
    x_train : array of training data to get right input shape
    
    """
    model = Sequential()
    model.add(Dense(100, input_dim=x_train.shape[1],  kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(100, activation='relu'))
                #model.add(Dropout(rate=0.5))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='mse', optimizer=opt, metrics=['mse','accuracy'])
    if verbose:
            print(model.summary())


    return model



def train(x_train, y_train, x_val, y_val,length,model, epochs,sampling_distance_x, sampling_distance_y, save_hist=True,  verbose=False, batch_size=100):

        """
        train model 

        x_train : known points ( coordinates)
        y_train : known points ( values)

        x_val : unknown points ( coordinates)
        y_val : unknown points ( values)

        model : Kera Model
        epochs : number of epochs
        batch_size : batch size

        
        """

        trainedModelPath = f'trainedModels/NonRandomResapling/samplerate_x{sampling_distance_x}_y{sampling_distance_y}_length{length}/'
         

        if not os.path.exists(trainedModelPath):
                            os.makedirs(trainedModelPath)
                


        callbacks = create_callback(trainedModelPath)

        x_train = x_train.to_numpy()
        x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)

        y_train = y_train.to_numpy()
        y_train = y_train.reshape(y_train.shape[0],1)

        x_val= x_val.to_numpy()
        x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
        y_val= y_val.to_numpy()
        y_val = y_val.reshape(y_val.shape[0],1)

        history = model.fit(x_train
                            , y_train
                            , epochs=epochs
                            , validation_data = (x_val, y_val)
                            , callbacks=callbacks
                            , batch_size=batch_size
                            , verbose=verbose
                                )

        if save_hist:
             pd.DataFrame(history.history).to_csv(f'{trainedModelPath}/history_x{sampling_distance_x}_y{sampling_distance_y}.csv')

        return model

def prediction(model,data_o, x_val, sampling_distance_x, sampling_distance_y):
    """

    Predict the output of the model on the validation set.
    model : pretrained Kera Model
    x_val : validation set input

    """
    prediction = model.predict(x_val)

    np.save(f'predictions/NN_x{sampling_distance_x}_y{sampling_distance_y}.npy', prediction)

    pred_stacked = pd.DataFrame([prediction[:,0],data_o['x'].to_numpy(),data_o['y'].to_numpy()]).transpose()
    pred_stacked.columns = ['z','x','y']

    pred_stacked = pred_stacked[['x','y','z']]
    pred_stacked.index = data_o.index

    prediction = pred_stacked.pivot_table(index='y', columns='x', values='z')
    return prediction, pred_stacked