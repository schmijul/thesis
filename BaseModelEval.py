import basemodel as bm
import data_preparation as dp
import numpy as np   
import pandas as pd

import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd

 

"""
    
    
    This Script is supossed to see how much the amount of trainable parameters influences the accuracy of the model
    
    
"""
    
def mae(y_true, y_pred):
        
    '''
    _summary_:
                Args:
                    y_true (np.array): True values in format Nx1
                    y_pred (np.array): Predicted values in format Nx1
                Returns:
                    mae (float): Mean absolute error
    
    '''
    
    return np.mean(np.abs(y_true - y_pred).dropna())


def mse(y_true, y_pred):
    
    '''
    _summary_:
                Args:
                    y_true (np.array): True values in format Nx1
                    y_pred (np.array): Predicted values in format Nx1
                Returns:
                    mse (float): Mean squared error
    
    '''
    
    return np.mean(np.square((y_true - y_pred).dropna()))


def relativeError(y_true, y_pred):
    
    '''
    _summary_:
                Args:
                    y_true (np.array): True values in format Nx1
                    y_pred (np.array): Predicted values in format Nx1
                Returns:
                    relativeError (float): Relative error ( abs and percentage)
    
    '''
    
    return np.abs(y_true - y_pred).dropna()/np.abs(y_true.dropna())


    

def prepareMap():
    
    global Map, StackedMap, knownPoints, unknownPoints, maxvals, minvals

    
    wholeMap = pd.read_csv('WholeMap_Rounds_40_to_17.csv')
    
    
    global Map, StackedMap
    
    global knownPoints, unknownPoints
    
    if reduceResolution:
        #wholeMap = wholeMap.iloc[int(len(wholeMap)/2):]
        wholeMap = wholeMap.iloc[:int(len(wholeMap)/2)]
        wholeMap = wholeMap.iloc[::12,:]
        
    Map, StackedMap = dp.prepare_map(wholeMap.copy(), length=length)
    
           
        
    knownPoints, unknownPoints = dp.resample(StackedMap.copy(), samplingDistance_x, samplingDistance_y)
    
    
    if random:
        
        """
         If random, then the unknown points are randomly selected from the stacked map.
         The unknown points remain the same
         The amount of known points should match the amount of known points in uniform resampling
        """
        
        
        knownPoints = dp.randomsampling(StackedMap.copy(), len(knownPoints))

    if interpolateWholeMap:
        
        unknownPoints = StackedMap.copy()

   
    
epochs = 1700
length=None
verbose=1
reduceResolution = 0
interpolateWholeMap = 0
random = 0
samplingDistance_x = 4
samplingDistance_y = samplingDistance_x * 12

scenario = 'baseModelEval'
prepareMap()
maes = []

unitList = [x*100 for x in range(1,15)]
for units in unitList:
    
    
       
    BaseModel = bm.build_model(units, verbose=verbose)
   
    BaseModel, trainedModelPathBase = bm.train(knownPoints[['x', 'y']], knownPoints[['z']], unknownPoints[['x','y']], unknownPoints['z'],BaseModel, epochs, scenario,save_hist=0, verbose=verbose)

    ResultBaseModel =  bm.predict(trainedModelPathBase, unknownPoints[['x','y']])
    
    maes.append(mae(ResultBaseModel, unknownPoints['z']))
    
np.save('maesBaseModel.npy', maes)

import matplotlib.pyplot as plt


plt.figure(figsize=(12,10))

plt.plot(unitList,maes)
plt.xlabel('units per dense Layer')
plt.ylabel('MAE')
plt.title('MAE vs. units per dense Layer for Base Model')
plt.savefig('maeBaseModel.png')