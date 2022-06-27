import os

# Data Proceing libraries

import numpy as np
import pandas as pd
import data_preparation as dp

# Interpol Libraries

import interpolation as ip
import skgstat as skg
import scipy


# Deep Learning Libraries

import tensorflow as tf
import deepkriging as dk
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import basemodel as bm

# Plotting Libraries

import matplotlib.pyplot as plt
import seaborn as sns
import ploting_utils as pu



# Fct for Error calcualtion

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

def normalizeDeepkrigingData():
    
    
    global  StackedMapNormalizedForDK, knownPointsDK, unknownPointsDK, maxvalsDK, minvalsDK
    
    StackedMapNormalizedForDK, knownPointsDK, unknownPointsDK, maxvalsDK, minvalsDK = dk.normalize_data(StackedMap.copy(), knownPoints.copy(), unknownPoints.copy()) # minmax normalize the data
    
    
    
def getWendlandParams():
    
    global N, numBasis
    
    if interpolateWholeMap:
        
        N = len(StackedMap)
        
    else:
        
        N = 1774043 #len(knownPoints) + len(unknownPoints)
    
        
    H = dk.calc_H_for_num_basis(N) # Calculate the number of basis functions
    
    numBasis = dk.findWorkingNumBasis(N,H,verbose=True) # Calculate the number of basis functions
            
        
        
def prepareKnownPointsDK():
    
    '''
    
        Prepare known points for Deep Kriging using the wendland Kernel
    
    '''
    
    global x_trainDK, y_trainDK
    
    x_trainDK = dk.wendlandkernel(knownPointsDK[['x','y']], numBasis)
    y_trainDK = knownPoints[['z']]
    
def prepareUnknownPointsDK():
    
    """
    
        Prepare unknown points for Deep Kriging using the wendland Kernel
        Either Prepare specific unknown points or prepare the entire map
        
    """
    
    global x_valDK, y_valDK
    
    if interpolateWholeMap:
    
        x_valDK = dk.wendlandkernel(unknownPointsDK[['x','y']], numBasis)
        y_valDK = unknownPoints[['z']]
    
       
    else:
        
        x_valDK = dk.wendlandkernel(unknownPointsDK[['x','y']], numBasis)
            
        y_valDK = unknownPoints[['z']]
    
    
        
def calcErrors():
    
    global relErrors
    
    maes = {}
    mses = {}
    
    keys = list(data.keys())
    for key in keys:
        
        maes[key] = mae(unknownPoints['z'],data[key]['z'])
        mses[key] = mse(unknownPoints['z'],data[key]['z'])
    
    
    f= open(f"{path}/error.txt","a+")
    f.write(f"{len(knownPoints)} known points = {len(knownPoints)/len(StackedMap) * 100 } % of the Map, len(unkwonPoints) unknown points = {len(unknownPoints)/len(StackedMap) *100} % of the Map \n")
    f.write('\n')
    f.write(f"{scenario}\n")
    f.write('\n')
    f.write('\n')
    
    f.write('MAEs: \n')
    for key, value in maes.items(): 

        f.write('%s:%s\n' % (key, value))

    f.write('\n')
    #f.write(f"{maes}\n")
    
    f.write('MSES: \n')
    
    for key, value in mses.items(): 

        f.write('%s:%s\n' % (key, value))

    f.write('\n')
    
    
    
    ### Rel Error
    
    relErrors = {}
    
    for key in list(data.keys()):

        
        relErrorDf = pd.DataFrame([unknownPoints['x'],unknownPoints['y'] ,relativeError(unknownPoints['z'], data[key]['z'])]).T

        relErrorDf.columns = ['x','y','z']
        
        relErrors[key] = relErrorDf

def main():
    
    # Data Set UP
    
    samplingDistance_x = 22
    
    samplingDistance_y = 22 * 12
    prepareMap()
    
    
    getWendlandParams()
    
    
    normalizeDeepkrigingData()
    
    prepareKnownPointsDK()
    
    prepareUnknownPointsDK()
    
        
        
    # Interpolation
    
    ## Linear Grid Interpolation
    
    
    ResutLinearInterpolation = ip.grid_interpolation(knownPoints.copy(), unknownPoints.copy(), method='linear', verbose=verbose)
    
    ## Kriging 
    
    
    ResultKriging = ip.kriging_skg(knownPoints.copy(), unknownPoints.copy(), 10 )
    
    # Deep Learning
    
    ## Training
    
    
        
    BaseModel = bm.build_model(verbose=verbose)
   
    BaseModel, trainedModelPathBase = bm.train(knownPoints[['x', 'y']], knownPoints[['z']], unknownPoints[['x','y']], unknownPoints['z'],BaseModel, epochs, scenario,save_hist=save_hist, verbose=verbose)

    ResultBaseModel =  bm.predict(trainedModelPathBase, unknownPoints[['x','y']])
    
    
    DeepKrigingModel = dk.build_model(x_trainDK.shape[1], verbose=verbose)
    
    DeepKrigingModel, trainedModelPathDK = dk.train_model(DeepKrigingModel, x_trainDK, y_trainDK, x_valDK, y_valDK, scenario, epochs, save_hist=save_hist, verbose=verbose)
    
    DeepKrigingPrediction = dk.predict(trainedModelPathDK, x_valDK)[:,0]
    
    print(f' deepkriging before min max min val : {DeepKrigingPrediction.min()}')
    
    ResultDeepKriging = DeepKrigingPrediction#dk.reminmax(DeepKrigingPrediction, minvalsDK['z'], maxvalsDK['z'])
    print(f' deepkriging after min max min val : {ResultDeepKriging.min()}')
    ## Transform Deep Kriging Result into a pandas DataFrame with cols x, y, z
               
             
    ResultDeepKriging= pd.DataFrame([ResultDeepKriging,unknownPoints['x'].to_numpy(),unknownPoints['y'].to_numpy()]).transpose() # create a DataFrame 
                        
    ResultDeepKriging.columns = ['z','x','y'] # Fix column names
    
    ResultDeepKriging  = ResultDeepKriging[['x','y','z']]
    
    ResultDeepKriging.index = unknownPoints.index # Fix index
    
    # Base Model

    

    # Work with Results
    
    global data
    
    data = {'Linear Interpolation':ResutLinearInterpolation,'Kriging':ResultKriging,'Deep Kriging':ResultDeepKriging,'Base Model':ResultBaseModel}
    
    
    ## Create Path to store results/ figures etc..
    
    global path
    path = f'/home/schmijul/source/repos/thesis/newplots/main/{scenario}/'
    
    if not os.path.exists(path):
            os.makedirs(path)
                    
                    
                    
                    
    
    # calculate errors
    
    calcErrors()

    # Plots
    
    maxValue = 0
    
    for key in list(data.keys()):
        if data[key]['z'].max() > maxValue:
            maxValue = data[key]['z'].max()
            
    # find min value from dictionary of pandas DataFrames
    
    
    minValue = np.Inf
    for key in list(data.keys()):
        if data[key]['z'].min() < minValue:
            minValue = data[key]['z'].min()
        
    for key in list(data.keys()):   
        pu.generateHeatMaps({key:data[key]},StackedMap, knownPoints, unknownPoints, maxValue, minValue, 0,path +f'{key}.png')
        
        
    pu.generateHeatMaps(data,StackedMap, knownPoints,unknownPoints, maxValue, minValue, 1,path+'heatmaps.png')
    
    maxError = 0
    
    for key in list(relErrors.keys()):
        if relErrors[key]['z'].max() > maxError:
            maxError = relErrors[key]['z'].max()
            
    # find min value from dictionary of pandas DataFrames
    
    
    minError = np.Inf
    for key in list(data.keys()):
        if relErrors[key]['z'].min() < minError:
            minError = relErrors[key]['z'].min()
            
            
    pu.generateHeatMaps(relErrors,StackedMap, knownPoints,unknownPoints,maxError, minError, 0,path+'relErrors.png')
    

    
if __name__ == '__main__':
    
    
    interpolateWholeMap = 0
    random = 0
    length = None
    epochs = 1700
    reduceResolution = 0
    verbose = 1
    save_hist = 0
    
    for interpolateWholeMap in [0,1]:
        
        if interpolateWholeMap:
            
            reduceResolution = 1
        else:
            reduceResolution = 0
                
        
        for samplingDistance_x in [34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2]: 
                
                samplingDistance_y = samplingDistance_x * 12
                
                for random in [0,1]:
                    if random:
                            
                        scenario = f'wholeMap_x-{samplingDistance_x}_y-{int(samplingDistance_y/12)}_RandomSampling'
                                    
                    else:
                                    
                        scenario = f'wholeMap_x-{samplingDistance_x}_y-{int(samplingDistance_y/12)}_UniformSampling'
                                    
                                    
                    if interpolateWholeMap:
                        scenario = scenario + '_InterpolateAllPoints'
                    
                    if length:
                        scenario = scenario + f'_length-{length}'
                
                
                    main()
                
                
            
    