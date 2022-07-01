import os
import re

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
def batchWisePrediction(pathToBestModel,stackedMapNormalizedForDK, ):
    
    x_test = stackedMapNormalizedForDK[['x','y']]

    prediction = np.zeros(len(x_test))

    H = dk.calc_H_for_num_basis(len(x_test))

    numBasis = dk.get_numBasis(H)

    for i in range(100,stackedMapNormalizedForDK.shape[0],100):
        
        testSlice = x_test.iloc[i-100:i-1] # stackedMapNormalizedForDK x and y coordinates are both 0-1 min max normalized
        
        
        testSlice = dk.wendlandkernel(testSlice, numBasis) # Apply Wendlandkernel to subsample of data ( as test slice)
        
        
        
        prediction[i-100:i-1] = dk.predict(pathToBestModel, testSlice)[:,0] # Predict and overwrite zeros in 

    return prediction

def mae(y_true, y_pred):
    
    '''
    _summary_:
                Args:
                    y_true (np.array): True values in format Nx1
                    y_pred (np.array): Predicted values in format Nx1
                Returns:
                    mae (float): Mean absolute error
    
    '''
    
    error = np.abs(y_true - y_pred)
    
    errorNoNan = error.dropna()
    
    if len(error) - len(errorNoNan) > 0:
        print('Warning: There are NaN values in the error vector')
        print('Number of NaN values: ', len(error) - len(errorNoNan))
        
    return np.mean(errorNoNan)
    
    

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
    
    global Map, stackedMap, knownPoints, unknownPoints, maxvals, minvals

    
    ##wholeMap = pd.read_csv('WholeMap_Rounds_40_to_17.csv')
    
    wholeMap = pd.read_csv('RadioEnvMaps/Back_Straight_SISO_Power_Map.csv').iloc[:30 ]
    wholeMap = wholeMap.transpose().iloc[:30].transpose()
    
    """
    if reduceResolution:
        #wholeMap = wholeMap.iloc[int(len(wholeMap)/2):]
        wholeMap = wholeMap.iloc[:int(len(wholeMap)/2)]
        wholeMap = wholeMap.iloc[::12,:]
    """
        
    Map, stackedMap = dp.prepare_map(wholeMap.copy(), length=length)
    print(Map.shape)
    print(stackedMap.shape)
           
        
    knownPoints, unknownPoints = dp.resample(stackedMap.copy(), samplingDistance_x, samplingDistance_y)
    
    
    if random:
        
        """
         If random, then the unknown points are randomly selected from the stacked map.
         The unknown points remain the same
         The amount of known points should match the amount of known points in uniform resampling
        """
        
        
        knownPoints = dp.randomsampling(stackedMap.copy(), len(knownPoints))
    """
    if interpolateWholeMap:
        
        unknownPoints = stackedMap.copy()
    """

def normalizeDeepkrigingData():
    
    
    global  stackedMapNormalizedForDK, knownPointsDK, unknownPointsDK, maxvalsDK, minvalsDK
    
    stackedMapNormalizedForDK, knownPointsDK, unknownPointsDK, maxvalsDK, minvalsDK = dk.normalize_data(stackedMap.copy(), knownPoints.copy(), unknownPoints.copy()) # minmax normalize the data
    
    
    
def getWendlandParams():
    
    global N, numBasis
    
    if interpolateWholeMap:
        
        N = len(stackedMap)
        
    else:
        
        N = 1774043 #len(knownPoints) + len(unknownPoints)
            
        
    H = dk.calc_H_for_num_basis(N) # Calculate the number of basis functions
    
    numBasis = dk.get_numBasis(H,verbose=True) # Calculate the number of basis functions
            
    print(sum(numBasis))
        
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
    f.write(f"{len(knownPoints)} known points = {len(knownPoints)/len(stackedMap) * 100 } % of the Map, len(unkwonPoints) unknown points = {len(unknownPoints)/len(stackedMap) *100} % of the Map \n")
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
    
        
   
    BaseModel = bm.build_model(units, verbose=verbose)
    BaseModel, trainedModelPathBase = bm.train(knownPoints[['x', 'y']], knownPoints[['z']], unknownPoints[['x','y']], unknownPoints['z'],BaseModel, epochs, scenario,save_hist=save_hist, verbose=verbose)

     
    
    DeepKrigingModel = dk.build_model(x_trainDK.shape[1], verbose=verbose)
    
    DeepKrigingModel, trainedModelPathDK = dk.train_model(DeepKrigingModel, x_trainDK, y_trainDK, x_valDK, y_valDK, scenario, epochs, save_hist=save_hist, verbose=verbose)
    
    
    
    #### Generate Results

    resultsLinearInterpolation = np.zeros(len(stackedMap))
    resultsKriging = np.zeros(len(stackedMap))
    resultsBaseModel = np.zeros(len(stackedMap))
    resultsDeepKriging = np.zeros(len(stackedMap))
    
    
    for i in range(100,stackedMapNormalizedForDK.shape[0],100):
        
        testSlice = stackedMap[['x', 'y', 'z']].iloc[i-100:i-1] # stackedMapNormalizedForDK x and y coordinates are both 0-1 min max normalized
        testSliceDk = stackedMapNormalizedForDK[['x', 'y']].iloc[i-100:i-1]
        
        testSliceDk = dk.wendlandkernel(testSliceDk, numBasis) # Apply Wendlandkernel to subsample of data ( as test slice)
        print(knownPoints.head())
        print(testSlice.head())
        linRes = ip.grid_interpolation(knownPoints, testSlice, method='linear', verbose=verbose)
        
        #resultsLinearInterpolation[i-100:i-1] = linRes
        resultsKriging[i-100:i-1] = ip.kriging_skg(knownPoints, testSlice, 10).z.to_numpy()
        resultsBaseModel[i-100:i-1] = bm.predict(trainedModelPathBase, testSlice[['x', 'y']]).z.to_numpy()
        resultsDeepKriging[i-100:i-1] = dk.predict(trainedModelPathDK, testSliceDk)[:,0]
        
        
    x_test = stackedMap[['x','y']]
    x_testDK = stackedMapNormalizedForDK[['x','y']]
    
    

    
    
    resultsLinearInterpolation = pd.DataFrame([resultsLinearInterpolation, stackedMap['x'].to_numpy(),stackedMap['y'].to_numpy()]).transpose()
    resultsLinearInterpolation.columns = ['z','x','y']
    resultsLinearInterpolation = resultsLinearInterpolation[['x','y','z']]
    resultsLinearInterpolation.index = stackedMap.index
    resultsLinearInterpolation.to_csv(f"LinearInterpolationResult_Random-{random}.csv")
    
    
    resultsKriging = pd.DataFrame([resultsKriging, stackedMap['x'].to_numpy(),stackedMap['y'].to_numpy()]).transpose()
    resultsKriging.columns = ['z','x','y'] # Fix column names
    
    resultsKriging  = resultsKriging[['x','y','z']]
    
    resultsKriging.index = stackedMap.index # Fix index
    resultsKriging.to_csv(f"KrigingResult_Random-{random}.csv")
    
             
    resultsDeepKriging= pd.DataFrame([resultsDeepKriging,stackedMap['x'].to_numpy(),stackedMap['y'].to_numpy()]).transpose() # create a DataFrame 
                        
    resultsDeepKriging.columns = ['z','x','y'] # Fix column names
    
    resultsDeepKriging  = resultsDeepKriging[['x','y','z']]
    
    resultsDeepKriging.index = stackedMap.index # Fix index
    resultsDeepKriging.to_csv(f"DeepKrigingResult_Random-{random}.csv")
    
    
    
    resultsBaseModel = pd.DataFrame([resultsBaseModel,stackedMap['x'].to_numpy(),stackedMap['y'].to_numpy()]).transpose() # create a DataFrame
    
    resultsBaseModel.columns = ['z','x','y'] # Fix column names
    resultsBaseModel = resultsBaseModel[['x','y','z']] # Fix column order
    resultsBaseModel.index = stackedMap.index # Fix index
    resultsBaseModel.to_csv(f"BaseModelResult_Random-{random}.csv")
    
        
        
        
    
        
    
    
    # Work with Results
    
    global data
    
    data = {'Linear Interpolation':resultsLinearInterpolation,'Kriging':resultsKriging,'Deep Kriging':resultsDeepKriging,'Base Model':resultsBaseModel}
    
    
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
        pu.generateHeatMaps({key:data[key]},stackedMap, knownPoints, unknownPoints, maxValue, minValue, 0,path +f'{key}.png')
        
        
    pu.generateHeatMaps(data,stackedMap, knownPoints,unknownPoints, maxValue, minValue, 1,path+'heatmaps.png')
    
    maxError = 0
    
    for key in list(relErrors.keys()):
        if relErrors[key]['z'].max() > maxError:
            maxError = relErrors[key]['z'].max()
            
    # find min value from dictionary of pandas DataFrames
    
    
    minError = np.Inf
    for key in list(data.keys()):
        if relErrors[key]['z'].min() < minError:
            minError = relErrors[key]['z'].min()
            
            
    pu.generateHeatMaps(relErrors,stackedMap, knownPoints,unknownPoints,maxError, minError, 0,path+'relErrors.png')
    

    
if __name__ == '__main__':
    
    
    interpolateWholeMap = 0
    random = 0
    length = None
    epochs = 2
    reduceResolution = 0
    verbose = 1
    save_hist = 0
    units = 1500 # Base Model units per Dense Layer
    
    for interpolateWholeMap in [1]:
        
        
                
        
        for samplingDistance_x in [10]:
                
                samplingDistance_y = samplingDistance_x #* 12
                
                for random in [0,1]:
                    if random:
                            
                        scenario = f'wholeMap_x-{samplingDistance_x}_y-{int(samplingDistance_y/12)}_RandomSampling'
                                    
                    else:
                                    
                        scenario = f'wholeMap_x-{samplingDistance_x}_y-{int(samplingDistance_y/12)}_UniformSampling'
                                    
                                    
                    if interpolateWholeMap:
                        scenario = scenario + '_InterpolateAllPoints_Back_Straight_SISO_Power'
                    
                    if length:
                        scenario = scenario + f'_length-{length}'
                
                
                    main()
                
                
  