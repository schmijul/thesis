import os
import pandas as pd
import datapreparation as dp
import deepkriging as dk
import basemodel as bm
import tensorflow as tf
import numpy as np
import interpolation_utils as iu
def preparemap():

    """
    bring map into workable format
    """

    global MAP, STACKEDMAP, KNOWNPOINTS, UNKNOWNPOINTS


    wholemap = pd.read_csv('/home/schmijul/source/repos/thesis/RadioEnvMaps/Main_Straight_SISO_Power_Map.csv').iloc[startpoint:endpoint]

    MAP, STACKEDMAP = dp.preparemap(wholemap, length=LENGTH)



    KNOWNPOINTS, UNKNOWNPOINTS = dp.resample(STACKEDMAP.copy(), samplingdistance, samplingdistance)


    if random:

        KNOWNPOINTS = dp.randomsampling(STACKEDMAP.copy(), len(KNOWNPOINTS), samplingdistance)



def normalizedata():

    """
    0-1 data normalization
    """
    # minmax normalize the data
    global  STACKEDMAP_NORMALIZED, KNOWNPOINTS_NORMALIZED, UNKNOWNPOINTS_NORMALIZED, MAXVALS, MINVALS
    STACKEDMAP_NORMALIZED, KNOWNPOINTS_NORMALIZED, UNKNOWNPOINTS_NORMALIZED, MAXVALS, MINVALS = dp.normalize_data(STACKEDMAP.copy(),
                                                                                                                  KNOWNPOINTS.copy(),
                                                                                                                  UNKNOWNPOINTS.copy())


def get_wendlandparams():

    """
    get parameter for wendland kernel
    """

    global NUMBASIS




    #numelements = dp.calc_h_for_num_basis(len(STACKEDMAP)) # Calculate the number of basis functions

    NUMBASIS = dp.get_numbasis(7) # Calculate the number of basis functions



def main():

    """
    main
    """

    preparemap()

    normalizedata()

    get_wendlandparams()

    x_train = KNOWNPOINTS_NORMALIZED[['x', 'y']]
    y_train = KNOWNPOINTS['z']
    


    x_val = UNKNOWNPOINTS_NORMALIZED[['x', 'y']]
    y_val = UNKNOWNPOINTS['z']

    basemodel = bm.build_model(2000)


    deepkrigingmodel = dk.build_model(sum(NUMBASIS))
    
    
    dk.train_model(deepkrigingmodel,
                   dp.wendlandkernel(x_train, NUMBASIS),
                   y_train,
                   dp.wendlandkernel(x_val, NUMBASIS),
                   y_val,
                   scenario,
                   EPOCHS,
                   batch_size=100,
                   save_hist=True,
                   verbose=1)



    bm.train(x_train,
             y_train,
             x_val,
             y_val,
             basemodel,
             EPOCHS,
             scenario,
             save_hist=True,
             verbose=VERBOSE,
             batch_size=100)


    
    # Predictions
    
    # Load best dk model:
    
    dk_model = tf.keras.models.load_model(f'trainedModels/deepkriging/{scenario}/best_model.h5')
    print('a')
    dk_prediction = dk_model.predict(dp.wendlandkernel(STACKEDMAP_NORMALIZED,NUMBASIS))[:,0]
    print('b')
    
    bm_model = tf.keras.models.load_model(f'trainedModels/baseModel/{scenario}/best_model.h5')

    bm_prediction = bm_model.predict(STACKEDMAP_NORMALIZED[['x', 'y']])[:,0]

    # Save predictions:
    
    # create cenario directory
    
    if not os.path.exists(f'slicedPredictions/{scenario}'):
        os.makedirs(f'slicedPredictions/{scenario}')

    np.save(f'slicedPredictions/{scenario}/dk_prediction.npy', dk_prediction)

    np.save(f'slicedPredictions/{scenario}/bm_prediction.npy', bm_prediction)
    
    
    # Linear interpolation:
    
    results_linear_interploation = iu.gridinterpolation(KNOWNPOINTS,
                                                        STACKEDMAP,
                                                        method='linear',
                                                        verbose=0)
    
    np.save(f'slicedPredictions/{scenario}/linear_interpolation.npy', results_linear_interploation)
    
    # Kriging

    results_kriging = iu.kriging(KNOWNPOINTS, STACKEDMAP)
     
    np.save(f'slicedPredictions/{scenario}/kriging.npy', results_kriging)


if __name__ == '__main__':

    # def globals at module level

    MAP = None
    STACKEDMAP = None
    KNOWNPOINTS = None
    UNKNOWNPOINTS = None
    STACKEDMAP_NORMALIZED = None
    KNOWNPOINTS_NORMALIZED = None
    UNKNOWNPOINTS_NORMALIZED = None
    MAXVALS = None
    MINVALS = None
    NUMBASIS = None 

    EPOCHS = 2
    LENGTH = None
    VERBOSE = 0
    for i in [1,3,5,7,9]:
        startpoint = i*92
        endpoint = (i+1)*92
        
        print(f"startpoint = {startpoint}")
        print(f"endpoint = {endpoint}")
        for random in [1,0]:
            for samplingdistance in range(8,4,-4):

                print(f"random: {random}, samplingdistance: {samplingdistance}")
                scenario = f'slice_from-{startpoint}_to-{endpoint}-{samplingdistance}-random-{random}_notnormalized'
                main()
