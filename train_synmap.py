import os
import tensorflow as tf
import numpy as np
import pandas as pd
import datapreparation as dp
import deepkriging as dk
import basemodel as bm
import interpolation_utils as iu

def preparemap():
    
    """
    bring map into workable format
    """

    global  STACKEDMAP, KNOWNPOINTS, UNKNOWNPOINTS


    STACKEDMAP = pd.read_csv(f'RadioEnvMaps/syndata/virtualmainmap_{y_min}_to_{y_max}.csv')[['x','y','z']]

    STACKEDMAP.x = STACKEDMAP.x * 100
    STACKEDMAP.y = STACKEDMAP.y * 100


    KNOWNPOINTS, UNKNOWNPOINTS = dp.resample(STACKEDMAP.copy(), samplingdistance, samplingdistance)
    print(f"{len(KNOWNPOINTS)} known points")

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
    print("map prepared")

    normalizedata()

    get_wendlandparams()

    x_train = KNOWNPOINTS_NORMALIZED[['x', 'y']]
    y_train = KNOWNPOINTS['z']
    


    x_val = UNKNOWNPOINTS_NORMALIZED[['x', 'y']]
    y_val = UNKNOWNPOINTS['z']

    basemodel = bm.build_model(2500)


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

    dk_prediction = dk_model.predict(dp.wendlandkernel(STACKEDMAP_NORMALIZED,NUMBASIS))[:,0]

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
        STACKEDMAP = None
        KNOWNPOINTS = None
        UNKNOWNPOINTS = None
        STACKEDMAP_NORMALIZED = None
        KNOWNPOINTS_NORMALIZED = None
        UNKNOWNPOINTS_NORMALIZED = None
        NUMBASIS = None
        EPOCHS = 2000
        VERBOSE = 0
        y_min = 0
        y_max = 0
        while y_max < 2020:
            y_max = y_min +400
            print(y_max)
            print(f"startpoint = {y_min}")
            print(f"endpoint = {y_max}")
            for random in [1,0]:
                for samplingdistance in[8]:

                    print(f"random: {random}, samplingdistance: {samplingdistance}")
                    scenario = f'Synthetic_slice_from-{y_min}_to-{y_max}-{samplingdistance}-random-{random}_notnormalized'
                    main()
            y_min = y_max+1
