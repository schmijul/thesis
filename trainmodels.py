import pandas as pd
import datapreparation as dp
import deepkriging as dk
import basemodel as bm

def preparemap():

    """
    bring map into workable format
    """

    global MAP, STACKEDMAP, KNOWNPOINTS, UNKNOWNPOINTS


    wholemap = pd.read_csv('/home/schmijul/source/repos/thesis/RadioEnvMaps/Main_Straight_SISO_Power_Map.csv')

    MAP, STACKEDMAP = dp.preparemap(wholemap, length=LENGTH)



    KNOWNPOINTS, UNKNOWNPOINTS = dp.resample(STACKEDMAP.copy(), samplingdistance, samplingdistance)


    if random:

        KNOWNPOINTS = dp.randomsampling(STACKEDMAP.copy(), len(KNOWNPOINTS))



def normalizedata():

    """
    0-1 data normalization
    """
    # minmax normalize the data
    global  STACKEDMAP_NORMALIZED, KNOWNPOINTS_NORMALIZED, UNKNOWNPOINTS_NORMALIZED, MAXVALS, MINVALS
    STACKEDMAP_NORMALIZED, KNOWNPOINTS_NORMALIZED, UNKNOWNPOINTS_NORMALIZED, MAXVALS, MINVALS = dk.normalize_data(STACKEDMAP.copy(),
                                                                                                                  KNOWNPOINTS.copy(),
                                                                                                                  UNKNOWNPOINTS.copy())


def get_wendlandparams():

    """
    get parameter for wendland kernel
    """

    global NUMBASIS




    numelements = dk.calc_h_for_num_basis(len(STACKEDMAP)) # Calculate the number of basis functions

    NUMBASIS = dk.get_numbasis(numelements) # Calculate the number of basis functions


def main():

    """
    main
    """

    preparemap()

    normalizedata()

    get_wendlandparams()

    x_train = KNOWNPOINTS_NORMALIZED[['x', 'y']]
    y_train = KNOWNPOINTS_NORMALIZED['z']


    x_val = UNKNOWNPOINTS_NORMALIZED[['x', 'y']]
    y_val = UNKNOWNPOINTS_NORMALIZED['z']

    basemodel = bm.build_model(2000)
    print('start training')
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

    print('finished training basemodel')
    deepkrigingmodel = dk.build_model(sum(NUMBASIS))

    dk.train_model(deepkrigingmodel,
                   dk.wendlandkernel(x_train, NUMBASIS),
                   y_train,
                   dk.wendlandkernel(x_val, NUMBASIS),
                   y_val,
                   scenario,EPOCHS,
                   batch_size=100,
                   save_hist=False,
                   verbose=VERBOSE)

    print('finished training deepkrigingmodel')

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

    EPOCHS = 2000
    LENGTH = None
    VERBOSE = 0

    for random in [1]:
        for samplingdistance in range(28,0,-4):
            print(f"random: {random}, samplingdistance: {samplingdistance}")
            scenario = f'Main_Straight_SISO_Power_Map_samplingDistance-{samplingdistance}-random-{random}'
            main()