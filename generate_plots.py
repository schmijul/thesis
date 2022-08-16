import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
import ploting_utils as pu
import datapreparation as dp


def load_basemodelprediction(dist,samplingorder='uniform'):
    """
    _summary_
    Args:
        dist ([int]): sampling distance in cm
        samplingorder ([str]): uniform or random
    Returns:
        prediction ([pd.DataFrame])
    """
    prediction = loadmat(f'basemodel_predictions/basemodel_dist_{dist}_{samplingorder}_predictions.mat')
    prediction = pd.DataFrame(prediction['prediction_data'])
    prediction.columns = ['x', 'y', 'z']
    return prediction

def load_dkmodel(dist,samplingorder='uniform'):
    """
    _summary_
    Args:
        dist ([int]): sampling distance in cm
        samplingorder ([str]): uniform or random
    Returns:
        prediction ([pd.DataFrame])
    """
    prediction = loadmat(f'dkmodel_predictions/dkmodel_dist_{dist}_{samplingorder}_predictions_highNeuron.mat')
    prediction = pd.DataFrame(prediction['predictions'])
    prediction.columns = ['x', 'y', 'z']
    return prediction

def load_krigingresults(dist,samplingorder='uniform'):
    """
    _summary_
    Args:
        dist ([int]): sampling distance in cm
        samplingorder ([str]): uniform or random
    Returns:
        krigingresults ([pd.DataFrame])
    """
    krigingresults = pd.read_csv(f'interpolationresults/results_kriging_dist-{dist}_{samplingorder}.csv').drop(['Unnamed: 0'], axis = 1)
    return krigingresults

def load_lininterpolationresults(dist,samplingorder='uniform'):
    """
    _summary_
    Args:
        dist ([int]): sampling distance in cm
        samplingorder ([str]): uniform or random
    Returns:
        lininterpolationresult ([pd.DataFrame])
    """

    lininterpolationresult = pd.read_csv(f'interpolationresults/results_linear_interploation_dist-{dist}_{samplingorder}.csv').drop(['Unnamed: 0'], axis = 1)
    return lininterpolationresult


def main():
    """
    Loop through all results, calc erros and generate heatmaps
    """

    # Load original map
    originalmap = dp.preparemap(pd.read_csv('RadioEnvMaps/Main_Straight_SISO_Power_Map.csv'))[1]
    knownpoints = pd.read_csv(f'dk_data/trainset/dist-{DIST}_{SAMPLINGORDER}.csv')[['x', 'y','z']]
    unknownpoints = pd.read_csv(f'dk_data/valset/dist-{DIST}_{SAMPLINGORDER}.csv')[['x', 'y','z']]

    results = {'lininterpolation': load_lininterpolationresults(DIST,samplingorder=SAMPLINGORDER),
               'kriging': load_krigingresults(DIST,samplingorder=SAMPLINGORDER),
               'dkmodel': load_dkmodel(DIST,samplingorder=SAMPLINGORDER)}

    path = f'results/plots/interpolate_wholemap/main_straigh_siso/dist-{DIST}_samplingorder-{SAMPLINGORDER}'
    print(DIST)
    if not os.path.exists(path):
        os.makedirs(path)


    # Plotting

    for key in list(results.keys()):
        pu.generateheatmaps({key:results[key]},
                            originalmap,
                            knownpoints,
                            originalmap,
                            originalmap['z'].max(),
                            originalmap['z'].min(),
                            0,
                            path +f'/{key}.png')

    pu.generateheatmaps(results,
                        originalmap,
                        knownpoints,
                        originalmap,
                        originalmap['z'].max(),
                        originalmap['z'].min(),
                        1,
                        path+'/heatmaps.png')


if __name__ == '__main__':

    for DIST in [ 12, 16]:
        for SAMPLINGORDER in ['random']:
             main()