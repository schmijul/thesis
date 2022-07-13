import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
import ploting_utils as pu
import datapreparation as dp
from deepkriging import reminmax


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
    prediction = loadmat(f'dkmodel_predictions/basemodel_dist_{dist}_{samplingorder}_predictions.mat')
    prediction = pd.DataFrame(prediction['prediction_data'])
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
    krigingresults = pd.read_csv(f'interpolationsresults/results_kriging_dist-{dist}_{samplingorder}.csv')
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

    lininterpolationresult = pd.read_csv(f'interpolationsresults/results_linear_interploation_dist-{dist}_{samplingorder}.csv')
    return lininterpolationresult


def main():
    """
    Loop through all results, calc erros and generate heatmaps
    """

    # Load original map
    originalmap = dp.preparemap(pd.read_csv('RadioEnvMaps/Main_Straight_SISO_Power_Map.csv'))[1]
    knownpoints, unknownpoints = dp.resample(originalmap.copy(), DIST, DIST)


    results = {'lininterpolation': load_lininterpolationresults(DIST,samplingorder=SAMPLINGORDER),
               'kriging': load_krigingresults(DIST,samplingorder=SAMPLINGORDER),
               'basemodel': load_basemodelprediction(DIST,samplingorder=SAMPLINGORDER),
               'dkmodel': load_dkmodel(DIST,samplingorder=SAMPLINGORDER)}

    path = f'results/plots/interpolate_wholemap/main_straigh_siso/dist-{DIST}_samplingorder-{SAMPLINGORDER}'

    if not os.path.exists(path):
        os.makedirs(path)

    # Reminmax of machine learning predictions

    for col in results['basemodel'].columns:
        results['basemodel'][col] = reminmax(results['basemodel'],
                                             originalmap[col].max(),
                                             originalmap[col].min())
        results['dkmodel'][col] = reminmax(results['dkmodel'][col],
                                           originalmap[col].max(),
                                           originalmap[col].min())
    # Plotting

    for key in list(results.keys()):
        pu.generateHeatMaps({key:results[key]},
                            results,
                            knownpoints,
                            unknownpoints,
                            originalmap[col].max(),
                            originalmap[col].min(),
                            0,
                            path +f'{key}.png')

    pu.generateHeatMaps(results,
                        originalmap,
                        knownpoints,
                        unknownpoints,
                        originalmap[col].max(),
                        originalmap[col].min(),
                        1,
                        path+'heatmaps.png')

    # Error calculation
    
if __name__ == '__main__':
    
    for DIST in [4, 8, 12, 16]:
        for SAMPLINGORDER in ['uniform', 'random']:
             main()