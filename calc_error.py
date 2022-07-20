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
    prediction = loadmat(f'dkmodel_predictions/predictions/dkmodel_dist_{dist}_{samplingorder}_predictions.mat')
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
    knownpoints = pd.read_csv(f'dk_data/dist-{DIST}_{SAMPLINGORDER}/trainset.csv')[['x', 'y','z']]
    unknownpoints = pd.read_csv(f'dk_data/dist-{DIST}_{SAMPLINGORDER}/valset.csv')[['x', 'y','z']]

    results = {'lininterpolation': load_lininterpolationresults(DIST,samplingorder=SAMPLINGORDER),
               'kriging': load_krigingresults(DIST,samplingorder=SAMPLINGORDER),
               'basemodel': load_basemodelprediction(DIST,samplingorder=SAMPLINGORDER),
               'dkmodel': load_dkmodel(DIST,samplingorder=SAMPLINGORDER)}

    
    # Reminmax of machine learning predictions

    for col in results['basemodel'].columns:
        
        results['basemodel'][col] = dp.reminmax(results['basemodel'][col],
                                             originalmap[col].max(),
                                             originalmap[col].min())
        results['dkmodel'][col] = dp.reminmax(results['dkmodel'][col],
                                           originalmap[col].max(),
                                           originalmap[col].min())
        knownpoints[col] = dp.reminmax(knownpoints[col],
                                       originalmap[col].max(),
                                       originalmap[col].min())
        unknownpoints[col] = dp.reminmax(unknownpoints[col],
                                         originalmap[col].max(),
                                         originalmap[col].min())
        
    # Plotting
    maes={}
    mses={}
    for prediction in list(results.keys()):
        
        prediction = results[prediction]['z']
        truth = originalmap['z']
        
        mae = np.mean(np.abs(prediction - truth))
        mse = np.mean(np.square(prediction - truth))
        
        maes[prediction] = [mae]
        mses[prediction] = [mse]
    pd.DataFrame(maes).to_csv(f'maes_dist-{DIST}_{SAMPLINGORDER}.csv')
    pd.DataFrame(mses).to_csv(f'mses_dist-{DIST}_{SAMPLINGORDER}.csv')
    
    
if __name__ == '__main__':
    
    for DIST in [16]:
        for SAMPLINGORDER in ['random']:
             main()