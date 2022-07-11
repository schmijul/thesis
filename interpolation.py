import pandas as pd
import interpolation_utils as iu
import datapreparation as dp

def preparemap():

    """
    bring map into workable format
    """

    global MAP, STACKEDMAP, KNOWNPOINTS, UNKNOWNPOINTS

    wholemap = pd.read_csv('/home/schmijul/source/repos/thesis/RadioEnvMaps/Main_Straight_SISO_Power_Map.csv')

    MAP, STACKEDMAP = dp.preparemap(wholemap, length=LENGTH)



    KNOWNPOINTS, UNKNOWNPOINTS = dp.resample(STACKEDMAP.copy(), samplingdistance, samplingdistance)


    if RANDOM:

        KNOWNPOINTS = dp.randomsampling(STACKEDMAP.copy(), len(KNOWNPOINTS))



def main():
    preparemap()
    results_linear_interploation = iu.gridinterpolation(KNOWNPOINTS,
                                                            UNKNOWNPOINTS,
                                                            method='linear',
                                                            verbose=True)

    results_linear_interploation.to_csv(f'interpolationresults/results_linear_interploation_dist-{samplingdistance}_uniform.csv')

    results_kriging = iu.kriging(KNOWNPOINTS, UNKNOWNPOINTS)
    results_kriging.to_csv(f'interpolationresults/results_kriging_dist-{samplingdistance}_random.csv')

if __name__ == '__main__':

    LENGTH = None
    RANDOM = 1
    for samplingdistance in range(28,0,-4):
        main()
