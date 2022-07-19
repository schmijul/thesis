import pandas as pd
import interpolation_utils as iu
import datapreparation as dp

def preparemap():

    """
    bring map into workable format
    """

    global MAP, STACKEDMAP, KNOWNPOINTS, UNKNOWNPOINTS

    wholemap = pd.read_csv('RadioEnvMaps/Main_Straight_SISO_Power_Map.csv')

    MAP, STACKEDMAP = dp.preparemap(wholemap, length=LENGTH)

    print(f' len stackedmap : {len(STACKEDMAP)}')

    KNOWNPOINTS, UNKNOWNPOINTS = dp.resample(STACKEDMAP.copy(), samplingdistance, samplingdistance)


    if RANDOM:

        KNOWNPOINTS = dp.randomsampling(STACKEDMAP.copy(), len(KNOWNPOINTS))



def main():
    preparemap()
    results_linear_interploation = iu.gridinterpolation(KNOWNPOINTS,
                                                        STACKEDMAP,
                                                        method='linear',
                                                        verbose=0)
    print(len(results_linear_interploation))

    results_linear_interploation.to_csv(f'interpolationresults/results_linear_interploation_dist-{samplingdistance}_random.csv')

    results_kriging = iu.kriging(KNOWNPOINTS, STACKEDMAP)
    results_kriging.to_csv(f'interpolationresults/results_kriging_dist-{samplingdistance}_random.csv')

if __name__ == '__main__':

    LENGTH = None
    RANDOM = 0
    for samplingdistance in range(28,0,-4):
        main()
