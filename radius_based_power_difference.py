
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import datapreparation as dp
import radiomap_construction as rmc





def radius_dependent_variance(map, radius, random_points_to_check):
    
    ys_cm = np.arange(map.shape[0])
    xs_cm = np.arange(map.shape[1])

    Xs_cm, Ys_cm = np.meshgrid(xs_cm, ys_cm)

    Xs_flatt = np.reshape(Xs_cm, (1, Xs_cm.shape[0] * Xs_cm.shape[1]))
    Ys_flatt = np.reshape(Ys_cm, (1, Ys_cm.shape[0] * Ys_cm.shape[1]))

    coors = np.concatenate((Xs_flatt, Ys_flatt)).transpose()
    data = np.empty((coors.shape[0], 3))
    data[:, 0:2] = coors
    data[:, 2] = np.nan

    counter = 0
    for row_index in range(map.shape[0]):

        for col_index in range(map.shape[1]):

            data[counter, 2] = map[row_index, col_index]
            counter += 1

    # dummy preparation
    power_diffferences_collection = np.empty((int(random_points_to_check * 100), ))
    power_diffferences_collection[:] = np.nan

    random_indices = np.random.permutation(np.arange(0, data.shape[0]))[0:random_points_to_check]

    # go through each data point and look for instances in list that have a certain distance
    current_sort_in_counter = 0
    counter = 0
    for data_sample in random_indices:

        counter += 1
        if np.mod(counter, 1000) == 0:
            print(counter)

        current_power = data[data_sample, 2]
        relevant_samples = data[np.array(np.linalg.norm(data[:, 0:2] - data[data_sample, 0:2], 2, 1) > (radius - 0.5)) &
                                np.array(np.linalg.norm(data[:, 0:2] - data[data_sample, 0:2], 2, 1) < (radius + 0.5)),
                                2]

        power_differences = relevant_samples - current_power
        power_diffferences_collection[current_sort_in_counter:current_sort_in_counter+power_differences.size] = \
            power_differences
        current_sort_in_counter = current_sort_in_counter + power_differences.size


    return power_diffferences_collection[~np.isnan(power_diffferences_collection)]





def get_pdc_per_slice(x_min, x_max, y_min, slicesize,  radius, random_points_to_check,dcor):
    pdc_syn = []
    pdc_ref = []
    y_max = 0
    while y_max < 2020:
        y_max = y_min + slicesize
        print(y_min, y_max)
        print
        map = rmc.generate_map(x_min, x_max, y_min, y_max, dcor=dcor)
        reference_map = dp.preparemap(pd.read_csv('RadioEnvMaps/Main_Straight_SISO_Power_Map.csv').iloc[y_min:y_max])[1]
        pdc_syn.append(radius_dependent_variance(map.to_numpy(), radius, random_points_to_check))
        pdc_ref.append(radius_dependent_variance(reference_map.to_numpy(), radius, random_points_to_check))
        y_min = y_max +1
    return pdc_syn, pdc_ref

def main(dcor):
    X_MIN = 0
    X_MAX = 91
    Y_MIN = 0
    pdc_syn_dcor, pdc_ref_dcor = get_pdc_per_slice(X_MIN, X_MAX, Y_MIN, 400,10, 400, dcor)

    pdc_syn = np.concatenate(pdc_syn_dcor)
    np.save(f'pdc_syn_{dcor}.npy', np.array(pdc_syn))
    pdc_ref = np.concatenate( pdc_ref_dcor)
    np.save('pdc_ref.npy', np.array(pdc_ref))


if __name__ == "__main__":
    





    pool = mp.Pool()
    
 

    pool.map(main,[dcor/100 for dcor in range(1,50,10)])