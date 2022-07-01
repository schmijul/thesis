import pandas as  pd

import numpy as np

import data_preparation as dp



wholeMap = pd.read_csv('RadioEnvMaps/Back_Straight_SISO_Power_Map.csv')
#wholeMap = wholeMap.iloc[int(len(wholeMap)/2):]
#wholeMap = wholeMap.iloc[::12,:]

Map, StackedMap = dp.prepare_map(wholeMap.copy())
print(StackedMap.shape)
#wholeMap = wholeMap.iloc[:int(len(wholeMap)/2)]
        

import deepkriging as dk
 

H = dk.calc_H_for_num_basis(len(StackedMap))
numBasis = dk.findWorkingNumBasis(len(StackedMap), H)

phi = dk.wendlandkernel(StackedMap, numBasis)

np.save('RadioEnvMaps/Back_Straight_SISO_Power_Map_Wendland.npy', phi.to_numpy())

