
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf

def create_callback(trainedModelPath,verbose=False):
    
        """
        quick function to create callbacks and or overwrite existing callbacks
        
        """
        
        return  [
            tf.keras.callbacks.ModelCheckpoint(
                trainedModelPath+"/best_model.h5", save_best_only=True, monitor="val_loss", verbose=verbose
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            )
            
        ]

def deep_kriging(map, known_points, unknown_points):
    print(f'len training-data = {len(known_points)}')
    print(f'len validation-data = {len(unknown_points)}')
    print(f'N = {len(map)}')

    x = map.x
    y = map.y
    #x = map['x'].unique()#/known_points.x.max()
    #y = map['y'].unique()#/known_points.y.max()

    x = (x-min(x))/(max(x)-min(x))
    y = (y-min(y))/(max(y)-min(y))
    N = x.shape[0]
    num_basis = [10**2,19**2,37**2,73**2, 145**2]

    knots_1dx = [np.linspace(0,1,int(np.sqrt(i))) for i in num_basis]
    knots_1dy = [np.linspace(0,1,int(np.sqrt(i))) for i in num_basis]
    ##Wendland kernel
    basis_size = 0
    phi = np.zeros((N, sum(num_basis)))
    for res in range(len(num_basis)):
        theta = 1/np.sqrt(num_basis[res])*2.5
        knots_x, knots_y = np.meshgrid(knots_1dx[res],knots_1dy[res])
        knots = np.column_stack((knots_x.flatten(),knots_y.flatten()))
        for i in range(num_basis[res]):
            d = np.linalg.norm(np.vstack((y,x)).T-knots[i,:],axis=1)/theta
            for j in range(len(d)):
                if d[j] >= 0 and d[j] <= 1:
                    phi[j,i + basis_size] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
                else:
                    phi[j,i + basis_size] = 0
        basis_size = basis_size + num_basis[res]

    phi = pd.DataFrame(phi)
    phi.index=phi.index+map.index.min()
    ##### idx for train/val split


    print(f'max phi index : {phi.index.max()}')
    print(f'max known points index : {known_points.index.max()}')
    print(f'max unknown points index : {unknown_points.index.max()}')
    train_idx = known_points.loc[known_points.index < phi.index.max()].index
    val_idx = unknown_points.iloc[unknown_points.index < phi.index.max()].index
    #train_idx=train_idx[0]
    #val_idx = val_idx[0]
    x_train = phi.loc[train_idx].to_numpy()
    y_train = known_points.z.loc[train_idx].to_numpy()
    x_val = phi.loc[val_idx].to_numpy()
    y_val = unknown_points.z.loc[val_idx].to_numpy()

    ##normalization
    maxval = np.max([np.max(y_train),np.max(y_val)])

    y_train = y_train / maxval
    y_val = y_val / maxval
    

    NB_START_EPOCHS = 1700
    BATCH_SIZE = 100


    # DeepKriging model for continuous data
    p = phi.shape[1]
    model = Sequential()
    model.add(Dense(100, input_dim = p,  kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))

    trainedModelPath = f'trainedModels/NonRandomResapling/deepkriging/'#samplerate_x{sampling_distance_x}_y{sampling_distance_y}_length{length}/'

    if not os.path.exists(trainedModelPath):
                                os.makedirs(trainedModelPath)
                    
        

    model.compile(optimizer='adam'
                        , loss='mse'
                        , metrics=['mse','mae'])
        
    history = model.fit(x_train
                        , y_train
                        , epochs=NB_START_EPOCHS
                        , batch_size=BATCH_SIZE
                        , validation_data=(x_val, y_val)
                        , callbacks = create_callback(trainedModelPath)
                        , verbose=0)
        
    model = tf.keras.models.load_model(trainedModelPath+'/best_model.h5')


    prediction = model.predict(x_val) * maxval

    prediction = pd.DataFrame(prediction)

    prediction.index = val_idx

    pred_stacked = pd.merge(unknown_points[['x','y']], prediction,how='inner', left_index=True, right_index=True)
    pred_stacked.columns = ['x','y','z']

    return pred_stacked.pivot_table(index='y', columns='x', values='z'),pred_stacked