#!/usr/bin/env python
# coding: utf-8

# In[1]:
import seaborn as sns
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error 
debug = True
import os
import pandas as pd
import tensorflow as tf
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.gaussian_process import GaussianProcess
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from data_preparation import *
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6)
import pylab 
import numpy as np
import pandas as pd
import skgstat as skg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from skgstat.models import spherical
from scipy.linalg import solve
from sklearn.metrics import mean_squared_error as mse

def rel_error(pred, true):
    return np.abs(pred - true) / np.abs(true)

def create_callback(trainedModelPath):

    return  [
        keras.callbacks.ModelCheckpoint(
            trainedModelPath+"/best_model.h5", save_best_only=True, monitor="val_loss", verbose=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        )
        
    ]

mses ={'lin':[], 'krig':[], 'nn':[]}

for sampling_distance_y in [12,26,50,76,100]:
    for sampling_distance_x in [4,8,10]:

        length = int(sampling_distance_y/12 * 1e3)


        whole_map = pd.read_csv('WholeMap_Rounds_40_to_17.csv')
        length = int(length)
        whole_map = whole_map.astype(float).set_index('Unnamed: 0')
        whole_map = whole_map.reset_index().drop(columns=['Unnamed: 0'])
        whole_map.columns = [x for x in range(len(whole_map.columns))]

        whole_map = whole_map.iloc[:length]

        whole_map_stacked = whole_map.stack().reset_index()
        whole_map_stacked.columns = ['y', 'x', 'z']
        whole_map_stacked = whole_map_stacked[['x', 'y', 'z']]
        data = whole_map_stacked[whole_map_stacked['y' ]%(sampling_distance_y/2) == 0]

        # resampling on y axis
        data_i = data[data['y'] % int(sampling_distance_y) == 0]
        data_o = data[data['y'] % sampling_distance_y != 0]


        # resampling on x axis
        data_i = data_i[data_i['x'] % int(sampling_distance_x) == 0]

        data_o = data_o[data_o['x'] % int(sampling_distance_x/2)  == 0 ]
        data_o =data_o[ data_o['x'] % int(sampling_distance_x) != 0]
        if data_o['x'].max() > data_i['x'].max():
            data_o = data_o[data_o['x'] != data_o['x'].max()]
      
        x_train = data_i[['x','y']]
        y_train = data_i[['z']]
        x_val = data_o[['x','y']]
        y_val = data_o[['z']]
        trainedModelPath = f'trainedModels/samplerate_x{sampling_distance_x}_y{sampling_distance_y}/'
                # DeepKriging model for continuous data with 4 hidden layers
        if not os.path.exists(trainedModelPath):
                            os.makedirs(trainedModelPath)
                
        model_base = Sequential()
        model_base.add(Dense(100, input_dim=x_train.shape[1],  kernel_initializer='he_uniform', activation='relu'))
        model_base.add(Dropout(rate=0.5))
        model_base.add(BatchNormalization())
        model_base.add(Dense(100, activation='relu'))
        model_base.add(Dropout(rate=0.5))
        model_base.add(Dense(100, activation='relu'))
                #model_base.add(Dropout(rate=0.5))
        model_base.add(BatchNormalization())
        model_base.add(Dense(1, activation='linear'))
        opt = keras.optimizers.Adam(learning_rate=0.0001)
        model_base.compile(loss='mse', optimizer=opt, metrics=['mse','accuracy'])

        callbacks = create_callback(trainedModelPath)
        x_train = x_train.to_numpy()
        x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
        y_train = y_train.to_numpy()
        y_train = y_train.reshape(y_train.shape[0],1)
        x_val= x_val.to_numpy()
        x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
        y_val= y_val.to_numpy()
        y_val = y_val.reshape(y_val.shape[0],1)

        history = model_base.fit(x_train
                                , y_train
                                , epochs=500
                                , validation_data = (x_val, y_val)
                                , callbacks=callbacks
                                )
        pd.DataFrame(history.history).to_csv(f'{trainedModelPath}/history_x{sampling_distance_x}_y{sampling_distance_y}.csv')
        model = tf.keras.models.load_model(trainedModelPath+"best_model.h5")
        prediction = model.predict(x_val)
                #mse_nn = np.mean((prediction- y_val.to_numpy())**2)

             #mses_nn.append( mse_nn)


                #np.save('mses_nn_r4.npy', np.array(mses_nn))
        np.save(f'predictions/NN_x{sampling_distance_x}_y{sampling_distance_y}.npy', prediction)

        pred_stacked = pd.DataFrame([prediction[:,0],data_o['x'].to_numpy(),data_o['y'].to_numpy()]).transpose()
        pred_stacked.columns = ['z','x','y']

        pred_stacked = pred_stacked[['x','y','z']]

        
        
        x = data_o['x']
        y = data_o['y']         
        xx, yy = np.mgrid[x.min():x.max()+1:sampling_distance_x, y.min():y.max()+1:sampling_distance_y]
        lin_interpol = griddata(data_i[['x', 'y']], # Points we know
                        data_i['z'], # Values we know
                        (xx.T, yy.T), # Points to interpolate
                        method='linear')
        lin_interpol= pd.DataFrame(lin_interpol)
        lin_interpol.to_csv(f'predictions/lin_interpol_x{sampling_distance_x}_y{sampling_distance_y}.csv')
        
        ground_truth = data_o.pivot_table(index='y', columns='x', values='z')

        V = skg.Variogram(data_i[['x','y']].values,data_i.z.values.flatten(), maxlag='median', n_lags=15, normalize=False, verbose=True)
        
        ok = skg.OrdinaryKriging(V, min_points=5, max_points=80, mode='exact')  
        
        field = ok.transform(x,y)
        kriging_stacked = pd.DataFrame([field,data_o['x'].to_numpy(),data_o['y'].to_numpy()]).transpose()
        kriging_stacked.columns = ['z','x','y']
        kriging_stacked  = kriging_stacked[['x','y','z']]

        np.save(f'krigingfield_x{sampling_distance_x}sampling_distance_y{sampling_distance_y}.npy',field)
        minval = np.min([np.min(kriging_stacked['z'].to_numpy()),np.min(ground_truth.to_numpy()),np.min(np.nan_to_num(lin_interpol)),np.min(pred_stacked['z'].to_numpy())])
        maxval = np.max([np.max(kriging_stacked['z'].to_numpy()),np.max(ground_truth.to_numpy()),np.max(np.nan_to_num(lin_interpol)),np.max(pred_stacked['z'].to_numpy())]) 

        lin_interpol.columns = ground_truth.columns.values
        lin_interpol.index = ground_truth.index.values

        lin = lin_interpol.stack().reset_index()
        lin.columns = ['y', 'x', 'z']
        lin = lin[['x', 'y', 'z']]
        print('c')
        lin.index = data_o.index
        print('c')
        fig, axes = plt.subplots(1, 4, figsize=(20, 8))
        sns.heatmap(ax=axes[0],data=ground_truth,vmin=minval-10, vmax=maxval)
        axes[0].title.set_text('ground truth')
        sns.heatmap(ax=axes[1],data=pd.DataFrame(lin_interpol),vmin=minval-10, vmax=maxval)
        axes[1].title.set_text('lin interpol')
        sns.heatmap(ax=axes[2],data=kriging_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval-10, vmax=maxval)
        axes[2].title.set_text('kriging')
        #sns.heatmap(ax=axes[1],data=kriging_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=maxval)
        sns.heatmap(ax=axes[3],data=pred_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval-10, vmax=0).set(title='Title of Plot')
        axes[3].title.set_text('NN')
        plt.savefig(f'plots/a_hm_x{sampling_distance_x}_y{sampling_distance_y}.png')
        plt.close()
        
        kriging_stacked.index = data_o.index
        #print('a')
        pred_stacked.index = data_o.index
       # print('a')
        wms = whole_map_stacked.copy()
        wms['z'] = 0
       # print('a')
        wms.loc[kriging_stacked.index] = np.nan

        kriging_whole_map = wms.combine_first(kriging_stacked)
        print('wms')
        pred_whole_map  = wms.combine_first(pred_stacked)
        
        lin_interpol_whole_map  = wms.combine_first(lin)
        gt = wms.combine_first(data_o)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 8))
        sns.heatmap(ax=axes[0],data=gt.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=maxval)
        axes[0].title.set_text('ground truth')
        sns.heatmap(ax=axes[1],data=lin_interpol_whole_map.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=maxval)
        axes[1].title.set_text('lin interpol')
        sns.heatmap(ax=axes[2],data=kriging_whole_map.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=maxval)
        axes[2].title.set_text('kriging')
        #sns.heatmap(ax=axes[1],data=kriging_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=maxval)
        sns.heatmap(ax=axes[3],data=pred_whole_map.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=0).set(title='Title of Plot')
        axes[3].title.set_text('NN')
        plt.savefig(f'plots/a_hm_full_x{sampling_distance_x}_y{sampling_distance_y}.png')
        
        

        mses['lin'].append(mse(lin['z'], data_o['z']))
        mses['krig'].append(mse(kriging_stacked['z'], data_o['z']))
        mses['nn'].append(mse(prediction, data_o['z']))


        interpol_error = rel_error(lin_interpol,ground_truth)
        krig_error = rel_error(kriging_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),ground_truth)
        nn_error = rel_error(pred_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),ground_truth)

            
        max_error = np.max([interpol_error.max(), krig_error.max(), nn_error.max()])
        min_error = np.min([interpol_error.min(), krig_error.min(), nn_error.min()])
        
            
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        sns.heatmap(ax=axes[0],data=interpol_error,vmin=min_error, vmax=max_error)
        axes[0].set_title='error lin interpol'
        sns.heatmap(ax=axes[1],data=krig_error,vmin=min_error, vmax=max_error)
        axes[1].set_title = 'error kriging'
        #sns.heatmap(ax=axes[1],data=kriging_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=maxval)
        sns.heatmap(ax=axes[2],data=nn_error,vmin=min_error, vmax=max_error)
        axes[2].set_title = 'error NN'
        plt.legend()
        plt.savefig(f'plots/a_error_x{sampling_distance_x}_y{sampling_distance_y}.png') 
        plt.close()   




pd.DataFrame(mses).to_csv('mses.csv')
"""
# Full map
mses ={'lin':[], 'krig':[], 'nn':[]}

for sampling_distance_y in [12,100]:
    for sampling_distance_x in [4,10]:

        length = int(sampling_distance_y/12 * 1e3)


        whole_map = pd.read_csv('WholeMap_Rounds_40_to_17.csv')
        length = int(length)
        whole_map = whole_map.astype(float).set_index('Unnamed: 0')
        whole_map = whole_map.reset_index().drop(columns=['Unnamed: 0'])
        whole_map.columns = [x for x in range(len(whole_map.columns))]

        whole_map = whole_map.iloc[:length]

        whole_map_stacked = whole_map.stack().reset_index()
        whole_map_stacked.columns = ['y', 'x', 'z']
        whole_map_stacked = whole_map_stacked[['x', 'y', 'z']]
        data = whole_map_stacked[whole_map_stacked['y' ]%(sampling_distance_y/2) == 0]

        # resampling on y axis
        data_i = data[data['y'] % int(sampling_distance_y) == 0]
        data_o = data[data['y'] % sampling_distance_y != 0]


        # resampling on x axis
        data_i = data_i[data_i['x'] % int(sampling_distance_x) == 0]

        data_o = data_o[data_o['x'] % int(sampling_distance_x/2)  == 0 ]
        data_o =data_o[ data_o['x'] % int(sampling_distance_x) != 0]
        if data_o['x'].max() > data_i['x'].max():
            data_o = data_o[data_o['x'] != data_o['x'].max()]
        data_o = whole_map_stacked
        x_train = data_i[['x','y']]
        y_train = data_i[['z']]
        x_val = data_o[['x','y']]
        y_val = data_o[['z']]
        trainedModelPath = f'trainedModels/samplerate_x{sampling_distance_x}_y{sampling_distance_y}_full/'
                # DeepKriging model for continuous data with 4 hidden layers
        if not os.path.exists(trainedModelPath):
                            os.makedirs(trainedModelPath)
                
        model_base = Sequential()
        model_base.add(Dense(100, input_dim=x_train.shape[1],  kernel_initializer='he_uniform', activation='relu'))
        model_base.add(Dropout(rate=0.5))
        model_base.add(BatchNormalization())
        model_base.add(Dense(100, activation='relu'))
        model_base.add(Dropout(rate=0.5))
        model_base.add(Dense(100, activation='relu'))
                #model_base.add(Dropout(rate=0.5))
        model_base.add(BatchNormalization())
        model_base.add(Dense(1, activation='linear'))
        opt = keras.optimizers.Adam(learning_rate=0.0001)
        model_base.compile(loss='mse', optimizer=opt, metrics=['mse','accuracy'])

        callbacks = create_callback(trainedModelPath)
        x_train = x_train.to_numpy()
        x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
        y_train = y_train.to_numpy()
        y_train = y_train.reshape(y_train.shape[0],1)
        x_val= x_val.to_numpy()
        x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
        y_val= y_val.to_numpy()
        y_val = y_val.reshape(y_val.shape[0],1)

        history = model_base.fit(x_train
                                , y_train
                                , epochs=1000
                                , validation_data = (x_val, y_val)
                                , callbacks=callbacks
                                )
        pd.DataFrame(history.history).to_csv(f'{trainedModelPath}/history_x{sampling_distance_x}_y{sampling_distance_y}.csv')
        model = tf.keras.models.load_model(trainedModelPath+"best_model.h5")
        prediction = model.predict(x_val)
                #mse_nn = np.mean((prediction- y_val.to_numpy())**2)

             #mses_nn.append( mse_nn)


                #np.save('mses_nn_r4.npy', np.array(mses_nn))
        np.save(f'predictions/NN_x{sampling_distance_x}_y{sampling_distance_y}_full.npy', prediction)

        pred_stacked = pd.DataFrame([prediction[:,0],data_o['x'].to_numpy(),data_o['y'].to_numpy()]).transpose()
        pred_stacked.columns = ['z','x','y']

        pred_stacked = pred_stacked[['x','y','z']]

        
        
        x = data_o['x']
        y = data_o['y']         
        xx, yy = np.mgrid[x.min():x.max()+1:sampling_distance_x, y.min():y.max()+1:sampling_distance_y]
        lin_interpol = griddata(data_i[['x', 'y']], # Points we know
                        data_i['z'], # Values we know
                        (xx.T, yy.T), # Points to interpolate
                        method='linear')
        lin_interpol= pd.DataFrame(lin_interpol)
        lin_interpol.to_csv(f'predictions/lin_interpol_x{sampling_distance_x}_y{sampling_distance_y}_full.csv')
        
        ground_truth = data_o.pivot_table(index='y', columns='x', values='z')

        V = skg.Variogram(data_i[['x','y']].values,data_i.z.values.flatten(), maxlag='median', n_lags=15, normalize=False, verbose=True)
        
        ok = skg.OrdinaryKriging(V, min_points=5, max_points=80, mode='exact')  
        
        field = ok.transform(x,y)
        kriging_stacked = pd.DataFrame([field,data_o['x'].to_numpy(),data_o['y'].to_numpy()]).transpose()
        kriging_stacked.columns = ['z','x','y']
        kriging_stacked  = kriging_stacked[['x','y','z']]

        np.save(f'krigingfield_x{sampling_distance_x}sampling_distance_y{sampling_distance_y}_full.npy',field)
        minval = np.min([np.min(kriging_stacked['z'].to_numpy()),np.min(ground_truth.to_numpy()),np.min(np.nan_to_num(lin_interpol)),np.min(pred_stacked['z'].to_numpy())])
        maxval = np.max([np.max(kriging_stacked['z'].to_numpy()),np.max(ground_truth.to_numpy()),np.max(np.nan_to_num(lin_interpol)),np.max(pred_stacked['z'].to_numpy())]) 

        lin_interpol.columns = ground_truth.columns.values
        lin_interpol.index = ground_truth.index.values

        lin = lin_interpol.stack().reset_index()
        lin.columns = ['y', 'x', 'z']
        lin = lin[['x', 'y', 'z']]
        print('c')
        lin.index = data_o.index
        print('c')
        fig, axes = plt.subplots(1, 4, figsize=(20, 8))
        sns.heatmap(ax=axes[0],data=ground_truth,vmin=minval-10, vmax=maxval)
        axes[0].title.set_text('ground truth')
        sns.heatmap(ax=axes[1],data=pd.DataFrame(lin_interpol),vmin=minval-10, vmax=maxval)
        axes[1].title.set_text('lin interpol')
        sns.heatmap(ax=axes[2],data=kriging_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval-10, vmax=maxval)
        axes[2].title.set_text('kriging')
        #sns.heatmap(ax=axes[1],data=kriging_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=maxval)
        sns.heatmap(ax=axes[3],data=pred_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval-10, vmax=0).set(title='Title of Plot')
        axes[3].title.set_text('NN')
        plt.savefig(f'plots/hm_x{sampling_distance_x}_y{sampling_distance_y}_full.png')
        plt.close()
        
        kriging_stacked.index = data_o.index
        #print('a')
        pred_stacked.index = data_o.index
       # print('a')
        wms = whole_map_stacked.copy()
        wms['z'] = 0
       # print('a')
        wms.loc[kriging_stacked.index] = np.nan

        kriging_whole_map = wms.combine_first(kriging_stacked)
        print('wms')
        pred_whole_map  = wms.combine_first(pred_stacked)
        
        lin_interpol_whole_map  = wms.combine_first(lin)
        gt = wms.combine_first(data_o)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 8))
        sns.heatmap(ax=axes[0],data=gt.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=maxval)
        axes[0].title.set_text('ground truth')
        sns.heatmap(ax=axes[1],data=lin_interpol_whole_map.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=maxval)
        axes[1].title.set_text('lin interpol')
        sns.heatmap(ax=axes[2],data=kriging_whole_map.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=maxval)
        axes[2].title.set_text('kriging')
        #sns.heatmap(ax=axes[1],data=kriging_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=maxval)
        sns.heatmap(ax=axes[3],data=pred_whole_map.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=0).set(title='Title of Plot')
        axes[3].title.set_text('NN')
        plt.savefig(f'plots/hm_full_x{sampling_distance_x}_y{sampling_distance_y}.png')
        
        

        mses['lin'].append(mse(lin['z'], data_o['z']))
        mses['krig'].append(mse(kriging_stacked['z'], data_o['z']))
        mses['nn'].append(mse(prediction, data_o['z']))


        interpol_error = rel_error(lin_interpol,ground_truth)
        krig_error = rel_error(kriging_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),ground_truth)
        nn_error = rel_error(pred_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),ground_truth)

            
        max_error = np.max([interpol_error.max(), krig_error.max(), nn_error.max()])
        min_error = np.min([interpol_error.min(), krig_error.min(), nn_error.min()])
        
            
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        sns.heatmap(ax=axes[0],data=interpol_error,vmin=min_error, vmax=max_error)
        axes[0].set_title='error lin interpol'
        sns.heatmap(ax=axes[1],data=krig_error,vmin=min_error, vmax=max_error)
        axes[1].set_title = 'error kriging'
        #sns.heatmap(ax=axes[1],data=kriging_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=maxval)
        sns.heatmap(ax=axes[2],data=nn_error,vmin=min_error, vmax=max_error)
        axes[2].set_title = 'error NN'
        plt.legend()
        plt.savefig(f'plots/error_x{sampling_distance_x}_y{sampling_distance_y}_full.png') 
        plt.close()   




pd.DataFrame(mses).to_csv('mses_full.csv')
"""
