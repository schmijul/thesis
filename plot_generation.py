from locale import normalize
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from interpolation import *
from data_preparation import *
from trainNN import *
import skgstat as skg


def build_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def path_for_plot(sampling_distance_y, sampling_distance_x, length,random=False):
    if random:
        path = f'plots/RandomResampling_x{sampling_distance_x}cm_y{sampling_distance_y}_cm_length{length}cm/'

    else:
        path = f'plots/UnifromResampling{sampling_distance_x}cm_y{sampling_distance_y}_cm_length{length}cm/'
    build_path(path)
    return path

if __name__ == '__main__':

        # use argparse to parse command line arguments
    """
        parser = argparse.ArgumentParser()
        parser.add_argument('-x', '--sampling_distance_x', type=float, default=4, help='sampling distance in x direction')
        parser.add_argument('-y', '--sampling_distance_y', type=float, default=24, help='sampling distance in y direction')
        parser.add_argument('-l','--length',type=int,default=7000, help= 'length of the map in cm')
        parser.add_argument('-v','--verbose',type=bool,default=False,help='verbose mode 0/1')
        parser.add_argument('-e','--epochs',type=int,default=1500,help='number of epochs')
        args = parser.parse_args()
    """
    sampling_distance_y = 12
    sampling_distance_x = 2
    length = 500
    epochs = 1500
    verbose = True
    normalize = True
    random = False
    
    for interpolate_whole_map in [False]:#,True]:
        
                
            for start_point in in [0,501,1002, 1503,2004,2505,3006]:
                
                for i in [6,4,2,1]:
                    for random in [False, True]:
                        
                    
                        
                        if verbose:
                            print(f' amount rows : {length}')
                            print(f' length : {length} cm')
                            print(f' sampling_distance_y in rows : {i*sampling_distance_y}')
                            print(f' sampling_distance_y in cm : {i}')
                            print(f' sampling_distance_x in rows : {i * sampling_distance_x}')
                            print(f' sampling_distance_x in cm : {i}')
                            
                            
                        """
                        sampling_distance_x = args.sampling_distance_x
                        sampling_distance_y = args.sampling_distance_y
                        length = args.length
                        verbose = args.verbose
                        epochs = args.epochs
        
                        """

                        sampling_distance_y = 12 *i
                        sampling_distance_x =  i
                        whole_map = pd.read_csv('WholeMap_Rounds_40_to_17.csv')
                        map = stack_map(whole_map) # create a stacked map dataframe with columns x, y, z
                        map = cut_map_len(map,start_point,length) # cut the map to the length of the map
                        data_i, data_o = resample(map, sampling_distance_x, sampling_distance_y) # resample the map
                        if random:
                            # As first random samping approach this will use randomly sampled known coordinates and values to predict the same unknown values
                            data_i = randomsampling(map,data_i['x'].min(),data_i['x'].max(),data_i['y'].min(),data_i['y'].max(),sampling_distance_x, sampling_distance_y,data_o,verbose=verbose)

                        if interpolate_whole_map:
                            data_o = map
                        # Lin interpolation

                        lin_interpol, lin_interpol_stacked = grid_interpolation(data_i, data_o,verbose=verbose)
                        lin_interpol = lin_interpol
                        # Kriging interpolation
                        print('a')
                        kriging_interpol, kriging_interpol_stacked = kriging_skg(data_i, data_o, 10 , verbose=verbose)
                        kriging_interpol=kriging_interpol
                        
                        #### Next line can cause a lot of trouble as possibly hundres of gb memorry are needed for calc
                        
                        kriging_VariogramWithWholeMap, kriging_VariogramWithWholeMap_stacked = kriging_skg(map, data_o,10, verbose=verbose)
                        

                        # NN prediction
                        x_train = data_i[['x', 'y']]
                        y_train = data_i[['z']]

                        x_val = data_o[['x', 'y']]
                        y_val = data_o[['z']]
                        if normalize:
                            minval = np.max([y_train.max(), y_val.max()])
                            y_train = y_train/minval
                            y_val = y_val/minval

                        model =  train(x_train, y_train, x_val, y_val,length,build_model(x_train),epochs,sampling_distance_x, sampling_distance_y, verbose=verbose)

                        predictions, pred_stacked = prediction(model,data_o, x_val, sampling_distance_x, sampling_distance_y)
                        if normalize:
                            predictions = predictions * minval
                            pred_stacked = pred_stacked * minval



                        # plot the results
                        path = path_for_plot(i, i, length,random=random)
                        f= open(f"{path}/description_x{i}cm_y{i}cm_length{length}cm.txt","w+")
                        f.write("description of the data\n")
                        if random:
                            f.write(f"sampling is done randomly\n")
                        else:
                            f.write(f"sampling is not done randomly\n")
                        f.write(f"sampling_distance_x: {i} cm\n")
                        f.write(f"sampling_distance_y: {i} cm\n")
                        f.write(f"length in y direction : {length} cm \n")
                        f.write(f"amount known points : {len(data_i)} \n")
                        f.write(f"amount unknown points : {len(data_o)} \n")
                        

                        f.close()

                        minval =np.min([data_o['z'].min(), lin_interpol_stacked['z'].min(), kriging_interpol_stacked['z'].min(), pred_stacked['z'].min()])
                        maxval = np.max([data_o['z'].max(), lin_interpol_stacked['z'].max(), kriging_interpol_stacked['z'].max(), pred_stacked['z'].max()])
                        
                        """
                        if minval * 10 < data_o.z.min():
                            minval = data_o.z.min()-10
                        fig, axes = plt.subplots(1, 4, figsize=(20, 8))
                        sns.heatmap(ax=axes[0],data=data_o.pivot_table(index='y', columns='x', values='z'),vmin=minval-10, vmax=maxval)
                        axes[0].title.set_text('ground truth')
                        sns.heatmap(ax=axes[1],data=lin_interpol,vmin=minval-10, vmax=maxval)
                        axes[1].title.set_text('lin interpol')
                        sns.heatmap(ax=axes[2],data=kriging_interpol,vmin=minval-10, vmax=maxval)
                        axes[2].title.set_text('kriging')
                        #sns.heatmap(ax=axes[1],data=kriging_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=maxval)
                        sns.heatmap(ax=axes[3],data=predictions,vmin=minval-10, vmax=maxval)
                        axes[3].title.set_text('NN')
                        
                        
                        """
                        
                        #plot each prediction 
                        sns.heatmap(data=data_o.pivot_table(index='y', columns='x', values='z'),vmin=minval-10, vmax=maxval,cmap='viridis')
                        plt.savefig(f"{path}/target_points_x_{sampling_distance_x}_y_{sampling_distance_y}_length_{length}cm.png")
                        plt.close()
                        sns.heatmap(data=lin_interpol,vmin=minval-10, vmax=maxval,cmap='viridis')
                        plt.savefig(f"{path}/lin_interpol_x_{sampling_distance_x}_y_{sampling_distance_y}_length_{length}cm.png")
                        plt.close()
                        sns.heatmap(data=kriging_interpol,vmin=minval-10, vmax=maxval,cmap='viridis')
                        plt.savefig(f"{path}/kriging_interpol_x_{sampling_distance_x}_y_{sampling_distance_y}_length_{length}cm.png")
                        
                        plt.close()
                        sns.heatmap(data=predictions,vmin=minval-10, vmax=maxval,cmap='viridis')
                        plt.savefig(f"{path}/nn_x_{sampling_distance_x}_y_{sampling_distance_y}_length_{length}cm.png")
                        plt.close()
                        sns.heatmap(data=kriging_VariogramWithWholeMap,vmin=minval-10, vmax=maxval,cmap='viridis')
                        plt.savefig(f"{path}/kriging_VariogramWithWholeMap_x_{sampling_distance_x}_y_{sampling_distance_y}_length_{length}cm.png")
                        plt.close()
                        
                        
                        
                        
                        
                        
                        
                        plt.figure(figsize=(18,16))
                        ax1 = plt.subplot2grid((2,4), (0, 1))
                        ax2 = plt.subplot2grid((2,4), (0, 2))
                        ax3 = plt.subplot2grid((2,3), (1, 0))
                        ax4 = plt.subplot2grid((2,3), (1, 1))
                        ax5 = plt.subplot2grid((2,3), (1, 2))
                        #ax6 = plt.subplot2grid((2,4), (1, 3))
                        
                        sns.heatmap(ax=ax1,data=map.pivot_table(index='y', columns='x', values='z'),vmin=minval-10, vmax=maxval,cmap='viridis')
                        ax1.plot(data_i['x'], data_i['y'], 'k.', ms=1)
                        ax1.set_title('whole_map with known points marked')
                        
                        sns.heatmap(ax=ax2,data=data_o.pivot_table(index='y',columns='x', values='z'),vmin=minval-10, vmax=maxval,cmap='viridis')
                        ax2.set_title('target points')
                        
                        sns.heatmap(ax=ax3, data=lin_interpol,vmin=minval-10, vmax=maxval,cmap='viridis')
                        ax3.set_title('lin interpol')
                        
                        sns.heatmap(ax=ax4, data=kriging_interpol,vmin=minval-10, vmax=maxval,cmap='viridis')       
                        ax4.set_title(' kriging  ( Variogram with known points)')
                        
                        sns.heatmap(ax=ax5, data=kriging_VariogramWithWholeMap,vmin=minval-10, vmax=maxval,cmap='viridis')
                        ax5.set_title(' kriging ( Variogram with whole map)')
                        
                        sns.heatmap(ax=ax6, data=predictions,vmin=minval-10, vmax=maxval,cmap='viridis')
                        ax6.set_title(' Base NN')
                        
                        plt.legend()
                    
                        
                        
                        if interpolate_whole_map:
                            plt.savefig(f'{path_for_plot(i, i,length,random=random)}/Resampling_x{i}cm_y{i}cm_Heatmap_interpolatedwholemap_length{length}cm.png')

                        else:
                            plt.savefig(f'{path_for_plot(i, i,length,random=random)}/Resampling_x{i}cm_y{i}cm_Heatmap_length{length}cm.png')
                        
                        plt.close()
                        

                        interpol_error = rel_error(lin_interpol,data_o.pivot_table(index='y', columns='x', values='z'))
                        krig_error = rel_error(kriging_interpol,data_o.pivot_table(index='y', columns='x', values='z'))
                        kriging_VariogramWithWholeMap_error = rel_error(kriging_VariogramWithWholeMap,data_o.pivot_table(index='y', columns='x', values='z'))
                        nn_error = rel_error(predictions,data_o.pivot_table(index='y', columns='x', values='z'))


                        max_error = np.max([interpol_error.max(), krig_error.max(), nn_error.max(), kriging_VariogramWithWholeMap_error.max()])
                        min_error = np.min([interpol_error.min(), krig_error.min(), nn_error.min(),kriging_VariogramWithWholeMap_error.min()])


                        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

                        sns.heatmap(ax=axes[0],data=interpol_error,vmin=min_error, vmax=max_error,cmap='viridis')
                        axes[0].set_title='error lin interpol'
                        sns.heatmap(ax=axes[1],data=krig_error,vmin=min_error, vmax=max_error,cmap='viridis')
                        axes[1].set_title = 'error kriging ( Variogram with known points)'
                        #sns.heatmap(ax=axes[1],data=kriging_stacked.pivot_table(values='z', index=['y'], columns='x', aggfunc='first'),vmin=minval, vmax=maxval)
                        sns.heatmap(ax=axes[2],data=kriging_VariogramWithWholeMap_error,vmin=min_error, vmax=max_error,cmap='viridis')
                        axes[2].set_title = 'error kriging ( Variogram with whole map)'
                        sns.heatmap(ax=axes[3],data=nn_error,vmin=min_error,vmax=max_error,cmap='viridis')
                        axes[3].set_title = 'error NN'
                        plt.legend()
                        
                        
                        
                        if interpolate_whole_map:
                            plt.savefig(f'{path_for_plot(i, i,length, random=random)}/Resampling_x{i}cm_y{i}cm_Heatmap_Error_whole_map_interpolated_length{length}cm.png')

                        else:
                            plt.savefig(f'{path_for_plot(i, i,length, random=random)}/Resampling_x{i}cm_y{i}cm_Heatmap_Error_length{length}cm.png')
                        plt.close()

                        # calc mses
                        mse_lin = np.mean((lin_interpol_stacked['z'] - data_o['z']).dropna()**2)
                        mse_kriging = np.mean((kriging_interpol_stacked['z'] - data_o['z']).dropna()**2)
                        mse_kriging_VariogramWithWholeMap = np.mean((kriging_VariogramWithWholeMap_stacked['z'] - data_o['z']).dropna()**2)
                        mse_nn = np.mean((pred_stacked['z'] - data_o['z']).dropna()**2)
                        
                        f= open(f"{path}/mses_x{i}cm_y{i}cm_length{length}cm.txt","a+")
                        if interpolate_whole_map:
                            f.write(f"mse linear interpolation for wholemap : {mse_lin}\n")
                            f.write(f"mse kriging interpolation  for wholemap: {mse_kriging}\n")
                            f.write(f"mse kriging interpolation with variogram, that gets  whole map: {mse_kriging_VariogramWithWholeMap}\n")
                            f.write(f"mse NN prediction for wholemap: {mse_nn}\n")

                        else:
                            f.write(f"mse linear interpolation : {mse_lin}\n")
                            f.write(f"mse kriging interpolation: {mse_kriging}\n")
                            f.write(f"mse kriging interpolation with variogram, that gets whole map : {mse_kriging_VariogramWithWholeMap}\n")
                            f.write(f"mse NN prediction: {mse_nn}\n")
                        f.close()



                    
