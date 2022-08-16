import os
import numpy as np
import tensorflow as tf
import pandas as pd
import interpolation_utils as iu
import datapreparation as dp


def load_model(iteration):
    return tf.keras.models.load_model(f"trainedModels/best_model_cv_iteration{iteration}.h5")  

def change_input_dim(model):

    model.save_weights("weights")

    input_shape = (None, None,1)
    
    model2 = tf.keras.Sequential()
    model2.add(tf.keras.layers.Conv2D(NUMBER_FILTERS_LAYER_1, kernel_size=(KERNEL_SIZE_LAYER_1, KERNEL_SIZE_LAYER_1), strides=(1, 1),
                     activation='relu',
                     padding='same',
                     input_shape=input_shape))
    model2.add(tf.keras.layers.Conv2D(NUMBER_FILTERS_LAYER_2, (KERNEL_SIZE_LAYER_2, KERNEL_SIZE_LAYER_2), strides=(1, 1), activation='relu',
                     padding='same'))
    model2.add(tf.keras.layers.Conv2D(NUMBER_FILTERS_LAYER_3, (KERNEL_SIZE_LAYER_3, KERNEL_SIZE_LAYER_3), strides=(1, 1), activation='relu',
                     padding='same'))

    model2.load_weights("weights")

    
    
    return model2



if __name__ == "__main__":
    
    # Global constants
    DISTANCE = 4 # 4 cm sampling distance
    MAP, STACKEDMAP = dp.preparemap(pd.read_csv("map_data/full_map_main.csv"))
    MAP = MAP.drop(columns=["Unnamed: 0"])

    Z_MAX = STACKEDMAP.z.max()
    Z_MIN = STACKEDMAP.z.min()
    STACKEDMAP.z = (STACKEDMAP.z-Z_MIN )/ (Z_MAX - Z_MIN)
    METHOD = "nearest"
    
    
    
    
        
    NUMBER_FILTERS_LAYER_1 = 256
    NUMBER_FILTERS_LAYER_2 = 128
    NUMBER_FILTERS_LAYER_3 = 1

    KERNEL_SIZE_LAYER_1 = 3
    KERNEL_SIZE_LAYER_2 = 1
    KERNEL_SIZE_LAYER_3 = 5

    # Globals
    
    
    
    downsampled_map = dp.resample(STACKEDMAP, DISTANCE, DISTANCE)[0]

    input_image = iu.gridinterpolation(downsampled_map, STACKEDMAP, method=METHOD).pivot_table(index='y', columns='x', values='z').to_numpy()

    # Change model input
    model = load_model(9)
    model.summary()
    
    newmodel = change_input_dim(model)
    
  
    prediction = newmodel.predict(input_image.reshape(input_image.shape[0], input_image.shape[1],1 ))[:,:,0,0]
    
    
    prediction = prediction*(Z_MAX - Z_MIN) + Z_MIN
    
    error = prediction -MAP.to_numpy()
    mae = np.mean(np.abs(error))
    
    print(f"MAE = ",mae)
