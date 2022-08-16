import math
import os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf

import basemodel as bm




def build_model(input_dim, verbose=False):

    """
    _summary_

        Args:
            input_dim (int): number of input dimensions
            which is the number of basis funtions -> phi.shape[1]
            verbose (bool): if True, print model summary

        Returns:
            Keras model ready for training

    """

    model = Sequential()
    model.add(Dense(1000,
                    input_dim = input_dim,
                    kernel_initializer='he_uniform',
                    activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(BatchNormalization())
    model.add(Dense(2500, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2500, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(Dense(2500, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse','mae'])

    if verbose:
        print(model.summary())

    return model

def train_model(model,
                x_train,
                y_train,
                x_val,
                y_val,
                scenario,
                epochs,
                batch_size=100,
                save_hist=False,
                verbose=False):

    """

     _summary_

        Args:
            model: compiled Keras model object
            x_train, x_val (pandas DataFrame): training and validation coordinates
            y_train, y_val (pandas DataFrame): training and validation
                values at specfific coordinates
            scenario (str): scenario of scenario -> to save model in correct directory
            epochs (int): number of epochs to train for
            batch_size (int): batch size for training data

            verbose (bool): if True, print more information

        Returns:
            trained model
            train-history as pandas DataFrame

    """


    # Checking input Args

    if not((len(x_train) == len(y_train)) and (len(x_val) == len(y_val))):
        print('Error x and y shapes do not match')
        return False

    if not ((len(x_train) != 0) or (len(y_train != 0))):
        print('Error empty Data ')
        return False

    # Fct begins here

    trainedmodelpath = f'trainedModels/deepkriging/{scenario}/'


    if not os.path.exists(trainedmodelpath):
        os.makedirs(trainedmodelpath)



    history = model.fit(x_train,y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        callbacks = bm.create_callbacks(trainedmodelpath,verbose=verbose),
                        verbose=verbose)

    if save_hist:
        pd.DataFrame(history.history).to_csv(f'{trainedmodelpath}/{scenario}_history.csv')

    return model, trainedmodelpath


def predict(trainedmodelpath, x_val):


    """
    _summary_

        Args:
                scenario (str): scenario of scenario this is used to
                load the best model for the choosen scenario
                x_val (pandas DataFrame): validation coordinates

        Returns:
                    predicted values at validation coordinates

    """

    # Checking input Args

    if not len(x_val) != 0:
        print('Error : x-val empty')
        return False


   # Fct begins here

    model = tf.keras.models.load_model(f'{trainedmodelpath}/best_model.h5')


    return model.predict(x_val)

