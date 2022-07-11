import math
import os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf

import basemodel as bm
def minmax(array, array_max, array_min):

    """
    _summary_
    Args:
        array ([np.array or pandas df]):
        array_max ([float]):maxval
        array_min ([float]): minval

    Returns:
        [np.array or pandas df]: 0-1, minmax normalized array
    """

    return (array-array_min)/(array_max-array_min)

def reminmax(array, array_max, array_min):

    """
    _summary_
    Args:
        array ([np.array or pandas df]):
        array_max ([float]):maxval
        array_min ([float]): minval

    Returns:
        [np.array or pandas df]: 0-1, minmax renormalized array
    """

    return array * (array_max-array_min) + array_min

def normalize_data(map_not_normalized, known_points, unknown_points):

    """

    _summary_

        Args:
            map_not_normalized (pandas DataFrame) : map with all points in x,y,z format

        Returns:
            map minmax normalized
            normvals

    """

    maxvals = {'x': map_not_normalized['x'].max(),
               'y': map_not_normalized['y'].max(),
               'z': map_not_normalized['z'].max()}

    minvals = {'x': map_not_normalized['x'].min(),
               'y': map_not_normalized['y'].min(),
               'z': map_not_normalized['z'].min()}

    map_normalized = pd.DataFrame()
    for col in known_points.columns:

        known_points[col] = minmax(known_points[col], maxvals[col], minvals[col])
        unknown_points[col] = minmax(unknown_points[col], maxvals[col], minvals[col])
        map_normalized[col] = minmax(map_not_normalized[col], maxvals[col], minvals[col])
    return map_normalized,known_points, unknown_points, maxvals, minvals


def calc_h_for_num_basis(number_points, n_dimensions=2, verbose=False):


    """
        _summary_

        Args:
            number_points (int) : number of points
            n_dimensions (int, optional) : Dimensions of points  Defaults to 2 (x,y).
            verbose(bool): show H

        Returns:
            h_for_num_basis (float

        _description_
            This Function calculates the Number of Elements in num_basis)

    """

    h_for_num_basis = 1 + (np.log2( number_points**(1/n_dimensions) / 10 ))

    if verbose:
        print("H: ", h_for_num_basis)

    return math.ceil(h_for_num_basis)


def get_numbasis(num_elements , n_dimensions=2, verbose=False):

    """
      _summary_

            Args:
                N (int) : number of points
                num_elements (float) : ( Number of Elements in num_basis)
                n_dimensions: number of dimensions of the coordinates

            Returns:
                num_basis (list): list of number of basis functions for each dimension

    _description_

            This function returns a list of number of basis functions for each dimension
            as descriped in : " https://arxiv.org/pdf/2007.11972.pdf " on page 12



    """


    numbasis = []

    for i in range(1,int(num_elements+1)):
        k = (9 * 2**(i-1) + 1 )
        numbasis.append(int(k)**n_dimensions)

    if verbose:
        print(f"amount base fct: {sum(numbasis)}")
    return numbasis


def findworkingnumbasis(len_data, num_elements, n_dimensions=2, verbose=False):

    """
    _summary_

        Args:
            len_data (int) : number of points
            num_elements (float) : ( Number of Elements in num_basis)
            n_dimensions (int, optional) : Dimensions of points  Defaults to 2 (x,y).
            verbose (bool): show info about recursion

        Returns:
            numbasis (list): list of number of basis functions for each dimension

    _description_
            Because the number of basis functions will become very high for a hugh Data set,
            this function will check recursively,
            if the number of basis functions is too high
            this fct will decrease the number of basis functions
            until it is below the maximum number of basis functions.

    """

    # Create ArrayMemoryError class to catch a custom exception
    # ArrayMemoryError is a numpy specific exception)
    class ArrayMemoryError(Exception):
        pass

    try:
        numbasis = get_numbasis( num_elements, n_dimensions, verbose)

        testvariable = np.zeros((len_data, int(sum(numbasis))),dtype=np.float32)
        del testvariable    # delete testvariable to free up memory



    except np.core._exceptions._ArrayMemoryError:
        if verbose:
            print('Error : Not enough memory to create basis functions')
            print('try to reduce H by 1')

            numbasis = findworkingnumbasis(len_data, (num_elements-1) , n_dimensions=2)

    return numbasis

def wendlandkernel(points, numbasis):

    """
    _summary_

        Args:
            points (pandas DataFrame) : cordinnates in format x, y
            numbasis (int): number of basis functions
        Returns:
            phi (pandas DataFrame): matrix of shape N x number_of_basis_functions

    _description_

        This funkction applies the wendlandkernel to a set of points
        and returns a matrix of shape Nxnumber_of_basis_functions
        typicalls x and y represented all points in the entire map

    """

    # Fct begins here




    knots_1dx = [np.linspace(0,1,int(np.sqrt(i))) for i in numbasis]
    knots_1dy = [np.linspace(0,1,int(np.sqrt(i))) for i in numbasis]

    ##Wendland kernel

    basis_size = 0
    # Create matrix of shape N x number_of_basis_functions
    # Use np.float32 to save memory

    phi = np.zeros((len(points), int(sum(numbasis))),dtype=np.float32)



    for res in range(len(numbasis)):

        theta = 1/np.sqrt(numbasis[res])*2.5
        knots_x, knots_y = np.meshgrid(knots_1dx[res],knots_1dy[res])
        knots = np.column_stack((knots_x.flatten(),knots_y.flatten()))

        for i in range(int(numbasis[res])):

            d = np.linalg.norm(np.vstack((points.y,points.x)).T-knots[i,:],axis=1)/theta

            for j in range(len(d)):

                if d[j] >= 0 and d[j] <= 1:

                    phi[j,i + basis_size] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3

                else:

                    phi[j,i + basis_size] = 0

        basis_size = basis_size + numbasis[res]

    return pd.DataFrame(phi)



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
    model.add(Dense(100,
                    input_dim = input_dim,
                    kernel_initializer='he_uniform',
                    activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(Dense(100, activation='relu'))
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


if __name__ == '__main__':


    Npoints = 900
    h_test = calc_h_for_num_basis(Npoints)

    num_basis = findworkingnumbasis(Npoints,h_test)
    print(num_basis)
    expected =  [10**2,19**2,37**2]
    print(expected)

    if num_basis == expected:
        print('from deepkriging.py: findworkingnumbasis: OK')

    # check if model building is working

    #test if model build is working

    deepkrigingmodel = build_model(sum(num_basis),verbose=True)
    