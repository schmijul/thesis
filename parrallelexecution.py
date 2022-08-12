import os

import numpy as np
import multiprocessing as mp
import time



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

    h_for_num_basis = 1 + (np.log2(number_points ** (1 / n_dimensions) / 10))

    if verbose:
        print("H: ", h_for_num_basis)

    return math.ceil(h_for_num_basis)


def get_numbasis(num_elements, n_dimensions=2, verbose=False):
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

    for i in range(1, int(num_elements + 1)):
        k = (9 * 2 ** (i - 1) + 1)
        numbasis.append(int(k) ** n_dimensions)

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
        numbasis = get_numbasis(num_elements, n_dimensions, verbose)

        testvariable = np.zeros((len_data, int(sum(numbasis))), dtype=np.float32)
        del testvariable  # delete testvariable to free up memory



    except np.core._exceptions._ArrayMemoryError:
        if verbose:
            print('Error : Not enough memory to create basis functions')
            print('try to reduce H by 1')

            print(f"exception : {np.core._exceptions._ArrayMemoryError}")

            numbasis = findworkingnumbasis(len_data, (num_elements - 1), n_dimensions=2)

    return numbasis



def wendlandkernel(inputs):
    
    points_x = inputs[0]
    points_y = inputs[1]
    numbasis = inputs[2]
    
    
    
    knots_1dx = [np.linspace(0, 1, int(np.sqrt(i))) for i in numbasis]
    knots_1dy = [np.linspace(0, 1, int(np.sqrt(i))) for i in numbasis]
    
    
    
    ##Wendland kernel

    basis_size = 0

    # Create matrix of shape N x number_of_basis_functions
    # Use np.float32 to save memory

    phi = np.zeros((len(points_x), int(sum(numbasis))), dtype=np.float32)
    
    
    for res in range(len(numbasis)):

        theta = 1 / np.sqrt(numbasis[res]) * 2.5
        knots_x, knots_y = np.meshgrid(knots_1dx[res], knots_1dy[res])
        knots = np.column_stack((knots_x.flatten(), knots_y.flatten()))

        for i in range(int(numbasis[res])):

            d =np.linalg.norm(np.vstack((points_x, points_y)).T - knots[i, :], axis=1) / theta

            for j in range(len(d)):

                if d[j] >= 0 and d[j] <= 1:

                    phi[j, i + basis_size] = (1 - d[j]) ** 6 * (35 * d[j] ** 2 + 18 * d[j] + 3) / 3

                else:

                    phi[j, i + basis_size] = 0

        basis_size = basis_size + numbasis[res]
    
    return phi



    
if __name__ == "__main__":
    
    
    H = 5
    
    POINTS_X = np.linspace(0, 1, 10000)
    POINTS_Y = np.linspace(0, 1, 10000)
    NUM_PROCESSES =  mp.cpu_count() *5


    

    numbasis = get_numbasis(H)
    print(f"working with {sum(numbasis)} basis functions & {len(POINTS_X)} points\n")
    
    
    
    num_cores = mp.cpu_count() 
    print(f"available cores : {num_cores} \n") 
    
    
    # Datenvorbereitung für parralelisierung
    
    points_x = POINTS_X.reshape(-1, NUM_PROCESSES)
    points_y = POINTS_Y.reshape(-1, NUM_PROCESSES)
    
    inputs = [[points_x[:,i], points_y[:,i], numbasis] for i in range(NUM_PROCESSES)] # Daten für parralele Prozesse aufteilen
    
    # Hier ist der eigentliche Teil für die parrallel ausführtung

    
    pool = mp.Pool()
    start_time = time.time()

    

    list_of_phis = np.array(pool.map(wendlandkernel,inputs))
    mp_time = time.time() - start_time
    
    
    
    phi = list_of_phis.reshape(len(POINTS_X), sum(numbasis))
    
    np.save('phi_mp.npy', phi)
    
    
    
    
    print(f"mp time: {mp_time}\n")
    
    
    
    
    start_time = time.time()
    np.save('phi_singleCPU.npy', wendlandkernel([POINTS_X, POINTS_Y,numbasis]))
    singcpu_time =time.time() - start_time
    print(f"single core time: {singcpu_time}\n")
    
    
    print(f"single results == mp results ? : {(np.load('phi_singleCPU.npy') == np.load('phi_mp.npy')).any()}\n")
    
    
    
    print(f"mp computing was {singcpu_time/mp_time} times faster than single core")
   