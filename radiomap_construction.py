import numpy as np
import pandas as pd


def gen_multivariate_normal(cov,verbose=0):
    """
    sampling via multivariate normal distribution with no trend 

    cov: variance-covariance matrix
    """
    if verbose:
        print("--gen_multivariate_normal()--")
    L = np.linalg.cholesky(cov)
    z = np.random.standard_normal(len(cov))

    return np.dot(L, z)

def gen_varcov_matrix(x, y, dcor, sigma, verbose=0):
    """
    variance-covariance matrix for log-normal shadowing
    
    x, y:  vector for measurement location
    dcor:  correlation distance of shadowing [m]
    sigma: standard deviation [dB]
    """
    if verbose: 
        print("--gen_varcov_matrix()--")
    dmat = distance(x, y, x[:, np.newaxis], y[:, np.newaxis]) # distance matrix
    tmp  = 0.6931471805599453 / dcor                          # np.log(2.0)/dcor

    return sigma * sigma * np.exp(-dmat * tmp)





def gen_location_vector(n_node, len_area, verbose=0):
    """
    for measurement location
    n_node:   number of nodes
    len_area: area length [m]
    """
    if verbose:
        print("--gen_location_vector()--")
    x = np.random.uniform(0.0, len_area, n_node)
    y = np.random.uniform(0.0, len_area, n_node)
    
    return x, y

def gen_combinations(arr):
      r, c = np.triu_indices(len(arr), 1)

      return np.stack((arr[r], arr[c]), 1)
  
def gen_emprical_semivar(data, d_max, num, verbose=0):
    
    if verbose:
        print("--gen_emprical_semivar()--")
    d_semivar   = np.linspace(0.0, d_max, num)
    SPAN        = d_semivar[1] - d_semivar[0]

    indx        = gen_combinations(np.arange(N))
    
    '''gen semivariogram clouds'''
    d           = distance(data[indx[:, 0], 0], data[indx[:, 0], 1], data[indx[:, 1], 0], data[indx[:, 1], 1])
    indx        = indx[d<=d_max]
    d           = d[d <= d_max]
    semivar     = (data[indx[:, 0], 2] - data[indx[:, 1], 2])**2

    '''average semivariogram clouds via binning'''
    semivar_avg = np.empty(num)
    for i in range(num):
        d1 = d_semivar[i] - 0.5*SPAN
        d2 = d_semivar[i] + 0.5*SPAN
        indx_tmp = (d1 < d) * (d <= d2) #index within calculation span
        semivar_tr = semivar[indx_tmp]
        # semivar_avg[i] = semivar_tr.mean()
        if len(semivar_tr)>0:
            semivar_avg[i] = semivar_tr.mean()
        else:
            semivar_avg[i] = np.nan

    return d_semivar[np.isnan(semivar_avg) == False], 0.5 * semivar_avg[np.isnan(semivar_avg) == False]




def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)



def pathloss(d, eta):
    return 10.0 * eta * np.log10(d+1.0) #+1: to avoid diverse of path loss


def OLS(d, p):
    '''Ordinary Least Squares for Path Loss Modeling'''
    A       = np.vstack([-10.0*np.log10(d+1.0), np.ones(len(p))]).T
    m, c    = np.linalg.lstsq(A, p, rcond=None)[0]
    return m, c



if __name__=="__main__":

    '''measurement configuration'''
    LEN_AREA    = 20
    DCOR        = 0.8 # 20.0      #correlation distance [m]
    STDEV       = 4.7             #standard deviation
    TX_X        = -0.3            #x coordinate of transmitter
    TX_Y        = 0.5 * LEN_AREA  #y coordinate of transmitter
    PTX         = 23.0            #transmission power [dBm]
    ETA         = 1.5             #path loss index

 
    
    '''get measurement dataset'''
    x_min = 0
    x_max = 91 
    num_vals_x = 92
    
    y_min = 0
    
    
    for i in range(5):
        
        y_max = y_min + 400

        x = [x/100 for x in range(x_min,x_max+1)]#np.linspace(0, LEN_AREA, N)
        y = [y/100 for y in range(y_min,y_max+1)]#np.linspace(0, LEN_AREA, N)
        
        xx, yy = np.meshgrid(x, y)
        
        print(f"creating map for y_min : {y_min} cm and y_max : {y_max} cm _> results in : grid shape : ", xx.shape)
        
        
        cov     = gen_varcov_matrix(xx.flatten(), yy.flatten(), DCOR, STDEV)

        z       = gen_multivariate_normal(cov)  #correlated shadowing vector[dB]

        d       = distance(TX_X, TX_Y, xx.flatten(), yy.flatten())
        l       = pathloss(d, ETA)              #[dB]
        prx     = PTX - l + z                   #received signal power [dBm]

    
        
        
        
        stackedmap = pd.DataFrame({'x':list(xx.flatten()), 'y':list(yy.flatten()), 'z':prx})
        
        
     
        stackedmap = stackedmap.sort_values(by=['x','y'])
        
        stackedmap.to_csv(f'virtualmainmap_{y_min}_to_{y_max}.csv')
        print(f"virtual main map saved to virtualmainmap_{y_min}_to_{y_max}.csv")
        y_min = y_max + 1