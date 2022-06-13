from curses import mouseinterval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import v
import seaborn as sns

def singleHeatMap(data, map, known_points, unknown_points, figsize=(12,8), cmap='viridis'):
    """_summary_

    Args:
        data (dict): data to plot key = title, value = np.array
        map (np.array): whole map 
        known_points (np.array): coordinate pairs of known points to plot them onto the whole map    
        unknown_points (np.array): target values 

    Returns:
        _return_: matplotlib.pyplot.figure: _description_ 
    """
    minval = -55
    maxval = 0
    plt.figure(figsize=figsize)
    
    ax1 = plt.subplot2grid((1,3), (0,0))
    ax2 = plt.subplot2grid((1,3), (0,1))
    ax3 = plt.subplot2grid((1,3), (0,2))
    
    sns.heatmap(ax=ax1,
                data=map.pivot_table(index='y', columns='x', values='z'),
                vmin=minval,
                vmax=maxval,
                cmap=cmap)
    ax1.plot(known_points['x'], known_points['y'], 'k.', ms=1)
    ax1.set_title('whole_map with known points marked')
    
    sns.heatmap(ax=ax2,
                data=unknown_points.pivot_table(index='y', columns='x', values='z'),
                vmin=minval,
                vmax=maxval,
                cmap=cmap)
    
    ax2.set_title('target heatmap')
    
    title = list(data.keys())[0]
    sns.heatmap(ax=ax3,
                data=data[title].pivot_table(index='y', columns='x', values='z'),
                vmin=minval,
                vmax=maxval,
                cmap=cmap)
    ax3.set_title(title)
    
    plt.legend()
    
          

def multipleHeatMaps(data, map, known_points, unknown_points, cmap='viridis'): 
    
    """
    _summary_
    
        Args:
            data (dict): dictionary of matrices to be plotted
            data.keys() (list): list of keys in data will be used as title for subplots
            data[key] (np-array): matrix to be plotted
                
            map (np-array): matrix of the map to be plotted
            
            known_points (np-array): will be used to mark known points on the map
            
            unknown_points (np-array): will be used to show the target heatmap

    
    """    
    minval = -55
    maxval = 0
    
    keys = list(data.keys())
    
    figsize=(4*len(keys),16)
    
    plt.figure(figsize=figsize)
    
    ax1 = plt.subplot2grid((2,len(keys)), (0,int(len(keys)/2-1)))
    ax2 = plt.subplot2grid((2,len(keys)), (0,int(len(keys)/2)))
    
    sns.heatmap(ax=ax1,
                data=map.pivot_table(index='y', columns='x', values='z'),
                vmin=minval,
                vmax=maxval,
                cmap=cmap)
    ax1.plot(known_points['x'], known_points['y'], 'k.', ms=1)
    ax1.set_title('whole_map with known points marked')
    
    sns.heatmap(ax=ax2,
                data=unknown_points.pivot_table(index='y', columns='x', values='z'),
                vmin=minval,
                vmax=maxval,
                cmap=cmap)
    ax2.set_title('target heat map')
    
    
    for i in range(len(keys)):
        
        ax = plt.subplot2grid((2,len(keys)), (1,i))
        
        sns.heatmap(ax=ax,
                    data=data[keys[i]].pivot_table(index='y', columns='x', values='z'),
                    vmin=minval,
                    vmax=maxval,
                    cmap=cmap)
        ax.set_title(keys[i])
    
    plt.legend()
    
    
    
                  
   
    
def generateHeatMaps(data, map, known_points, unknown_points , path):
    
    '''
    _summary_

        Args:
            data (dict): dictionary of matrices to be plotted
            data.keys() (list): list of keys in data will be used as title for subplots
            data[key] (np-array): matrix to be plotted
                
            map (pandas DataFrame): matrix of the map to be plotted in format x, y, z
            
            known_points (np-array): will be used to mark known points on the map
            
            unknown_points (np-array): will be used to show the target heatmap
    
    '''
    
    keys = list(data.keys())
    
    # If there is only one key in data, then it is a single matrix There fore the Fct will plot 3 heatmaps ( map, unknown_points, data )
    
    if len(keys) == 1:
        
        singleHeatMap(data, map, known_points, unknown_points)
    
    
    else:
        
        multipleHeatMaps(data, map, known_points, unknown_points)
                    
    plt.savefig(path)
                   
if __name__ == '__main__':
    
    import data_preparation as dp
    
    
    wholeMap = pd.read_csv('WholeMap_Rounds_40_to_17.csv')
    
    sampling_distance_x = 4
    sampling_distance_y = 4 * 12
    
    Map, StackedMap = dp.prepare_map(wholeMap.copy())

    known_points, unknown_points = dp.resample(StackedMap.copy(), sampling_distance_x, sampling_distance_y)
    
    path='plot.png'
    
    
    generateHeatMaps({'test':unknown_points, 'test2':unknown_points}, StackedMap, known_points, unknown_points,path)
    
    