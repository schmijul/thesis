
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

def singleHeatMap(data, map, known_points, unknown_points, maxValue, minValue, figsize=(12,8), cmap='viridis'):
    """
    
    _summary_

        Args:
            data (dict): dictionary of matrices to be plotted
            data.keys() (list): list of keys in data will be used as title for subplots
            data[key] (np-array): matrix to be plotted
                
            map (pandas DataFrame): matrix of the map to be plotted in format x, y, z
            
            known_points (pandas DataFrame)): will be used to mark known points on the map 
            
            unknown_points (pandas DataFrame): will be used to show the target heatmap in format x, y, z

            maxValue (float): maximum value of the heatmap scala
            minValue (float): minimum value of the heatmap scala
            
    """
    
    
    
    
    
    
    
    plt.figure(figsize=figsize)
    
    ax1 = plt.subplot2grid((1,3), (0,0))
    ax2 = plt.subplot2grid((1,3), (0,1))
    ax3 = plt.subplot2grid((1,3), (0,2))
    
    sns.heatmap(ax=ax1,
                data=map.pivot_table(index='y', columns='x', values='z'),
                vmin=minValue,
                vmax=maxValue,
                cmap=cmap)
    
    ax1.plot(known_points['x'], known_points['y'], 'k.', ms=1)
    
    ax1.set_title('whole_map with known points marked')
    
    sns.heatmap(ax=ax2,
                data=unknown_points.pivot_table(index='y', columns='x', values='z'),
                vmin=minValue,
                vmax=maxValue,
                cmap=cmap)
    
    ax2.set_title('target heatmap')
    
    title = list(data.keys())[0]
    sns.heatmap(ax=ax3,
                data=data[title].pivot_table(index='y', columns='x', values='z'),
                vmin=minValue,
                vmax=maxValue,
                cmap=cmap)
    ax3.set_title(title)
    
    plt.legend()
    
def multipleHeatMapsNoBaseMap(data, maxValue, minValue,figsize=(16,7),cmap='viridis'):
    
    """
    _summary_
    
        Args:
            data (dict): dictionary of matrices to be plotted
            data.keys() (list): list of keys in data will be used as title for subplots
            data[key] (np-array): matrix to be plotted
              
            maxValue (float): maximum value of the heatmap scala
            minValue (float): minimum value of the heatmap scala    
    
    """      
    keys = list(data.keys())
    figsize=figsize
    
    plt.figure(figsize=figsize)
    
    
    for i in range(len(keys)):
        
        ax = plt.subplot2grid((1,len(keys)), (0,i))
        
        sns.heatmap(ax=ax,
                    data=data[keys[i]].pivot_table(index='y', columns='x', values='z'),
                    vmin=minValue,
                    vmax=maxValue,
                    cmap=cmap)
        ax.set_title(keys[i])
    
    plt.legend()

def multipleHeatMaps(data, map, known_points, unknown_points, maxValue, minValue, cmap='viridis'): 
        
    """
    _summary_
    
        Args:
            data (dict): dictionary of matrices to be plotted
            data.keys() (list): list of keys in data will be used as title for subplots
            data[key] (np-array): matrix to be plotted
                
            map (pandas DataFrame): matrix of the map to be plotted in format x, y, z
            
            known_points (pandas DataFrame)): will be used to mark known points on the map 
            
            unknown_points (pandas DataFrame): will be used to show the target heatmap in format x, y, z
            
            maxValue (float): maximum value of the heatmap scala
            minValue (float): minimum value of the heatmap scala

    
    """    
    
    
    keys = list(data.keys())
    
    figsize=(4*len(keys),16)
    
    plt.figure(figsize=figsize)
    
    ax1 = plt.subplot2grid((2,len(keys)), (0,int(len(keys)/2-1)))
    ax2 = plt.subplot2grid((2,len(keys)), (0,int(len(keys)/2)))
    
    sns.heatmap(ax=ax1,
                data=map.pivot_table(index='y', columns='x', values='z'),
                vmin=minValue,
                vmax=maxValue,
                cmap=cmap)
    ax1.plot(known_points['x'], known_points['y'], 'k.', ms=1)
    ax1.set_title('whole_map with known points marked')
    
    sns.heatmap(ax=ax2,
                data=unknown_points.pivot_table(index='y', columns='x', values='z'),
                vmin=minValue,
                vmax=maxValue,
                cmap=cmap)
    ax2.set_title('target heat map')
    
    
    for i in range(len(keys)):
        
        ax = plt.subplot2grid((2,len(keys)), (1,i))
        
        sns.heatmap(ax=ax,
                    data=data[keys[i]].pivot_table(index='y', columns='x', values='z'),
                    vmin=minValue,
                    vmax=maxValue,
                    cmap=cmap)
        ax.set_title(keys[i])
    
    plt.legend()
    
    
    
                  
   
    
def generateHeatMaps(data, map, known_points, unknown_points , maxValue, minValue,showMap, path):
    
    '''
    _summary_

        Args:
            data (dict): dictionary of matrices to be plotted
            data.keys() (list): list of keys in data will be used as title for subplots
            data[key] (np-array): matrix to be plotted
                
            map (pandas DataFrame): matrix of the map to be plotted in format x, y, z
            
            known_points (pandas DataFrame): will be used to mark known points on the map
            
            unknown_points (pandas DataFrame): will be used to show the target heatmap
            
            maxValue (float): maximum value of the heatmap scala
            minValue (float): minimum value of the heatmap scala
    
    '''
    
    keys = list(data.keys())
    
    # If there is only one key in data, then it is a single matrix There fore the Fct will plot 3 heatmaps ( map, unknown_points, data )
    
    if len(keys) == 1:
        
        singleHeatMap(data, map, known_points, unknown_points, maxValue, minValue)
    
    
    else:
        if showMap:
            multipleHeatMaps(data, map, known_points, unknown_points, maxValue, minValue)
        else:
            multipleHeatMapsNoBaseMap(data, maxValue, minValue)
                    
    plt.savefig(path)
                   
