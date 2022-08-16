import matplotlib.pyplot as plt
import seaborn as sns

def singleheatmap(data, wholemap, unknown_points,
                  maxvalue, minvalue, figsize=(12,8), cmap='viridis'):
    """
    _summary_
        Args:
            data (dict): dictionary of matrices to be plotted
            data.keys() (list): list of keys in data will be used as title for subplots
            data[key] (np-array): matrix to be plotted

            wholemap (pandas DataFrame): matrix of the map to be plotted in format x, y, z

            known_points (pandas DataFrame)): will be used to mark known points on the map

            unknown_points (pandas DataFrame): will be used to show
            the target heatmap in format x, y, z
            maxvalue (float): maximum value of the heatmap scala
            minvalue (float): minimum value of the heatmap scala
    """

    plt.figure(figsize=figsize)

    ax_1 = plt.subplot2grid((1,3), (0,0))
    ax_2 = plt.subplot2grid((1,3), (0,1))
    ax_3 = plt.subplot2grid((1,3), (0,2))

    sns.heatmap(ax=ax_1,
                data=wholemap.pivot_table(index='y', columns='x', values='z'),
                vmin=minvalue,
                vmax=maxvalue,
                cmap=cmap)

    #ax_1.plot(known_points['x'], known_points['y'], 'k.', ms=1)

    ax_1.set_title('whole_map with known points marked')

    sns.heatmap(ax=ax_2,
                data=unknown_points.pivot_table(index='y', columns='x', values='z'),
                vmin=minvalue,
                vmax=maxvalue,
                cmap=cmap)

    ax_2.set_title('target heatmap')

    title = list(data.keys())[0]
    sns.heatmap(ax=ax_3,
                data=data[title].pivot_table(index='y', columns='x', values='z'),
                vmin=minvalue,
                vmax=maxvalue,
                cmap=cmap)
    ax_3.set_title(title)

    plt.legend()

def multipleheatmaps_nobasemap(data, maxvalue, minvalue,fig_size=(16,7),cmap='viridis'):
    """
    _summary_

        Args:
            data (dict): dictionary of matrices to be plotted
            data.keys() (list): list of keys in data will be used as title for subplots
            data[key] (np-array): matrix to be plotted

            maxvalue (float): maximum value of the heatmap scala
            minvalue (float): minimum value of the heatmap scala
    """
    keys = list(data.keys())

    plt.figure(figsize=fig_size)

    for i in range(len(keys)):

        ax_i = plt.subplot2grid((1,len(keys)), (0,i))

        sns.heatmap(ax=ax_i,
                    data=data[keys[i]].pivot_table(index='y',
                                                   columns='x',
                                                   values='z'),
                    vmin=minvalue,
                    vmax=maxvalue,
                    cmap=cmap)
        ax_i.set_title(keys[i])

    plt.legend()

def multipleheatmaps(data, stackedmap, known_points, unknown_points, maxvalue, minvalue, cmap='viridis'):
    """
    _summary_
        Args:
            data (dict): dictionary of matrices to be plotted
            data.keys() (list): list of keys in data will be used as title for subplots
            data[key] (np-array): matrix to be plotted

            stackedmap (pandas DataFrame): matrix of the map to be plotted in format x, y, z

            known_points (pandas DataFrame)): will be used to mark known points on the map

            unknown_points (pandas DataFrame): will be used to show the
            target heatmap in format x, y, z

            maxvalue (float): maximum value of the heatmap scala
            minvalue (float): minimum value of the heatmap scala

    """

    keys = list(data.keys())

    figsize=(4*len(keys),16)

    plt.figure(figsize=figsize)

    ax_1 = plt.subplot2grid((2,len(keys)), (0,int(len(keys)/2-1)))
    ax_2 = plt.subplot2grid((2,len(keys)), (0,int(len(keys)/2)))

    sns.heatmap(ax=ax_1,
                data=stackedmap.pivot_table(index='y', columns='x', values='z'),
                vmin=minvalue,
                vmax=maxvalue,
                cmap=cmap)
    ax_1.plot(known_points['x'], known_points['y'], 'k.', ms=1)
    ax_1.set_title('whole_map with known points marked')

    sns.heatmap(ax=ax_2,
                data=unknown_points.pivot_table(index='y', columns='x', values='z'),
                vmin=minvalue,
                vmax=maxvalue,
                cmap=cmap)
    ax_2.set_title('target heat map')


    for i in range(len(keys)):

        ax_i = plt.subplot2grid((2,len(keys)), (1,i))

        sns.heatmap(ax=ax_i,
                    data=data[keys[i]].pivot_table(index='y',
                                                   columns='x',
                                                   values='z'),
                    vmin=minvalue,
                    vmax=maxvalue,
                    cmap=cmap)
        ax_i.set_title(keys[i])

    plt.legend()


def generateheatmaps(data, stackedmap, known_points, unknown_points , maxvalue, minvalue,showmap, path):
    '''
    _summary_
        Args:
            data (dict): dictionary of matrices to be plotted
            data.keys() (list): list of keys in data will be used as title for subplots
            data[key] (np-array): matrix to be plotted

            stackedmap (pandas DataFrame): matrix of the map to be plotted in format x, y, z

            known_points (pandas DataFrame): will be used to mark known points on the map

            unknown_points (pandas DataFrame): will be used to show the target heatmap

            maxvalue (float): maximum value of the heatmap scala
            minvalue (float): minimum value of the heatmap scala
    '''

    keys = list(data.keys())

    # If there is only one key in data, then it is a single matrix
    # There for the Fct will plot 3 heatmaps ( map, unknown_points, data )

    if len(keys) == 1:
        singleheatmap(data, stackedmap, known_points, unknown_points, maxvalue, minvalue)

    else:
        if showmap:
            multipleheatmaps(data, stackedmap, known_points, unknown_points, maxvalue, minvalue)
        else:
            multipleheatmaps_nobasemap(data, maxvalue, minvalue)

    plt.savefig(path)
                   