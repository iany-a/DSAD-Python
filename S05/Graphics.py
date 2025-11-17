import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from openpyxl.styles.alignment import vertical_aligments


#Python Graph Gallery
#https://python-graph-gallery.com/
def correlogram(R2, dec=2, title='Correlogram', valMin=-1, valMax=1):
    plt.figure(num=title, figsize=(10,8))
    plt.title(label=title, fontsize=12, color='Blue', verticalalignment='bottom')
    sb.heatmap(data=R2, vmin=valMin, vmax=valMax, cmap='bwr', annot=True)

#correlation circle
def correlation_circle(R2, v1=0, v2=1, dec=2, title='Correlation circle'):
    plt.figure(num=title, figsize=(8,7))
    plt.title(label=title, fontsize=12, color='Blue', verticalalignment='bottom')
    #draw a circle of radius 1
    theta = [t for t in np.arange(start=0, stop=2*np.pi, step=0.01)]
    x = [np.cos(t) for t in theta]
    y = [np.sin(t) for t in theta]
    plt.plot(x, y)
    plt.axhline(y=0, color='Green')
    plt.axvline(x=0, color='Green')

    if isinstance(R2, np.ndarray):
        plt.xlabel(xlabel='Variable'+ str(v1 + 1), fontsize=10, color='Blue', verticalalignment='bottom')
        plt.ylabel(ylabel='Variable'+ str(v1 + 1), fontsize=10, color='Blue', verticalalignment='bottom')
        plt.scatter(x=R2[:, v1], y=R2[:, v2], color='Red')
        for i in range(R2.shape[0]):
            #same condition as before, except we take i
            plt.text(x=R2[i, v1], y=R2[i, v2], color='Black',
                     s='('+(str(np.round(R2[i,v1], decimals=dec))+', ' +
                        str(np.round(R2[i,v2], decimals=dec))+')'))
    elif isinstance(R2, pd.DataFrame):
        #axis labels
        plt.xlabel(xlabel=R2.columns[v1], fontsize=10, color='Blue', verticalalignment='bottom')
        plt.ylabel(ylabel=R2.columns[v2], fontsize=10, color='Blue', verticalalignment='bottom')
        #plt.scatter(x=R2.values[:,v1], y=R2.values[:,v2], color = 'Blue')
        plt.scatter(x=R2.iloc[:].iloc[v1], y=R2.iloc[:].iloc[v2], color = 'Blue')
        for i in range(R2.index.size):
            plt.text(x=R2.iloc[i, v1], y=R2.iloc[i, v2], color='Black',
                     s=R2.index[i])
    else:
        raise Exception('Invalid data type')

#simple line graphic
def explained_variance(eigenvalues, title='PCA - Explained variance by principal components'):
    plt.figure(num=title, figsize=(10,8))
    plt.title(label=title, fontsize=12, color='Blue', verticalalignment='bottom')
    plt.xlabel(xlabel='Principal components', fontsize=10, color='Blue', verticalalignment='bottom')
    plt.ylabel(ylabel='Explained variance', fontsize=10, color='Blue', verticalalignment='bottom')
    components = ['C'+str(i+1) for i in range(len(eigenvalues))]
    plt.plot(components, eigenvalues, 'o-')
    plt.axhline(y=1, color='Red')

def show():
    plt.show()