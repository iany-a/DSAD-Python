import numpy as np
import seaborn as sb
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
def correlation_circle(R2, v1=0, v2=1, title='Correlation circle'):
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
        plt.scatter(x=R2[:, v1], y=R2[:, v2], color='Red')


def show():
    plt.show()