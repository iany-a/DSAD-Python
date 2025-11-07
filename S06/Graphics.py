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



def show():
    plt.show()