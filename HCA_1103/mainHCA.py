import pandas as pd
import numpy as np
import graphicsHCA as g
import utilsHCA as utl
import scipy.cluster.hierarchy as hclust
import scipy.spatial.distance as hdist
import sklearn.decomposition as dec
import matplotlib as mpl


fileName = './dataIN/Indicators_EN.csv'

# mpl.rcParams['figure.max_open_warning'] = 50
# print(mpl.rcParams['figure.max_open_warning'])

table = pd.read_csv(fileName, index_col=0)
print(table)

obs = table.index.values
print(obs, type(obs), obs.shape)
n = len(obs)
print('No. of observations:', n)

vars = table.columns[1:].values
print(vars, type(vars), vars.shape)
m = len(vars)
print('No. of variables:', m)

table_nda = table[vars].values
print(type(table_nda), table_nda.shape)

X = utl.replace_na(table_nda)
X_std = utl.standardise(X)
X_std_df = pd.DataFrame(data=X_std, index=obs, columns=vars)
X_std_df.to_csv('./dataOUT/X_std.csv')

# create a list of clustering methods
methods = list(hclust._LINKAGE_METHODS)
print(methods, type(methods))

# create a list of metrics
metrics = hdist._METRICS_NAMES
print(metrics, type(metrics))

# determine the cluster of observations
h_1 = hclust.linkage(y=X_std, method='single', metric='cityblock')
print(h_1, type(h_1))
# compute the threshold, the junction at which the partition of maximum
# stability occurs, and the maximum number of junctions
threshold, j, k = utl.threshold(h_1)

# create the dendrogram graphic
g.dendrogram(h=h_1, labels=obs,
    title="Hierarchical classification - method='single', metric='cityblock'",
    threshold=threshold, colors=None)
# g.show()

# determine the cluster of variables
h_2 = hclust.linkage(y=X_std.T, method='complete', metric='correlation')
print(h_2, type(h_2))
# compute the threshold, the junction at which the partition of maximum
# stability occurs, and the maximum number of junctions
threshold, j, k = utl.threshold(h_2)

# create the dendrogram graphic
g.dendrogram(h=h_2, labels=vars,
    title="Hierarchical classification - method='median', metric='correlation'",
    threshold=threshold, colors=None)
g.show()

# TODO
