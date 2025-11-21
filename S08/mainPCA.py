import Functions as f
import pandas as pd
import Graphics as g
import matplotlib.pyplot as plt
from pca.PCA import PCA

#apply PCA on variables in Teritorial

table = pd.read_csv('./dataIN/Teritorial.csv', index_col=0)
print(table)

n=table.index.values.shape[0]
print('No. of obs:', n)
#collection of observation labels
obs = table.index.values
print(obs, type(obs), obs.shape)

#no of variables
m = table.columns[1:].size
print('No of observed variables:', m)
vars = table.columns[1:].values
print(vars, type(vars), vars.shape)

x = table[vars].values
print(type(x), x.shape)

#standardized matrix using function in Functions.py
x_std = f.standardize(x)
x_std_df = pd.DataFrame(data=x_std, index =(ob for ob in obs),
                        columns =(var for var in vars))
x_std_df.to_csv('./dataOUT/x_std.csv')

#create an instance of PCA class
pca_model = PCA(x_std)