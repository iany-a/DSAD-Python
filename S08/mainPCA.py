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
#extract the eigenvalues (the explained variance of principal components)
alpha = pca_model.getAlpha()
g.eigenvalues(val=alpha)
#g.show()
#extract the principal components
comp = pca_model.getComponents()
#save pca in csv
comp_df = pd.DataFrame(data=comp, index=(observation for observation in obs),
                       columns=('C'+str(j+1) for j in range(comp.shape[1])))
comp_df.to_csv('./dataOUT/PrincipalComponents.csv')

#create graphical representation of PCA
g.link_intensity(matrix=comp_df, title='Principal Components ')
g.show()

#extract the scores
scores = pca_model.getScores()
#save pca in csv
scores_df = pd.DataFrame(data=scores, index=(observation for observation in obs),
                       columns=('C'+str(j+1) for j in range(comp.shape[1])))
scores_df.to_csv('./dataOUT/Scores.csv')
g.link_intensity(matrix= scores_df, title='Scores standardized principal components', color='Blues')
g.show()