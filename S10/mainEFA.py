from operator import index

import numpy as np
import pandas as pd
from pyparsing import alphas

import utils as utl
import efa.EFA as efa
import factor_analyzer as fa
import Graphics as g
from sklearn.preprocessing import StandardScaler


tabel = pd.read_csv('dataIN/MortalityEU.csv',
                    index_col=0, na_values=':')
print(tabel)

# extract the row labels
obs = tabel.index.values
print(obs, type(obs))
# extract columns labels
vars = tabel.columns.values
print(vars, type(vars))
# the input matrix as numpy.ndarray
matrice_numerica = tabel.values
# print(matrice_numerica, type(matrice_numerica), matrice_numerica.shape)

# no. of observations
n = matrice_numerica.shape[0] # len(obs)
print('no. of observations:', n)
# no. of observed variables
m = matrice_numerica.shape[1]
print('no. of observed variables:', m)

# replace NAN
X = utl.replaceNAN(matrice_numerica)
X_df = pd.DataFrame(data=X, index=obs, columns=vars)
X_df.to_csv(path_or_buf='./dataOUT/X.csv')

# standardize X
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_std_df = pd.DataFrame(data=X_std, index=obs, columns=vars)
X_std_df.to_csv(path_or_buf='./dataOUT/X_std.csv')

# compute the Bartlett sphericity test
sphericity = fa.calculate_bartlett_sphericity(X_std)
print(sphericity, type(sphericity))
if sphericity[0] > sphericity[1]:
    print('There is at least one common factor!')
else:
    print('There is NO common factor!')
    exit(-1)

# compute Kaiser-Meyer-Olkin (KMO) indices
kmo = fa.calculate_kmo(X_std)
print(kmo, type(kmo))
# verify the global KMO index
if kmo[1] > 0.5:
    print('The observed variables can be factorized!')
else:
    print('The observed variables cannot be factorized!')
    exit(-2)

# convert a vector into a 1 column matrix
vector = kmo[0]
print(vector, type(vector), vector.shape)
matrix = vector[:, np.newaxis]
print(matrix, type(matrix), matrix.shape)
matrix_df = pd.DataFrame(data=matrix, index=vars, columns=['Indici KMO'])
g.intensity_map(matrix=matrix_df, title='Kaiser-Meyer-Olkin Indices')
# g.show()

efaModel = efa.EFA(X)
# identifying the number of significant factors
no_significant_factors = 2
chi2TabMin = 1
for k in range(no_significant_factors, m):
# for k in range(no_significant_factors, 5):
    faModel = fa.FactorAnalyzer(n_factors=k)
    faModel.fit(X=X_std)
    print(faModel.loadings_, faModel.get_uniquenesses())
    chi2Calc, chi2Tab = efaModel.computetBartlettTest(faModel.loadings_,
                            faModel.get_uniquenesses())
    print(chi2Calc, chi2Tab)

    if np.isnan(chi2Calc) or np.isnan(chi2Tab):
        break

    if chi2Tab < chi2TabMin:
        chi2TabMin = chi2Tab
        no_significant_factors = k

print('No. of significant factors:', no_significant_factors)

# create a model with the numbaer of significant factors determined
faModelFit = fa.FactorAnalyzer(n_factors=no_significant_factors)
faModelFit.fit(X=X_std)
# the factor loadings will provide the correlation between the observed variables
# and the significant factors
factor_loadings = faModelFit.loadings_
# save teh factor loadings in a CSV file
factor_loadings_df = pd.DataFrame(data=factor_loadings, index=vars,
                        columns=('F'+str(j+1) for j in range(no_significant_factors)))
factor_loadings_df.to_csv(path_or_buf='./dataOUT/FactorLoadings.csv')

# create the correlogram of factor loadings
g.correlogram(R2=factor_loadings_df, title='Factor loadings')
# g.show()

# extract the eigenvalues from FactorAnalyzer object
alpha = faModelFit.get_eigenvalues()
print(alpha, type(alpha))
g.eigenvalues(val=alpha[0])
# g.show()
# extract the scores (stadardized principal components)
scores = efaModel.getScoruri()
scores_df = pd.DataFrame(data=scores, index=obs,
                columns=('C'+str(j+1) for j in range(scores.shape[1])))
g.intensity_map(matrix=scores_df,
                title='The matrix of scores (stadardized principal components)')

# extract the quality of observations in the axis of the principal components
qual_obs = efaModel.getCalObs()
qual_obs_df = pd.DataFrame(data=qual_obs, index=obs,
                columns=('C'+str(j+1) for j in range(qual_obs.shape[1])))
g.intensity_map(matrix=qual_obs_df, color='Greens',
                title='The matrix of quality of observations in the axis of the principal components')
g.show()


# TODO