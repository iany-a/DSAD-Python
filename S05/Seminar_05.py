import numpy as np
import Graphics as g
import pandas as pd
matrix = np.random.uniform(low=1,high=10, size=(30,15))

corr = np.corrcoef(x=matrix, rowvar=False)
#print(corr, type(corr), corr.shape)

#g.correlogram(R2=corr)

corr_df= pd.DataFrame(data=corr, index=('V'+str(i+1) for i in range(corr.shape[0])),
                      columns=('V'+str(j+1) for j in range (corr.shape[1])))
#g.correlogram(R2=corr_df,dec=1,title='Correlogram from a pandas dataframe')
#g.show()


#call in the correlation circle
g.correlation_circle(R2=corr)

#g.correlation_circle('what')
#call the correlation circle passing a pandas.DataFrame
g.correlation_circle(R2=corr_df, title ='Correlation circle from a pandas.DataFrame')
#g.show()

#create a vector of 12 random values in [0.3 , 3)
#values = np.random.uniform(low=0.3, high=3, size=(12,3))
values = np.random.uniform(low=0.3, high=3, size=12)
print (np.round(values, 2), type(values))
values = np.sort(a=values)
#values in descending order
print(np.round(values[::-1], 2))

#call the function for the explained variance by the principal components (PCA)
#values in descending order
g.explained_variance(eigenvalues = values[::-1])
g.show()