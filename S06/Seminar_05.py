import numpy as np
import Graphics as g
import pandas as pd
matrix = np.random.uniform(low=1,high=10, size=(30,15))

corr = np.corrcoef(x=matrix, rowvar=False)
print(corr, type(corr), corr.shape)

g.correlogram(R2=corr)

corr_df= pd.DataFrame(data=corr, index=('V'+str(i+1) for i in range(corr.shape[0])),
                      columns=('V'+str(j+1) for j in range (corr.shape[1])))
g.correlogram(R2=corr_df,dec=1,title='Correlogram from a pandas dataframe')
g.show()