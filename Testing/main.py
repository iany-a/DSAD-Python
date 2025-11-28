import numpy as np
import pandas as pd
from pyparsing import NotAny

matrix = np.ndarray((7,7), float)
matrix[:] = 7
matrix[1:6, 1:6] = 0
np.fill_diagonal(matrix, 33.)
matrix[3,3]=77.
print(matrix)

b = np.ndarray((7,7), None)
b[:] = 'nan'
print(b)

b[0:1]=5.
b[0:7, 0:1]=5.
b[0:7, 6:7]=5.
b[6:7]=5.
b[1:6, 1:6]=0
print(b)
subset = b[1:6, 1:6]
print(subset)

vector = np.round(np.random.uniform(0, 10, 100), 2)
print(vector)
labels = [f"L+{x}" for x in range (0,100)]
series = pd.Series(data=vector, index=labels)
print(series)

uneven = np.ndarray((11,5), float)
uneven[:] = np.round(np.random.uniform(0,10,(11,5)),2)
print (uneven)
row_labels = [f"L{x}" for x in range(0, 11)]
#print(row_labels)
col_labels = [f"C{x}" for x in range(0, 5)]
#print(col_labels)
dataframe = pd.DataFrame(data=uneven, index=row_labels, columns=col_labels)
print(dataframe)

keys = [f"Stud{i}" for i in range(1,8)]
labels_series = [f'Ex{i}' for i in range(1,6)]
dict = {x: pd.Series(np.random.randint(0,11,5), index=labels_series)
        for x in keys}
dataframe2 = pd.DataFrame(dict)
print(dataframe2)

