import pandas as pd
from pandas import Series
import numpy as np

def randomAB(a=None, b=None, n=None):
    return a+np.random.rand(n) * (b-a)

list_1 = randomAB(-5, 5, 7)
print(type(list_1))
serie_3 = pd.Series(list_1, index=('L'+str(i+1) for i in range(7)))
print(serie_3)

serie_4 = pd.Series(data = [-8, 'sir', True, 1.1],
                    index = ('a', 'b', 'c', 'd'))
print(serie_4)
serie_4['d']=-5.1
print(serie_4[['a', 'c', 'd']])

serie_5 = pd.Series(data=[-8, 7.0, 12, 1.1],
index=['a', 'b', 'c', 'd'])
print(serie_5[serie_5 > 0.0]) #get only the positive values
print(serie_5*2) #return the values in a multiplication operation
print(np.exp(serie_5)) # return a whole function on all series values: e**(serie_5)

vector = randomAB(1, 3, 15) #creates a vector of 15 random real values between 1 and 3
nda_1 = np.ndarray(shape=(5, 3), buffer=vector, dtype=float)
#keep in mind to use n >= shape values (rows, columns) multiplied, otherwise it will return an error
#dtype needs to be defined as float
df_1 = pd.DataFrame(data=nda_1) #provide the ndarray as input for DataFrame, without labels (auto-generated)
print(df_1)

# DataFrame from dict of lists or tuples
dict_2 = {'Alina':[10.00, 9.50, 8.90, 9.40], #list
          'Anul':(2018, 2019, 2020, 2021), #tupil
          'Luna':('ianuarie', 'februarie', #tupil
                  'martie', 'aprilie')}
df_2 = pd.DataFrame(data=dict_2, #feed the dictionary into the DataFrame
                    index=(1, 2, 3, 4)) #labels provided for rows
print(df_2)

# DataFrame from dict of Series, indexes are merge
s_1 = pd.Series(data=(1, 2, 3),
                index=['Lin1'+str(i+1) for i in range(3)])
s_2 = pd.Series(data=(4, 5, 6),
                index=['Lin2'+str(i+1) for i in range(3)])
dict_3 = {'Col1':s_1, 'Col2':s_2}
df_3 = pd.DataFrame(data=dict_3)
print(df_3)


d_1 = {'Maria':1, 'Ioana':2, 'Marin':3, 'Cornel':4}
d_2 = {'Maricica':1, 'Ion':2, 'Marina':3, 'Cornel':4}
d_3 = {'An 1':d_1, 'An 2':d_2}
df_4 = pd.DataFrame(data=d_3)
print(df_4)

d_1 = {'Maria':1, 'Ioana':2, 'Marin':3, 'Cornel':4}
d_2 = {'Maricica':1, 'Ion':2, 'Marina':3, 'Cornel':4}
list_1 = [d_1, d_2]
df_5 = pd.DataFrame(data=list_1)
print(df_5)

s_1 = pd.Series(data=(1, 2, 3), index=['Lin1'+str(i+1) for i in
                                       range(3)])
s_2 = pd.Series(data=(4, 5, 6), index=['Lin2'+str(i+1) for i in
                                       range(3)])
list_1 = [s_1, s_2]
df_6 = pd.DataFrame(data=list_1)
print(df_6)

# list of list or tuple (equivalent of 2D ndarray)
list_2 = [(1, 2, 3),
          (4, 5, 6)]
df_7 = pd.DataFrame(data=list_2)
print(df_7)

#access row and column names

note = pd.DataFrame({"Data Structures":[5, 4, 6],'Data Analysis':[10, 6, 7], 'Algebra':[7, 8, 7]},
                    index=["Ionescu Dan", "Popescu Diana", 'Georgescu Radu'])
print(note)
print(note.index, note.columns, sep='\n')
print(note.index[0], note.columns[1])

print(note.loc["Popescu Diana","Algebra"])
print(note.loc["Popescu Diana"]["Algebra"])
print(note.loc["Popescu Diana",:"Algebra"]) #this will not include "Algebra" as the interval is open on the end
print(note.loc["Popescu Diana"][:"Algebra"]) #rewritten previous syntax
print(note.loc["Popescu Diana":,:"Algebra"])
print(note.loc["Popescu Diana":][:"Algebra"])
print(note.iloc[1, 2])
print(note.iloc[1][2])
print(note.iloc[[1,2],2]) #print(note.iloc[[1,2]][2]) # wrong
print(note.iloc[1:, 2]) #print(note.iloc[1:][2]) # wrong
print(note.iloc[[1,0],[1,2]])
print(note.iloc[1][[1,2]])
print(note.iloc[1,1:])
print(note.iloc[1][1:])
print(note.iloc[1:,1:])