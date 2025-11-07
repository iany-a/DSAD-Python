#dictionary comprehension discussion final words
#comprehension of 2 or more variables
import numpy as np
import Functions as f
import pandas as pd

l_1 = [x+1 for x in range(10)]
print(l_1, type(l_1))
l_2 = (1,2) #tuple
print(l_2, type(l_2))
dict_1 = {(x+y): (x+y)*2 for x in l_1 for y in l_2}
print(dict_1, type(dict_1))
print(len(dict_1)) #11 pairs of key: value
#no cartesian product resulted
#duplicate keys are being overwritten e.g. 1*2 and 2*1, hence the length is 11

dict_2 = {(x,y): (x+y)*2 for x in l_1 for y in l_2} #(x,y) = tuple instead of addition
print(dict_2, type(dict_2))
print(len(dict_2))
#in this manner we can obtain a cartesian product

#the zip() function usage
# collection of [1,2,3,4,5,6] and [1,2,3], the zip is returning as output a list of tuple and tuple
#1st element from first collection with 1st element from 2nd collection
#((1,1), (2,2), (3,3))
#length will be equal to the lowest length from the 2 tuples

gen_1 = zip(l_1, l_2)
print(gen_1, type(gen_1))
tlp_1 = tuple(zip(l_1, l_2)) #actually shows the tuple resulted from zip
print(tlp_1, type(tlp_1))

dict_3 = {(x+y): (x+y)*2 for (x,y) in zip(l_1, l_2)}
print(dict_3)
print(len(dict_3))
dict_4 = {(x+y+z): (x+y+z)*2 for (x,y,z) in zip(l_1, l_2, (3,4,5))}
print(dict_4)
print(len(dict_4))

#lambda expressions in Python
mean = lambda a,b: (a + b)/2
print('The mean of 7 and 11: ', mean(7,11))#lambda will show the variables a: and b: as defined

mean_vector = lambda vector: np.mean(a=vector)
print('The mean of the vector: ',
      mean_vector([1,2,3,4,5,6,7]))

#generators in Python = special function which does not return any value but yields the next in programmatic sequence
#create a generator to produce the natural numbers from 1 to infinity

def generator():
    t = 0
    while True: #goes to infinite (no limits)
    #for i in range(10): #limits to 10 value max
        t+=1
        yield t #reference to be used in other functions

call = generator() # assign the generator's reference
#print the first 10 natural numbers from 1 to 10
for i in range(10):
    try:
        print('Next natural number: ', next(call))
    except:
        break

#there is no tuple comprehension in Python
gen_2 = ((x+1) for x in range(10))
print(gen_2, type(gen_2))

for i in range(3):
    try:
        print('Next natural number: ', next(gen_2))
    except:
        break



list_1 = list(gen_2)
print(list_1, type(list_1))
print('_________________________________________________________________________________________________')
print('_________________________________________________________________________________________________')
#we can still call explicitly the next() function
# while True:
#     try:
#         print('Continue with Next natural number: ', next(gen_2))
#     except:
#         break

#create a dictionary with keys in the following format:
#keys: S_1, S_2, ... S_5, as strings
#values: lists of 6 randomly generated values in [1,10]

#first we generate random values

nda_1 = np.random.rand(6)
print(nda_1, type(nda_1))

dict_5 = {'S_'+str(i+1): [x for x in f.random_AB(1,10, 6)]
          for i in range(5)}
print(dict_5)

for (k,v) in dict_5.items():
    print(k, ':', v)

#create a pandas.DataFrame from a dictionary
df_1 = pd.DataFrame(data = dict_5)
print(df_1, type(df_1))

#create a pandas DataFrame from a numpy array (np.ndarray)
#create a 2D ndarray of (4, 6) with randomly generated integers in [1,10]
nda_2 = np.random.randint(low=1, high=10, size=(4,6)) #size can be an int or a tuple, tuple shows the number of rows and columns
print(nda_2, type(nda_2))

df_2 = pd.DataFrame(data=nda_2)
print(df_2, type(df_2))


#Homework: provide labels to rows and columns for the numpy ndarray
#C_1, C_2, .... C_6
#L_1, .... L_4

