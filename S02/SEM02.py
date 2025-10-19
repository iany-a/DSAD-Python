import Functions as fun
import matplotlib.pyplot as plt

#how parameters are passed to functions in Python
a = 7
b = 11
print(a, id(a))
print(b, id(b))

fun.swap(a,b)
print(a, id(a))
print(b, id(b))
#the swapping does not work as in C++ way, we need to parse a list and change its members values

#The parameters to functions in Python are passed by object reference
vector = [7,11]
print(vector, type(vector))
fun.swap_2(vector)
print(vector, type(vector))

#list comprehension = feature that allows us to generate list content in programmatic manner
list_1 = [1, 'mama', 3.14, [1,2]]
print(list_1)
list_2 = [x for x in list_1]
print(list_2)

#create a list containing the first 100 natural numbers starting with 1

list_3 = [x for x in range(1,101)] #(begin value, end value, pace)
#default value for begin is 0
#the default pace is 1
#the described interval of integers is open on the right hand side
print (list_3)
list_4 = [x+1 for x in range(100)] #again x+1 used because we start the range with 0, but the assignment specifies to start with 1. starting value and pace not needed in this case.
print (list_4)

#generate the squares of the first 100 natural numbers, starting with 1
list_5 = [(x+1)**2 for x in range(100)] # ** = POW
print(list_5)

#plt.plot(list_5) # (x,y) but we usually skip the x axis
#plt.show()

#comprehension of 2 variables
list_6 = [2,4]
print(list_6)
list_7 = [(x+y)*3 for x in list_6
          for y in list_4]
print (list_7)

list_8 = [x for x in range(-10,10)]
print (list_8)
#comprehension with conditions
#select from list_8 odd values greater than zero
list_9 = [x for x in list_8 if x>0 and x%2==1]
print(list_9)
#print(list_8[list_8 > 0]) think about it

#Python dictionaries {}
#associative collections of pairs (key, value)
dict_1 = {'Monday':'Mama','Tuesday':[1,2,3],"Wednesday": 3.14}
print(dict_1, type(dict_1))
#direct access based on the key
dict_1['Tuesday'] = 'a nicer string' #update existing dictionary entry
print(dict_1, type(dict_1))
dict_1['Thursday']= 'even pops is here' #insertion of new pairs into the dictionary
print(dict_1, type(dict_1))
print(dict_1.keys(), type(dict_1.keys()))
#any sequence of objects can be assumed to be a list, and can be converted into a list
print(list(dict_1.keys()), type(dict_1.keys()))
print(dict_1.values(), type(dict_1.values()))
#access to a list pairs (k, v)
print(dict_1.items(), type(dict_1.items()))
#extract pairs of (k,v) values from the dictionary
for (k, v) in dict_1.items():
    print(k, ':', v)

#dictionary comprehension
dict_2 = {x: (x+1) for x in range(100)}
print(dict_2)
#generate the cubes of integers in [-10, 10]
dict_3 = {x+5: (x+1)**3 for x in range(-10, 11)}
#plt.plot(dict_3.keys(), dict_3.values())
#plt.show()
print(dict_3)

#dictionary comprehension with 2 variables
dict_4 = {(x+y): (x+1)**y for x in range (-10, 10) for y in (1,2,3)}
#dict_4 = {(x+y): (x+1)**y for x in range (-10, 10) if (x+1)!=0 for y in (-3, -2, 1,2,3)}
print(dict_4)
print(len(dict_4))
#plt.plot(dict_4.keys(), dict_4.values())
#plt.show()

#do dictionary comprehension having the key and values from 2 distinct data collections
#dictionaries of 2 or more variables

