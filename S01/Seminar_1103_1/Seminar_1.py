print('Hello from Python!')

# This is a comment (CTRL+/) or '''
'''
A comment extended
on multiple lines
'''

# Python tutorials:
# https://www.w3schools.com/python/
# https://www.tutorialspoint.com/python/index.htm

# a = 'test'
# print(a, type(a))
#since lines of codes work by line, 1 label (a) can have multiple values which is not desired
#naming convention: string value = string_something
a = 5
b = 3
print(a,type(a))
print('Rest of 5/3:', a % b)
#new and Python specific numerical operators
print('Integral part of 5/3:', a // b)
print('b to the power of a', b ** a) #POW (^)

#Data types
#strings

str_1 = 'test'
print(str_1, type(str_1), id(str_1))
str_1 += 'cool' #concatenating two strings with += operator. All operators are overloaded,
# and will run instructions based on the data types
print(str_1, type(str_1), id(str_1))
#this will preserve both values, strings are immutable

str_2 = 'mama said: "do not be late"'
print(str_2)
str_3 = '''mama said: "don't be late"''' #we can use triplet of apostrophes at beginning and end of a string,
# in case the message contains 1 or more apostrophes
print(str_3)

#lists - use []
list_1 = [1, 3.14, 'mama', [1, 2, 3]]
print(list_1, type(list_1))

#list slicing
list_2 = [1,2,3,4,5,6,7,8,9]
print(list_2)
print(list_2[:])
print(list_2[::]) #[begin index : end index : pace]
#access the last element of a list
print(list_2[-1])
print(list_2[0:-1:1]) #this to not be used too often, the intervals are open, we must not mess with it if not needed, use the simple 1st version

#print elements of the list with odd indexes using slicing
print(list_2[1::2])
#reverse the order of the elements of a list
print(list_2[-1::-1])
#Python lists are circular
print(list_2[::-1]) #the first index is not needed as the list will go 1 position down from 1st index to last, and continue in that order

#list comprehension
list_3 = [x for x in list_2]
print(list_3)