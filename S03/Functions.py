import numpy as np
#create a functions that translates n values from [0,1) to a generic one [a, b)
#[0,1)------->f(n)--------->[a,b)
def random_AB(a, b, n):
    return np.round(a+np.random.rand(n) * (b-a))
