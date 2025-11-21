import numpy as np


def standardize(x):
    #assume that we receive numpy.ndarray
    means = np.mean(a=x, axis =0) #the means are computed on columns
    stds = np.std(a=x, axis=0) #variables on the columns
    return (x - means) / stds
