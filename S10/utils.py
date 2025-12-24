import numpy as np


def replaceNAN(X):
    # assume that X is numpy.ndarray
    means = np.nanmean(X, axis=0) # we have variables on the columns
    pos = np.where(np.isnan(X))
    print(pos, type(pos))
    X[pos] = means[pos[1]]
    # TODO
    return X
