'''
A class for implementing PCA (Principal Components Analysis)
'''

import Functions as f

import numpy as np



class PCA:
    def __init__(self, x):
        self.x_sts = f.standardize(x)
        #variance-covariance matrix of x standardized computation
        self.cov=np.cov(self.x_sts, rowvar=False)
        #variabless are on columns
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov)
        print(eigenvectors.shape)
        print(eigenvalues.shape)
        #sort the eigenvalues and the corresponding eigenvectors in descending order
        k_desc = np.argsort(eigenvalues)[::-1]
        #k_desc = [k for k in np.argsort(a=eigenvalues)]
        print(k_desc)
        self.alpha = eigenvalues[k_desc]
        self.a = eigenvectors[:, k_desc]

        #compute the principal components
        self.C = self.x_sts @ self.a


    def getAlpha(self):
        return self.alpha

    def getComponents(self):
        return self.C

    def getScores(self):
        return self.C / np.sqrt(self.alpha)


