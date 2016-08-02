from __future__ import print_function, division
import math
import numpy as np
from scipy import stats


class GMM(object):
    """
        A class to hold a Gaussian mixture model

        Attributes
        ----------
        weights : list
            The probability weights for all the components.
            Should normally sum 
        means : list(ndarray)
            The N dimesional means for the K components. Will have
            shape (K,N)
        covars : list(ndarray)
            The covariance matrices for the K components. Will have
            shape (K,N,N)
    """

    def __init__(self):
        """ __init__()
            
            Initialise a GMM object.
        """

        self.weights = []
        self.means = []
        self.covars = []

        self.N = None
        self.K = 0

    def add_component(self, weight, mean, covar):
        """ add_component(weight, mean, covar)

            Add a component to a GMM instance.

            Parameters
            ----------
            weight : float
                The probability weight of the component
            mean : ndarray
                The N dimensional mean, shape is (N,)
            covar : ndarray
                The covariance matrix, shape is (N,N)
        """

        if self.N is None:
            self.N = mean.shape[0]

        if isinstance(weight, float):
            self.weights.append(weight)
        else:
            raise TypeError("Weight should be a float")

        if mean.ndim==1 and mean.shape[0]==self.N:
            self.means.append(mean)
        else:
            raise ValueError("The mean has the wrong dimension")

        if (covar.ndim==2 and covar.shape[0]==self.N 
            and covar.shape[1]==self.N):
            self.covars.append(covar)
        else:
            raise ValueError("The covariance matrix has the wrong "
                             "shape.")

        self.K += 1

    def normalise_weights(self):
        """ normalise_weights()

            Normalise the weights so that they sum to 1.
        """
        weights_sum = np.sum(self.weights)
        self.weights /= weights_sum

    def set_mean_covar(self):
        """ set_mean_covar()

            Set the mean and covariance across the GMM
        """

        self.gmm_mean = np.zeros(self.N)
        gmm_2ndmoment = np.zeros((self.N, self.N))
        self.gmm_covar = np.zeros((self.N, self.N))

        if np.sum(self.weights)!=1:
            self.normalise_weights()

        for it_K in range(self.K):
            self.gmm_mean += self.weights[it_K]*self.means[it_K]
            gmm_2ndmoment += self.weights[it_K]*(self.covars[it_K]
                                          +np.outer(self.means[it_K],
                                                    self.means[it_K]))

        self.gmm_covar = gmm_2ndmoment - np.outer(self.gmm_mean,
                                                  self.gmm_mean)
            
            
