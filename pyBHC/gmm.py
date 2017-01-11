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

    def __init__(self, weights=None, means=None, covars=None):
        """ __init__(weights=None, means=None, covars=None)

            Initialise a GMM object. If no weights etc are supplied
            then an 'empty' mixture is created.

            Parameters
            ----------
        weights : list, optional
            The probability weights for all the components.
            Should normally sum
        means : list(ndarray), optional
            The N dimesional means for the K components. Will have
            shape (K,N)
        covars : list(ndarray), optional
            The covariance matrices for the K components. Will have
            shape (K,N,N)
        """

        if weights is None or means is None or covars is None:
            self.weights = []
            self.means = []
            self.covars = []

            self.N = None
            self.K = 0

        else:
            self.weights = weights
            self.means = means
            self.covars = covars

            self.N = self.means[0].shape[0]
            self.K = len(self.means)

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

        if mean.ndim == 1 and mean.shape[0] == self.N:
            self.means.append(mean)
        else:
            raise ValueError("The mean has the wrong dimension")

        if (covar.ndim == 2 and covar.shape[0] == self.N
                and covar.shape[1] == self.N):
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

    def get_covar_Ls(self):
        """ get_covar_Ls()

            Find and store the Cholesky decompositions of the
            covariance matrices.
        """

        self.covar_Ls = []

        for it in range(self.K):
            covar_L = np.linalg.cholesky(self.covars[it])
            self.covar_Ls.append(covar_L)

    def get_covar_invs(self):
        """ get_covar_invs()

            Find and store the inverses of the
            covariance matrices.
        """

        self.covar_invs = []

        for it in range(self.K):
            covar_inv = np.linalg.inv(self.covars[it])
            self.covar_invs.append(covar_inv)

    def set_mean_covar(self):
        """ set_mean_covar()

            Set the mean and covariance across the GMM
        """

        self.gmm_mean = np.zeros(self.N)
        gmm_2ndmoment = np.zeros((self.N, self.N))
        self.gmm_covar = np.zeros((self.N, self.N))

        if np.sum(self.weights) != 1:
            self.normalise_weights()

        for it_K in range(self.K):
            self.gmm_mean += self.weights[it_K]*self.means[it_K]
            gmm_2ndmoment += self.weights[it_K]*(self.covars[it_K]
                                                 + np.outer(self.means[it_K],
                                                            self.means[it_K]))

        self.gmm_covar = gmm_2ndmoment - np.outer(self.gmm_mean,
                                                  self.gmm_mean)

    @classmethod
    def as_merge(cls, gmm_1, gmm_2):
        """ as_merge(gmm_1, gmm_2)

            Create a new GMM by merging two existing GMMs
        """

        weights = gmm_1.weights + gmm_2.weights
        means = gmm_1.means + gmm_2.means
        covars = gmm_1.covars + gmm_2.covars

        merged_gmm = cls(weights, means, covars)
        merged_gmm.normalise_weights()

        return merged_gmm

    def sample(self, n_samples=1000):
        """ sample(n_samples=1000)

            Sample from the GMM

            Parameters
            ----------
            n_samples : int, optional
                The number of samples to take

            Returns
            -------
            samples : ndarray
                An array of samples, shape (n_samples, N)
        """
        samples = np.zeros((n_samples, self.N))

        # Verify weights sum to 1
        self.normalise_weights()

        # get cholesky decompositions of covariances
        self.get_covar_Ls()

        # get sample of component numbers
        comps = np.random.choice(self.K, size=n_samples,
                                 p=self.weights)

        for it in range(n_samples):

            # sample within a component
            samples[it] = (self.means[comps[it]] +
                           np.dot(self.covar_Ls[comps[it]],
                                  np.random.randn(self.N)))

        return samples
