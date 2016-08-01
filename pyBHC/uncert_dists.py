from __future__ import print_function, division
import numpy as np


import math
from scipy import stats
from scipy.special import gammaln, multigammaln

from dists import CollapsibleDistribution


LOG2PI = math.log(2*math.pi)
LOG2 = math.log(2)
LOGPI = math.log(math.pi)

class uncert_NormalFixedCovar(CollapsibleDistribution):
    """
    Multivariate Normal likelihood with multivariate Normal prior on 
    mean and a fixed covariance matrix.
    All math taken from Kevin Murphy's 2007 technical report, 
    'Conjugate Bayesian analysis of the Gaussian distribution'.
    """

    def __init__(self, **prior_hyperparameters):
        self.mu_0 = prior_hyperparameters['mu_0']
        self.sigma_0 = prior_hyperparameters['sigma_0']
        self.S = prior_hyperparameters['S']

        self.d = float(len(self.mu_0))

        sgn, self.sigma_0_det = np.linalg.slogdet(self.sigma_0)
        sgn, self.S_det = np.linalg.slogdet(self.S)

        self.sigma_0_inv = np.linalg.inv(self.sigma_0)
        self.S_inv = np.linalg.inv(self.S)

        self.log_z0 = self.calc_log_z(self.mu_0, self.sigma_0, self.S)

    @staticmethod
    def update_parameters(X, X_uncert, _mu, _sigma, S, _d):
        if X.shape[0]!=X_uncert.shape[0]:
            raise ValueError("The shapes of X and X_uncert do not "
                             "agree. {0} {1}".format(X.shape,
                                                     X_uncert.shape))
        n = X.shape[0]

        sigma_sum = np.linalg.inv(_sigma)
        mu_sum = np.dot(np.linalg.inv(_sigma), _mu)

        for it in range(n):
            inv_uncert = np.linalg.inv(X_uncert[it,:,:]+S)
            sigma_sum += inv_uncert
            mu_sum += np.dot(inv_uncert, X[it,:])

        sigma_n = np.linalg.inv(sigma_sum)
        mu_n = np.dot(sigma_n, mu_sum)

        assert(mu_n.shape[0] == _mu.shape[0])
        assert(sigma_n.shape[0] == _sigma.shape[0])
        assert(sigma_n.shape[1] == _sigma.shape[1])

        return mu_n, sigma_n, S

    @staticmethod
    def update_remove(X, X_uncert, _mu, _sigma, S, _d):
        if X.shape[0]!=X_uncert.shape[0]:
            raise ValueError("The shapes of X and X_uncert do not "
                             "agree. {0} {1}".format(X.shape,
                                                     X_uncert.shape))
        # Ensure correct dimensionality when X is only one datum
        if X.ndim==1:
            X = X[np.newaxis,:]
        if X_uncert.ndim==2:
            if X_uncert.shape[0]!=X_uncert.shape[1]:
                raise IndexError("Covariance array is not square")
            X_uncert = X_uncert[np.newaxis,:,:]

        n = X.shape[0]

        sigma_sum = np.linalg.inv(_sigma)
        mu_sum = np.dot(np.linalg.inv(_sigma), _mu)


        for it in range(n):
            inv_uncert = np.linalg.inv(X_uncert[it,:,:]+S)
            sigma_sum -= inv_uncert
            mu_sum -= np.dot(inv_uncert, X[it,:])

        sigma_n = np.linalg.inv(sigma_sum)
        mu_n = np.dot(sigma_n, mu_sum)

        assert(mu_n.shape[0] == _mu.shape[0])
        assert(sigma_n.shape[0] == _sigma.shape[0])
        assert(sigma_n.shape[1] == _sigma.shape[1])

        return mu_n, sigma_n, S       


    @staticmethod
    def calc_log_z(_mu, _sigma, S):
        d = len(_mu)
        sign, detr = np.linalg.slogdet(_sigma)
        _sigma_inv = np.linalg.inv(_sigma)

        log_z = detr/2 + np.sum(_mu*np.dot(_sigma_inv, _mu))

        return log_z

    def log_marginal_likelihood(self, X, X_uncert):
        n = X.shape[0]
        params_n = self.update_parameters(X, X_uncert, self.mu_0,
                                          self.sigma_0, self.S,
                                          self.d)
        log_z_n = self.calc_log_z(*params_n)

        Q = 0.
        log_det_prod = 0.
        for it in range(n):
            uncert_inv = np.linalg.inv(X_uncert[it,:,:]+self.S)
            sgn, minus_log_det = np.linalg.slogdet(uncert_inv)

            log_det_prod -= minus_log_det
            Q += np.sum(X[it,:]*np.dot(uncert_inv, X[it,:]))

        lml = (log_z_n - self.log_z0 - LOG2PI*(n*self.d/2) - Q
                - log_det_prod/2)

        return lml


    def log_posterior_predictive(self, X_new, X_uncert_new,
                                 X_old, X_uncert_old):
        """ log_posterior_predictive(X_new, X_uncert_new,
                                     X_old, X_uncert_old)

            Find the posterior predictive probabilitiy p(X_new|X_old)
            where X_old is some data we already have and X_new is the
            point at which we want the posterior predictive prob.

            The posterior predictive distribution is a (multivariate) 
            t-distribution.
            The formula required is given by 
            en.wikipedia.org/wiki/Conjugate_prior
            
            Parameters
            ----------
            X_old : ndarray
                The existing data on which the posterior predicitve
                is to be conditioned.
            X_uncert_old : ndarray
                The uncertainty on the measurements of the existing 
                data.
            X_new : ndarray
                The point for which we want the posterior predicitve.
            X_new : ndarray
                The uncertainty on the point for which we want the
                posterior predicitve.
        """
        params_old = self.update_parameters(X_old, X_uncert_old,
                                            self.mu_0, self.sigma_0,
                                            self.S, self.d)

        z_sigma = params_old[1]+self.S+X_uncert_new
        z_sigma_inv = np.linalg.inv(z_sigma)
        diff = X_new-params_old[0]

        z = np.sum(diff*np.dot(z_sigma_inv, diff))

        sgn, det = np.linalg.slogdet(z_sigma)

        prob = (- self.d/2*LOG2PI - det/2 - z/2)
    
        return prob


    def single_posterior(self, datum, datum_uncert, cluster_params):
        """ single_posterior(datum, datum_uncert, cluster_params)

            Find the marginal posterior for the parameters of a 
            single datum in a cluster.

            Parameters
            ----------
            datum : ndarray
                The measurement of the data point of interest
            datum_uncerts : ndarray
                The uncertianty on the measurement - assumed to be 
                Gaussian and given as a covariance matrix.
            cluster_params : dict
                The posterior parameters of the cluster, in the 
                form given by update_params().

            Returns
            -------
            mu_post : ndarray(d)
                The mean of the posterior
            sigma_post : ndarray(d,d)
                The covariance matrix of the posterior
        """
        # First update params, removing datum
        cavity_params = self.update_remove(datum, datum_uncert, 
                                           cluster_params[0], 
                                           cluster_params[1],
                                           cluster_params[2],
                                           self.d)

        # now calculate posterior params

        prior_like_inv = np.linalg.inv(cavity_params[1]
                                        +cavity_params[2])
        sigma_post_inv = (prior_like_inv+np.linalg.inv(datum_uncert))
        sigma_post = np.linalg.inv(sigma_post_inv)

        mu_sum = (np.dot(prior_like_inv, cavity_params[0])
                  +np.dot(np.linalg.inv(datum_uncert), datum))
        mu_post = np.dot(sigma_post, mu_sum)

        return mu_post, sigma_post

