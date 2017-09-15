from __future__ import print_function, division
import numpy as np


import math
from scipy import stats
from scipy.special import gammaln, multigammaln

from dists import CollapsibleDistribution, FrozenDistribution


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
        if X.shape[0] != X_uncert.shape[0]:
            raise ValueError("The shapes of X and X_uncert do not "
                             "agree. {0} {1}".format(X.shape,
                                                     X_uncert.shape))
        n = X.shape[0]

        sigma_sum = np.linalg.inv(_sigma)
        mu_sum = np.dot(np.linalg.inv(_sigma), _mu)

        for it in range(n):
            inv_uncert = np.linalg.inv(X_uncert[it, :, :]+S)
            sigma_sum += inv_uncert
            mu_sum += np.dot(inv_uncert, X[it, :])

        sigma_n = np.linalg.inv(sigma_sum)
        mu_n = np.dot(sigma_n, mu_sum)

        assert(mu_n.shape[0] == _mu.shape[0])
        assert(sigma_n.shape[0] == _sigma.shape[0])
        assert(sigma_n.shape[1] == _sigma.shape[1])

        return mu_n, sigma_n, S

    @staticmethod
    def update_remove(X, X_uncert, _mu, _sigma, S, _d):
        if X.shape[0] != X_uncert.shape[0]:
            raise ValueError("The shapes of X and X_uncert do not "
                             "agree. {0} {1}".format(X.shape,
                                                     X_uncert.shape))

        # Ensure correct dimensionality when X is only one datum
        if X.ndim == 1:
            X = X[np.newaxis, :]
        if X_uncert.ndim == 2:
            if X_uncert.shape[0] != X_uncert.shape[1]:
                raise IndexError("Covariance array is not square")
            X_uncert = X_uncert[np.newaxis, :, :]

        n = X.shape[0]

        sigma_sum = np.linalg.inv(_sigma)
        mu_sum = np.dot(np.linalg.inv(_sigma), _mu)

        for it in range(n):
            inv_uncert = np.linalg.inv(X_uncert[it, :, :]+S)
            sigma_sum -= inv_uncert
            mu_sum -= np.dot(inv_uncert, X[it, :])

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

        log_z = detr/2 + np.sum(_mu*np.dot(_sigma_inv, _mu))/2

        return log_z

    def log_marginal_likelihood(self, X, X_uncert, verbose=False):
        n = X.shape[0]
        params_n = self.update_parameters(X, X_uncert, self.mu_0,
                                          self.sigma_0, self.S,
                                          self.d)
        log_z_n = self.calc_log_z(*params_n)

        Q = 0.
        log_det_prod = 0.
        for it in range(n):
            uncert_inv = np.linalg.inv(X_uncert[it, :, :]+self.S)
            sgn, minus_log_det = np.linalg.slogdet(uncert_inv)

            log_det_prod -= minus_log_det
            Q += np.sum(X[it, :]*np.dot(uncert_inv, X[it, :]))

        lml = (log_z_n - self.log_z0 - LOG2PI*(n*self.d/2) - Q/2
               - log_det_prod/2)

        if verbose:
            print(lml, log_z_n, -Q, -log_det_prod/2, log_z_n-self.log_z0,
                  params_n[1])

        return lml

    def log_posterior_predictive(self, X_new, X_uncert_new,
                                 X_old, X_uncert_old):
        """ log_posterior_predictive(X_new, X_uncert_new,
                                     X_old, X_uncert_old)

            Find the posterior predictive probabilitiy p(X_new|X_old)
            where X_old is some data we already have and X_new is the
            point at which we want the posterior predictive prob.

            The posterior predictive distribution is a (multivariate)
            Normal distribution.

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
        # first get 'cavity prior'

        cavity_mu, cavity_sigma = self.cavity_prior(datum,
                                                    datum_uncert,
                                                    cluster_params)

        cavity_sigma_inv = np.linalg.inv(cavity_sigma)
        uncert_inv = np.linalg.inv(datum_uncert)

        sigma_post_inv = (cavity_sigma_inv + uncert_inv)
        sigma_post = np.linalg.inv(sigma_post_inv)

        mu_sum = (np.dot(cavity_sigma_inv, cavity_mu)
                  + np.dot(uncert_inv, datum))
        mu_post = np.dot(sigma_post, mu_sum)

        return mu_post, sigma_post

    def cavity_prior(self, datum, datum_uncert, cluster_params):
        """ cavity_prior(datum, datum_uncert, cluster_params)

            Find the 'cavity prior' for the parameters of a
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
            mu_cavity : ndarray(d)
                The mean of the cavity prior
            sigma_cavity : ndarray(d,d)
                The covariance matrix of the cavity prior
        """
        # First update params, removing datum
        cavity_params = self.update_remove(datum, datum_uncert,
                                           cluster_params[0],
                                           cluster_params[1],
                                           cluster_params[2],
                                           self.d)

        # now calculate cacity prior params
        sigma_cavity = cavity_params[1]+cavity_params[2]
        mu_cavity = cavity_params[0]

        return mu_cavity, sigma_cavity

    def freeze_posterior_predictive(self, X_old, X_uncert_old):
        """ freeze_posterior_predictive(X_old, X_uncert_old)

            Dump a frozen version of the posterior predictive
            distribution p(X_new | X_old)

            Parameters
            ----------
            X_old : ndarray
                The existing data on which the posterior predicitve
                is to be conditioned.
            X_uncert_old : ndarray
                The uncertainty on the measurements of the existing
                data.

            Returns
            -------
            frozen_dist : FrozenNFCPosteriorPred
                The frozen distribution
        """
        params_old = self.update_parameters(X_old, X_uncert_old,
                                            self.mu_0, self.sigma_0,
                                            self.S, self.d)

        frozen_dist = FrozenNFCPosteriorPred(params_old[0],
                                             params_old[1]+self.S)
        return frozen_dist


class FrozenNFCPosteriorPred(FrozenDistribution):
    """ The log posterior predicitve distribution implied by a
        uncert_NormalFixedCovar instance with its parameters
        (i.e. mean, covariance) frozen.

        Parameters
        ----------
        mu : ndarray
            The mean of the distribution
        sigma : ndarray
            The (marginal) covariance of the distribution
    """

    def __init__(self, mu, sigma):
        self.__mu = mu
        self.__sigma = sigma
        self.__d = float(len(self.__mu))

        sgn, self.__det = np.linalg.slogdet(sigma)
        if sgn <= 0:
            raise AttributeError("Covariance matrix is not PSD")

    def __call__(self, X, X_uncert):
        """ __call__(X, X_uncert)

            Returns the log-probability of a noisy datum (assumed
            Gaussian) given this distribution.

            Parameters
            ----------
            X : ndarray
                The measurement
            X_uncert : ndarray
                The uncertainties on the measurment, expressed as a
                covariance array

            Returns
            -------
            log_prob : float
                The log-probability of the datum given this
                distribution
        """
        z_sigma = self.__sigma + X_uncert
        z_sigma_inv = np.linalg.inv(z_sigma)

        diff = X - self.__mu

        z = np.dot(diff, np.dot(z_sigma_inv, diff))

        log_prob = - self.__d*LOG2PI/2. - self.__det - z
        return log_prob

    def __str__(self):
        out_string = ("FrozenNFCPosteriorPred with \n"
                      "mean : [")
        for i in range(self.__mu.shape[0]):
            out_string += "{0}".format(self.__mu[i])
            if i+1 < self.__mu.shape[0]:
                out_string += ", "
        out_string += "]"

        return out_string
