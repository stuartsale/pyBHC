from __future__ import print_function, division
import numpy as np


from numpy.linalg import slogdet
import math
from scipy import stats
from scipy.special import gammaln, multigammaln


LOG2PI = math.log(2*math.pi)
LOG2 = math.log(2)
LOGPI = math.log(math.pi)


class CollapsibleDistribution(object):
    """ Abstract base class for a family of conjugate distributions.
    """

    def log_marginal_likelihood(self, X):
        """ Log of the marginal likelihood, P(X|prior).  """
        pass


class NormalInverseWishart(CollapsibleDistribution):
    """
    Multivariate Normal likelihood with multivariate Normal prior on
    mean and Inverse-Wishart prior on the covariance matrix.
    All math taken from Kevin Murphy's 2007 technical report,
    'Conjugate Bayesian analysis of the Gaussian distribution'.
    """

    def __init__(self, **prior_hyperparameters):
        self.nu_0 = prior_hyperparameters['nu_0']
        self.mu_0 = prior_hyperparameters['mu_0']
        self.kappa_0 = prior_hyperparameters['kappa_0']
        self.lambda_0 = prior_hyperparameters['lambda_0']

        self.d = float(len(self.mu_0))

        self.log_z = self.calc_log_z(self.mu_0, self.lambda_0, self.kappa_0,
                                     self.nu_0)

    @staticmethod
    def update_parameters(X, _mu, _lambda, _kappa, _nu, _d):
        n = X.shape[0]
        xbar = np.mean(X, axis=0)
        kappa_n = _kappa + n
        nu_n = _nu + n
        mu_n = (_kappa*_mu + n*xbar)/kappa_n

        S = np.zeros(_lambda.shape) if n == 1 else (n-1)*np.cov(X.T)
        dt = (xbar-_mu)[np.newaxis]

        back = np.dot(dt.T, dt)
        lambda_n = _lambda + S + (_kappa*n/kappa_n)*back

        assert(mu_n.shape[0] == _mu.shape[0])
        assert(lambda_n.shape[0] == _lambda.shape[0])
        assert(lambda_n.shape[1] == _lambda.shape[1])

        return mu_n, lambda_n, kappa_n, nu_n

    @staticmethod
    def calc_log_z(_mu, _lambda, _kappa, _nu):
        d = len(_mu)
        sign, detr = slogdet(_lambda)
        log_z = (LOG2*(_nu*d/2.0)
                 + (d/2.0)*math.log(2*math.pi/_kappa)
                 + multigammaln(_nu/2, d) - (_nu/2.0)*detr)

        return log_z

    def log_marginal_likelihood(self, X):
        n = X.shape[0]
        params_n = self.update_parameters(X, self.mu_0, self.lambda_0,
                                          self.kappa_0, self.nu_0, self.d)
        log_z_n = self.calc_log_z(*params_n)

        return log_z_n - self.log_z - LOG2PI*(n*self.d/2)

    def log_posterior_predictive(self, X_new, X_old):
        """ log_posterior_predictive(X_new, X_old)

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
            X_new : ndarray
                The point for which we want the posterior predicitve.
        """
        params_old = self.update_parameters(X_old, self.mu_0,
                                            self.lambda_0,
                                            self.kappa_0, self.nu_0,
                                            self.d)
        t_sigma = ((params_old[2]+1) / (params_old[2]*(params_old[3]-self.d+1))
                   * params_old[1])
        t_sigma_inv = np.linalg.inv(t_sigma)
        t_dof = params_old[3]-self.d+1

        t_z = X_new - params_old[0]
        t_logdiff = math.log(1+np.sum(t_z*np.dot(t_sigma_inv, t_z))
                             / t_dof)

        sgn, det = np.linalg.slogdet(t_sigma)

        prob = (gammaln((t_dof+self.d)/2)
                - gammaln(t_dof/2)
                - self.d/2*math.log(t_dof)
                - self.d/2*LOGPI
                - det/2
                - (t_dof+self.d)/2*t_logdiff)

        return prob

    def conditional_sample(self, X, size=1):
        """ conditional_sample(X)

            Sample from the posterior predictive distribution
            conditioned on some data X.

            The posterior predicitve distribution follows a
            multivariate t distribution, as per
            en.wikipedia.org/wiki/Conjugate_prior .

            The multivariate is sampled by performing
            x = mu + Z sqrt(nu/u) ,
            where
            Z ~ N(0, sigma) ,
            u ~ chi2(nu) ,
            this implies
            x ~ t_nu(mu, sigma)

            Parameters
            ----------
            X : ndarray
                The existing data on which the posterior predicitve
                is to be conditioned.
            size : int, optional
                The number of samples to be drawn.
        """
        output = np.zeros((size, self.d))

        params_n = self.update_parameters(X, self.mu_0, self.lambda_0,
                                          self.kappa_0, self.nu_0,
                                          self.d)

        t_dof = params_n[3] - self.d + 1
        t_cov = (params_n[2]+1) / (params_n[2]*t_dof) * params_n[1]

        mvn_rv = stats.multivariate_normal(cov=t_cov)
        chi2_rv = stats.chi2(t_dof)

        for it in range(size):
            # Sample u from chi2 dist
            u = chi2_rv.rvs()

            # Sample from multivariate Normal
            z = mvn_rv.rvs()

            output[it, :] = params_n[0] + z*math.sqrt(t_dof/u)

        return output


class NormalFixedCovar(CollapsibleDistribution):
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
    def update_parameters(X, _mu, _sigma, S, _d):
        n = X.shape[0]
        xbar = np.mean(X, axis=0)

        _sigma_inv = np.linalg.inv(_sigma)
        S_inv = np.linalg.inv(S)

        # update variance on mean
        sigma_n_inv = _sigma_inv + n*S_inv
        sigma_n = np.linalg.inv(sigma_n_inv)

        # update mean
        mu_n = (np.dot(_sigma_inv, _mu)
                + n*np.dot(S_inv, xbar))
        mu_n = np.dot(sigma_n, mu_n)

        assert(mu_n.shape[0] == _mu.shape[0])
        assert(sigma_n.shape[0] == _sigma.shape[0])
        assert(sigma_n.shape[1] == _sigma.shape[1])

        return mu_n, sigma_n, S

    @staticmethod
    def calc_log_z(_mu, _sigma, S):
        d = len(_mu)
        sign, detr = slogdet(_sigma)
        _sigma_inv = np.linalg.inv(_sigma)

        log_z = detr/2 + np.sum(_mu*np.dot(_sigma_inv, _mu))

        return log_z

    def log_marginal_likelihood(self, X):
        n = X.shape[0]
        params_n = self.update_parameters(X, self.mu_0, self.sigma_0,
                                          self.S, self.d)
        log_z_n = self.calc_log_z(*params_n)

        Q = 0.
        for i in range(n):
            Q += np.sum(X[i, :]*np.dot(self.S_inv, X[i, :]))

        lml = log_z_n - self.log_z0 - LOG2PI*(n*self.d/2) - Q - self.S_det*n/2

        return lml

    def log_posterior_predictive(self, X_new, X_old):
        """ log_posterior_predictive(X_new, X_old)

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
            X_new : ndarray
                The point for which we want the posterior predicitve.
        """
        params_old = self.update_parameters(X_old, self.mu_0,
                                            self.sigma_0, self.S,
                                            self.d)

        z_sigma = params_old[1]+self.S
        z_sigma_inv = np.linalg.inv(z_sigma)
        diff = X_new-params_old[0]

        z = np.sum(diff*np.dot(z_sigma_inv, diff))

        sgn, det = np.linalg.slogdet(z_sigma)

        prob = (- self.d/2*LOG2PI - det/2 - z/2)

        return prob

    def conditional_sample(self, X, size=1):
        """ conditional_sample(X)

            Sample from the posterior predictive distribution
            conditioned on some data X.

            For the Normal distribution the samples
            are found by sampling froma (multivariate) Normal.

            Parameters
            ----------
            X : ndarray
                The existing data on which the posterior predicitve
                is to be conditioned.
            size : int, optional
                The number of samples to be drawn.
        """

        output = np.zeros((size, self.d))

        params_n = self.update_parameters(X, self.mu_0, self.sigma_0, self.S,
                                          self.d)

        for it in range(size):
            # get covariance
            cov = params_n[1]+self.S

            # Sample from multivariate Normal
            output[it, :] = stats.multivariate_normal.rvs(
                                mean=params_n[0], cov=cov)

        return output
