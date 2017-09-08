from __future__ import print_function, division
import math
import numpy as np
from sklearn.cluster import MiniBatchKMeans


class hard_EM_GMM(object):
    """ A class for performing hard-EM clustering into a Gaussian
        mixture model.

        Hard-EM clustering is like 'normal' EM clustering, but each
        object can only 'belong' to a single cluster, i.e. instead of
        membership probabilities being stored for each object, we
        instead just record the cluster with the maximum membersip
        probability.

        Attributes
        ----------
        Ndata : int
            The number of data points

        Parameters
        ----------
        X : ndarray(Ndata, Ndim)
            The observed data
        Nclusters : int
            The number of clusters/components
    """

    def __init__(self, X, Nclusters):

        self.X = X
        fallback_sigma = np.cov(self.X, rowvar=False)

        self.Ndata = X.shape[0]
        self.Ndim = X.shape[1]

        self.Nclusters = Nclusters

        self.clusters = []
        for n in range(Nclusters):
            self.clusters.append(EMCGMM_cluster(self.Ndim, self.Ndata,
                                                fallback_sigma))

        self.assignments = np.zeros(self.Ndata, dtype=np.int)

    def random_seed(self):
        """ random_seed()

            Start the clusters by 'seeding' them with an individual
            datum each. Then filter all other data onto their
            nearest cluster.

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        seed_inds = np.random.choice(np.arange(self.Ndata), self.Nclusters,
                                     replace=False)

        for i, index in enumerate(seed_inds):
            self.assignments[index] = i

            # Set data for each cluster and then params
            self.clusters[i].add_datum(self.X[index])
            self.clusters[i].set_params()

        # Assign all data to clusters
        self.assign_data()

    def kmeans_init(self):
        """ kmeans_init()

            Use a k-means clustering to provide the initial
            assignments for the EM.

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=self.Nclusters,
                              batch_size=50)
        mbk.fit(self.X)

        self.assignments = mbk.labels_.copy()
        for i in range(self.Ndata):
            self.clusters[self.assignments[i]].add_datum(self.X[i])

    def assign_data(self):
        """ assign_data()

            assign the data points onto the best fitting
            cluster for each (the E-step)

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        for cluster in self.clusters:
            cluster.clear_data()

        max_logprob = np.zeros(self.Ndata) - np.inf
        max_j = np.zeros(self.Ndata, dtype=np.int) - 1

        for j in range(self.Nclusters):
            logprob = self.clusters[j].logprob(self.X)
            mask = logprob > max_logprob
            max_logprob[mask] = logprob[mask]
            max_j[mask] = j

        self.assignments = max_j

        for i in range(self.Ndata):
            self.clusters[max_j[i]].add_datum(self.X[i])

    def set_params(self):
        """ set_params()

            Set the parameters of each cluster to their maaximum
            likelihood values (the M-step)

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        for i in range(self.Nclusters):
            self.clusters[i].set_params()

    def fit(self, Nsteps):
        """ fit(Nsteps)

            Fit the GMM to the data using Nsteps iteations of
            hard EM

            Parameters
            ----------
            Nsteps : int
                The number of steps of EM to perform

            Returns
            -------
            None
        """
        for i in range(Nsteps):
            self.set_params()
            self.assign_data()

    @classmethod
    def init_fit(cls, X, Nclusters, Nsteps):
        """ init_fit(X, Nclusters, Nsteps)

            Factory method to init and then perform hard-EM
            fitting on data

            Parameters
            ----------
            X : ndarray(Ndata, Ndim)
                The observed data
            Nclusters : int
                The number of clusters/components
            Nsteps : int
                The number of steps of EM to perform

            Returns
            -------
            EM_obj : hard_EMGMM
                A hard_EMGMM object on which Nsteps iterations of
                hard-EM have been performed
        """
        EM_obj = cls(X, Nclusters)
        EM_obj.kmeans_init()
        EM_obj.fit(Nsteps)

        return EM_obj


class EMCGMM_cluster(object):
    """ A class that describes an individual cluster in a GMM
        scheme that is found/refined by EM

        Attributes
        ----------
        mu : ndarray(Ndim)
            The mean vector of the cluster
        sigma : ndarray(Ndim, Ndim)
            The covariance of each cluster
        weight : float
            The weight of the cluster

        Parameters
        ----------
        Ndim : int
            The number of dimens of the space in which the cluster is
            defined
        Ndata : int
            The total number of data points to be clustered
        fallback_sigma : int
            a sigma to fall back on if the number of points
            in the cluster is 1
    """

    def __init__(self, Ndim, Ndata, fallback_sigma):
        self.Ndim = Ndim
        self.Ndata = Ndata
        self.fallback_sigma = fallback_sigma

        self.mu = np.zeros(self.Ndim)
        self.sigma = np.zeros((self.Ndim, self.Ndim))
        self.weight = 0.

        self.data = []

    def clear_data(self):
        """ clear_data()

            Clear the contents of data and data_uncerts

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        self.data = []

    def add_datum(self, datum):
        """ add_datum(datum, datum_uncert)

            Add a datum to the cluster
        """
        self.data.append(datum)

    def set_params(self):
        """ set_params(X, X_uncert)

            Set the parameters of the cluster given the
            noisy data assigned to it

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        if len(self.data) > 0:
            self.mu = np.mean(self.data, axis=0)
        else:
            self.mu = np.zeros(self.Ndim)
        if len(self.data) > 1:
            self.sigma = np.cov(self.data, rowvar=False)
        else:
            self.sigma = self.fallback_sigma
        self.weight = len(self.data)/self.Ndata

    def logprob(self, x):
        """ prob(x)

            Find the (log)probability of a datum x given it is
            a member of this cluster

            Parameters
            ----------
            x : ndarray
                The position of the datum

            Returns
            -------
            logprob : float
                The log probability of the datum assuming it is a
                member of this cluster
        """
        q = np.linalg.solve(self.sigma, (x-self.mu).T).T
        if self.weight > 0:
            log_prob = (math.log(self.weight)
                        - np.linalg.slogdet(self.sigma)[1]/2
                        - np.sum((x-self.mu) * q, axis=1)/2)
        else:
            log_prob = -np.inf
        return log_prob
