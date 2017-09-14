from __future__ import print_function, division
import math
import numpy as np

import gmm
from hardEM import hard_EM_GMM
from noisy_bhc import noisy_bhc


class noisy_EMBHC(object):
    """
        An instance of EM Bayesian hierarchical clustering CRP
        mixture model, where there is noise on the observations.

        EMBHC works by clustering the data into a small number
        of clusters using hard-EM and then performs BHC on the
        clusters

        Attributes
        ----------
        EM_clusters : hard_EM_GMM
            The clustering of the data into some GMM clustering by
            hard EM
        self.cluster_bhc : noisy_bhc
            The aglomorative BHC clustering of the (preclustered)
            data

        Parameters
        ----------
        data : numpy.ndarray (n, d)
            Array of data where each row is a data point and each
            column is a dimension.
        data_uncerts: numpy.ndarray (n, d, d)
            Array of uncertainties on the data, such that the first
            axis is a data point and the second two axes are for the
            covariance matrix.
        data_model : CollapsibleDistribution
            Provides the approprite ``log_marginal_likelihood``
            function for the data.
        crp_alpha : float (0, Inf)
            CRP concentration parameter.
        Nclusters : int
            The number of EM clusters formed
        verbose : bool
            If true various bits of information, possibly with
            diagnostic uses, will be printed.

        Notes
        -----
        The cost of rBHC scales as O(n) and so should be preferred
        for very large data sets.
    """

    def __init__(self, data, data_uncerts, data_model, crp_alpha=1.0,
                 Nclusters=50, verbose=False):
        self.data = data
        self.data_uncerts = data_uncerts
        self.data_model = data_model
        self.crp_alpha = crp_alpha
        self.Nclusters = Nclusters

        self.verbose = verbose

        # Make clusters in ~Identity uncert space
        mean_datum = np.nanmean(self.data, axis=0)
        mean_uncert = np.nanmean(self.data_uncerts, axis=(0))

        uncert_L_inv = np.linalg.cholesky(np.linalg.inv(mean_uncert))

        clean_mask = ~np.isnan(self.data).any(axis=1)

        clean_data = self.data[clean_mask]
        clean_data_uncerts = self.data_uncerts[clean_mask]
        transformed_data = np.dot(clean_data - mean_datum, uncert_L_inv)

        # form the EM clusters
        self.EM_clusters = hard_EM_GMM.init_fit(transformed_data,
                                                self.Nclusters, 20)

        # segregate the data according to cluster assignments
        clustered_data = []
        clustered_data_uncerts = []

        for i in range(self.EM_clusters.Nclusters):
            mask = self.EM_clusters.assignments == i

            if np.sum(mask) > 0:
                clustered_data.append(clean_data[mask])
                clustered_data_uncerts.append(clean_data_uncerts[mask])

        # Put the clusters through BHC
        self.cluster_bhc = noisy_bhc.from_preclustered(
                                clustered_data, clustered_data_uncerts,
                                self.data_model, crp_alpha=self.crp_alpha,
                                verbose=self.verbose)

        self.assignments = np.zeros(self.data.shape[0], dtype=np.int) - 9
        self.assignments[clean_mask] = self.EM_clusters.assignments

    def __str__(self):
        return str(self.cluster_bhc)

    def get_individual_posterior(self, index):
        """ get_individual_posterior(index)

            Find the posteriors for a data point as a Gaussian
            mixture, with each component in he mixture corresponding
            to a node that the data point appears in.

            Parameters
            ----------
            index : int
                The index of the data point

            Returns
            -------
            post_GMM : gmm.GMM
                A Gaussian mixture model description of the posterior
        """
        # initialise a GMM
        post_GMM = gmm.GMM()

        # Check if posteriors have been found for cluster_bhc
        if self.cluster_bhc.post_GMMs is None:
            self.cluster_bhc.get_single_posteriors()

        # get required posterior dist
        if self.assignments[index] >= 0:
            return self.cluster_bhc.post_GMMs[self.assignments[index]]
        else:
            return None

    @property
    def root_node(self):
        return self.cluster_bhc.root_node
