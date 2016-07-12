from __future__ import print_function, division
import itertools as it
import numpy as np

from numpy import logaddexp
import math


class bhc(object):

    def __init__(self, data, data_model, crp_alpha=1.0):
        """
        Bayesian hierarchical clustering CRP mixture model.
        Notes
        -----
        The cost of BHC scales as O(n^2) and so becomes inpractically 
        large for datasets of more than a few hundred points.
        ----------
        data : numpy.ndarray (n, d)
            Array of data where each row is a data point and each column 
            is a dimension.
        data_model : CollapsibleDistribution
            Provides the approprite ``log_marginal_likelihood`` function 
            for the data.
        crp_alpha : float (0, Inf)
            CRP concentration parameter.
        Returns
        -------
        assignments : list(list(int))
            list of assignment vectors. assignments[i] is the assignment 
            of data to i+1 clusters.
        lml : float
            log marginal likelihood estimate.
        """
        # initialize the tree
        nodes = dict((i, Node(np.array([x]), data_model, crp_alpha))
                     for i, x in enumerate(data))
        n_nodes = len(nodes)
        assignment = [i for i in range(n_nodes)]
        self.assignments = [list(assignment)]
        rks = [0]

        while n_nodes > 1:
            print(n_nodes)
            max_rk = float('-Inf')
            merged_node = None

            # for each pair of clusters (nodes), compute the merger 
            # score.
            for left_idx, right_idx in it.combinations(nodes.keys(),
                                                       2):
                tmp_node = Node.as_merge(nodes[left_idx],
                                         nodes[right_idx])

                logp_left = nodes[left_idx].logp
                logp_right = nodes[right_idx].logp
                logp_comb = tmp_node.logp

                log_pi = tmp_node.log_pi

                numer = log_pi + logp_comb

                neg_pi = math.log(-math.expm1(log_pi))
                denom = logaddexp(numer, neg_pi+logp_left+logp_right)

                log_rk = numer-denom

                if log_rk > max_rk:
                    max_rk = log_rk
                    merged_node = tmp_node
                    merged_right = right_idx
                    merged_left = left_idx

            rks.append(math.exp(max_rk))

            # Merge the highest-scoring pair
            del nodes[merged_right]
            nodes[merged_left] = merged_node

            for i, k in enumerate(assignment):
                if k == merged_right:
                    assignment[i] = merged_left
            self.assignments.append(list(assignment))

            n_nodes -= 1

        # The denominator of log_rk is at the final merge is an 
        # estimate of the marginal likelihood of the data under DPMM
        self.lml = denom


class Node(object):
    """ A node in the hierarchical clustering.
    Attributes
    ----------
    nk : int
        Number of data points assigned to the node
    data : numpy.ndarrary (n, d)
        The data assigned to the Node. Each row is a datum.
    crp_alpha : float
        CRP concentration parameter
    log_dk : float
        Some kind of number for computing probabilities
    log_pi : float
        For to compute merge probability
    """

    def __init__(self, data, data_model, crp_alpha=1.0, log_dk=None,
                 log_pi=0.0):
        """
        Parameters
        ----------
        data : numpy.ndarray
            Array of data_model-appropriate data
        data_model : idsteach.CollapsibleDistribution
            For to calculate marginal likelihoods
        crp_alpha : float (0, Inf)
            CRP concentration parameter
        log_dk : float
            Cached probability variable. Do not define if the node is 
            a leaf.
        log_pi : float
            Cached probability variable. Do not define if the node is 
            a leaf.
        """
        self.data_model = data_model
        self.data = data
        self.nk = data.shape[0]
        self.crp_alpha = crp_alpha
        self.log_pi = log_pi

        if log_dk is None:
            self.log_dk = math.log(crp_alpha)
        else:
            self.log_dk = log_dk

        self.logp = self.data_model.log_marginal_likelihood(self.data)

    @classmethod
    def as_merge(cls, node_left, node_right):
        """ Create a node from two other nodes
        Parameters
        ----------
        node_left : Node
            the Node on the left
        node_right : Node
            The Node on the right
        """
        crp_alpha = node_left.crp_alpha
        data_model = node_left.data_model
        data = np.vstack((node_left.data, node_right.data))

        nk = data.shape[0]
        log_dk = logaddexp(math.log(crp_alpha) + math.lgamma(nk),
                           node_left.log_dk + node_right.log_dk)
        log_pi = -math.log1p(math.exp(node_left.log_dk 
                                      + node_right.log_dk
                                      - math.log(crp_alpha) 
                                      - math.lgamma(nk) ))

        if log_pi == 0:
            raise RuntimeError('Precision error')

        return cls(data, data_model, crp_alpha, log_dk, log_pi)
