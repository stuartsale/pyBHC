from __future__ import print_function, division
import itertools as it
import numpy as np

from numpy import logaddexp
import math


class bhc(object):
    """
    An instance of Bayesian hierarchical clustering CRP mixture model.
    Attributes
    ----------
    assignments : list(list(int))
        A list of lists, where each list records the clustering at 
        each step by giving the index of the leftmost member of the
        cluster a leaf is traced to.
    root_node : Node
        The root node of the clustering tree.
    lml : float
        An estimate of the log marginal likelihood of the model 
        under a DPMM.
    Notes
    -----
    The cost of BHC scales as O(n^2) and so becomes inpractically 
    large for datasets of more than a few hundred points.
    """


    def __init__(self, data, data_model, crp_alpha=1.0):
        """
        Init a bhc instance and perform the clustering.

        Parameters
        ----------
        data : numpy.ndarray (n, d)
            Array of data where each row is a data point and each 
            column is a dimension.
        data_model : CollapsibleDistribution
            Provides the approprite ``log_marginal_likelihood`` 
            function for the data.
        crp_alpha : float (0, Inf)
            CRP concentration parameter.
        Returns
        -------
        assignments : list(list(int))
            list of assignment vectors. assignments[i] is the 
            assignment of data to i+1 clusters.
        lml : float
            log marginal likelihood estimate.
        """
        # initialize the tree
        nodes = dict((i, Node(np.array([x]), data_model, crp_alpha,
                              indexes=i))
                     for i, x in enumerate(data))
        n_nodes = len(nodes)
        assignment = [i for i in range(n_nodes)]
        self.assignments = [list(assignment)]
        rks = []

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

                if tmp_node.log_rk > max_rk:
                    max_rk = tmp_node.log_rk
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

        print(rks)

        self.root_node = nodes[0]
        self.assignments = np.array(self.assignments)

        # The denominator of log_rk is at the final merge is an 
        # estimate of the marginal likelihood of the data under DPMM
#        self.lml = denom

    def left_run(self):
        node = self.root_node
        while node.left_child is not None:
            print(node.indexes, np.mean(node.data, axis=0), node.data.shape)
            node = node.left_child
        print(node.indexes, np.mean(node.data, axis=0), node.data.shape)

    def right_run(self):
        node = self.root_node
        while node.right_child is not None:
            print(node.indexes, np.mean(node.data, axis=0), node.data.shape)
            node = node.right_child
        print(node.indexes, np.mean(node.data, axis=0), node.data.shape)

    def find_path(self, index):
        """ find_path(index)

            Finds the sequence of left and right merges needed to
            run from the root node to a particular leaf.

            Parameters
            ----------
            index : int
                The index of the leaf for which we want the path 
                from the root node.
        """
        merge_path = []
        last_leftmost_index = self.assignments[-1][index]
        last_right_incluster = (self.assignments[-1]
                                ==last_leftmost_index)

        for it in range(len(self.assignments)-2, -1, -1):
            new_leftmost_index = self.assignments[it][index]

            if new_leftmost_index!=last_leftmost_index:
                # True if leaf is on the right hand side of a merge
                merge_path.append("right")
                last_leftmost_index = new_leftmost_index
                last_right_incluster = (self.assignments[it]
                                        ==new_leftmost_index)
        
            else:       # Not in a right hand side of a merge

                new_right_incluster = (self.assignments[it]
                                        ==last_leftmost_index)

                if (new_right_incluster!=last_right_incluster).any():
                    # True if leaf is on the left hand side of a merge
                    merge_path.append("left")
                    last_right_incluster = new_right_incluster

        return merge_path


class Node(object):
    """ A node in the hierarchical clustering.
    Attributes
    ----------
    nk : int
        Number of data points assigned to the node
    data : numpy.ndarrary (n, d)
        The data assigned to the Node. Each row is a datum.
    crp_alpha : float
        Chinese restaurant process concentration parameter
    log_dk : float
        Used in the calculation of the prior probability. Defined in
        Fig 3 of Heller & Ghahramani (2005).
    log_pi : float
        Prior probability that all associated leaves belong to one
        cluster.
    log_rk : float
        The log-probability of the merge that created the node. For
        nodes that are leaves (i.e. not created by a merge) this is
        None.   
    left_child : Node
        The left child of a merge. For nodes that are leaves (i.e.
        the original data points and not made by a merge) this is 
        None.
    right_child : Node
        The right child of a merge. For nodes that are leaves 
        (i.e. the original data points and not made by a merge) 
        this is None.
    index : int
        The indexes of the leaves associated with the node in some 
        indexing scheme.
    """

    def __init__(self, data, data_model, crp_alpha=1.0, log_dk=None,
                 log_pi=0.0, log_rk=None, left_child=None,
                 right_child=None, indexes=None):
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
        left_child : Node, optional
            The left child of a merge. For nodes that are leaves (i.e.
            the original data points and not made by a merge) this is 
            None.
        right_child : Node, optional
            The right child of a merge. For nodes that are leaves 
            (i.e. the original data points and not made by a merge) 
            this is None.
        index : int, optional
            The index of the node in some indexing scheme.
        """
        self.data_model = data_model
        self.data = data
        self.nk = data.shape[0]
        self.crp_alpha = crp_alpha
        self.log_pi = log_pi
        self.log_rk = log_rk

        self.left_child = left_child
        self.right_child = right_child

        if isinstance(indexes, int):
            self.indexes = [indexes]
        else:
            self.indexes = indexes

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
        indexes = node_left.indexes + node_right.indexes
        indexes.sort()

        nk = data.shape[0]
        log_dk = logaddexp(math.log(crp_alpha) + math.lgamma(nk),
                           node_left.log_dk + node_right.log_dk)
        log_pi = -math.log1p(math.exp(node_left.log_dk 
                                      + node_right.log_dk
                                      - math.log(crp_alpha) 
                                      - math.lgamma(nk) ))

        # Calculate log_rk - the log probability of the merge

        logp_comb = data_model.log_marginal_likelihood(data)
        numer = log_pi + logp_comb

        neg_pi = math.log(-math.expm1(log_pi))
        denom = logaddexp(numer, neg_pi+node_left.logp+node_right.logp)

        log_rk = numer-denom

        if log_pi == 0:
            raise RuntimeError('Precision error')

        return cls(data, data_model, crp_alpha, log_dk, log_pi, 
                   log_rk, node_left, node_right, indexes)
