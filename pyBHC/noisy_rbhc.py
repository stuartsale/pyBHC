from __future__ import print_function, division
import math
import numpy as np

import gmm
from noisy_bhc import noisy_bhc


class noisy_rbhc(object):
    """
    An instance of Randomized Bayesian hierarchical clustering CRP
    mixture model, where there is noise on the observations.
    Attributes
    ----------

    Notes
    -----
    The cost of rBHC scales as O(nlogn) and so should be preferred
    for large data sets.
    """

    def __init__(self, data, data_uncerts, data_model, crp_alpha=1.0,
                 sub_size=50, verbose=False):

        """
        Init a rbhc instance and perform the clustering.

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
        sub_size : int
            The size of the random subset of pooints used to form the
            tree whose top split is employed to filter the data.
            Denoted m in the Heller & Ghahramani (2005b).
        verbose : bool
            If true various bits of information, possibly with
            diagnostic uses, will be printed.
        """

        self.data = data
        self.data_uncerts = data_uncerts
        self.data_model = data_model
        self.crp_alpha = crp_alpha
        self.sub_size = sub_size

        self.verbose = verbose

        self.nodes = {}

        # initialize the tree

        self.assignments = []

        root_node = noisy_rbhc_Node(data, data_uncerts, data_model,
                                    crp_alpha)
        self.nodes[0] = {0: root_node}

#        self.tree = rbhc_Node.recursive_split(root_node, 50)
        self.recursive_split(root_node)

        self.find_assignments()
        self.refine_probs()

        self.params_set = False

    def recursive_split(self, parent_node):

        rBHC_split, children = noisy_rbhc_Node.as_split(parent_node,
                                                        self.sub_size)

        if self.verbose:
            print("Parent node [{0}][{1}] ".format(
                       parent_node.node_level,
                       parent_node.level_index), end="")

        if rBHC_split:      # continue recussing down
            if children[0].node_level not in self.nodes:
                self.nodes[children[0].node_level] = {}

            self.nodes[children[0].node_level][children[0].level_index] = (
                                                                children[0])
            self.nodes[children[1].node_level][children[1].level_index] = (
                                                                children[1])

            if self.verbose:
                print("split to children:\n"
                      "\tnode [{0}][{1}], size : {2}\n"
                      "\tnode [{3}][{4}], size : {5}\n".format(
                       children[0].node_level,
                       children[0].level_index, children[0].nk,
                       children[1].node_level,
                       children[1].level_index, children[1].nk))

            self.recursive_split(children[0])
            self.recursive_split(children[1])

        else:               # terminate
            if parent_node.tree_terminated and self.verbose:
                print("terminated with bhc tree")
            elif parent_node.truncation_terminated and self.verbose:
                print("truncated")

    def find_assignments(self):
        """ find_assignements()

            Find which Node each data point is assigned to on each
            level.
            This fills self.assignemnts - which is a list, with an
            ndarray for each level. The array for each level gives
            the level index of the nde it is associated with.
            If a data point is not assigned to a node on a given
            level it is given the value -1.
        """

        self.assignments.append(np.zeros(self.data.shape[0], int))

        for level_key in self.nodes:
            if level_key != 0:
                self.assignments.append(
                            np.zeros(self.data.shape[0], int)-1)

                for index_key in self.nodes[level_key]:
                    if index_key % 2 == 0:
                        parent_index = int(index_key/2)
                        write_indexes = (self.assignments[level_key-1]
                                         == parent_index)

                        self.assignments[level_key][write_indexes] = (
                              parent_index*2+1
                              - self.nodes[level_key-1][parent_index].
                              left_allocate.astype(int))

    def refine_probs(self):
        """ refine_probs()

            Improve the estimated probabilities used by working with
            the full set of data allocated to each node, rather than
            just the initial sub-set used to create/split nodes.
        """
        # travel up from leaves improving log_rk etc.

        for level_it in range(len(self.assignments)-1, -1, -1):

            for node_it in self.nodes[level_it]:
                node = self.nodes[level_it][node_it]

                if node.tree_terminated:
                    if node.nk > 1:
                        # log_rk, etc are accurate
                        node.log_dk = node.true_bhc.root_node.log_dk
                        node.log_pi = node.true_bhc.root_node.log_pi
                        node.logp = node.true_bhc.root_node.logp
                        node.log_ml = node.true_bhc.root_node.log_ml
                        node.log_rk = node.true_bhc.root_node.log_rk
                    else:
                        node.log_dk = self.crp_alpha
                        node.log_pi = 0.
                        node.logp = self.data_model.log_marginal_likelihood(
                                            node.data, node.data_uncerts)[0]
                        node.log_ml = node.logp
                        node.log_rk = 0.

                elif node.truncation_terminated:
                    node.log_dk = (math.log(self.crp_alpha)
                                   + math.lgamma(node.nk))
                    node.log_pi = 0.
                    node.logp = self.data_model.log_marginal_likelihood(
                                            node.data, node.data_uncerts)[0]
                    node.log_ml = node.logp
                    node.log_rk = 0.

                else:
                    left_child = self.nodes[level_it+1][node_it*2]
                    right_child = self.nodes[level_it+1][node_it*2+1]

                    node.log_dk = np.logaddexp(
                           math.log(self.crp_alpha)
                           + math.lgamma(node.nk),
                           left_child.log_dk + right_child.log_dk)

                    exponent = (left_child.log_dk + right_child.log_dk
                                - math.log(self.crp_alpha)
                                - math.lgamma(node.nk))
                    if exponent < 5:
                        node.log_pi = -math.log1p(math.exp(exponent))
                    else:
                        node.log_pi = -exponent

                    if node.log_pi == 0:
                        q = (left_child.log_dk + right_child.log_dk
                             - math.log(self.crp_alpha)
                             - math.lgamma(node.nk))
                        neg_pi = q
                    else:
                        neg_pi = math.log(-math.expm1(node.log_pi))

                    node.logp = self.data_model.log_marginal_likelihood(
                                                        node.data,
                                                        node.data_uncerts)[0]

                    node.log_ml = np.logaddexp(node.log_pi+node.logp,
                                               neg_pi + left_child.log_ml
                                               + right_child.log_ml)
                    node.log_rk = node.log_pi + node.logp - node.log_ml

        # travel down from top improving

        for level_it in range(1, len(self.assignments)):
            for node_it in self.nodes[level_it]:
                node = self.nodes[level_it][node_it]
                parent_node = self.nodes[level_it-1][int(node_it/2)]

                node.prev_wk = (parent_node.prev_wk
                                * (1-math.exp(parent_node.log_rk)))

    def __str__(self):
        bhc_str = ("==================================\n"
                   "rBHC fit to {0} (noisy) data points, with "
                   "alpha={1} and sub_size={2} .\n".format(
                        self.data.shape[0], self.crp_alpha, self.sub_size))

        for l_it in range(len(self.nodes)):
            bhc_str += "===== LEVEL {0} =====\n".format(l_it)
            for n_it in self.nodes[l_it]:
                node = self.nodes[l_it][n_it]
                bhc_str += ("node : {0} size : {1} "
                            "node_prob : {2:.9G} ({3:G} {4:G})\n".format(
                                   n_it, node.nk,
                                   node.prev_wk*np.exp(node.log_rk),
                                   node.params[0][0], node.params[0][1]))
        return bhc_str

    def get_global_posterior(self):
        """ get_global_posteriors()

            Find the posterior implied by the clustering as a Gaussian
            mixture, with each component in he mixture corresponding
            to a node in the clustering.

        """
        # travese tree setting params

        if not self.params_set:
            self.set_params()

        # initialise a GMM
        self.global_GMM = gmm.GMM()

        # Traverse tree
        for level_it in range(len(self.assignments)):

            for node in self.nodes[level_it].values():

                mu = node.params[0]
                sigma = node.params[1] + node.params[2]

                if node.log_rk is not None:
                    weight = node.prev_wk*math.exp(node.log_rk)
                else:       # leaf
                    weight = node.prev_wk

                if weight > 0:
                    self.global_GMM.add_component(weight, mu, sigma)

                # deal with bhc tree children
                if node.tree_terminated and node.nk > 1:

                    # check if single posteriors need finding
                    if node.true_bhc.global_GMM is None:
                        node.true_bhc.get_global_posterior()

                    self.global_GMM.weights.extend(
                        node.prev_wk * node.true_bhc.global_GMM.weights[1:])
                    self.global_GMM.means.extend(
                                node.true_bhc.global_GMM.means[1:])
                    self.global_GMM.covars.extend(
                                node.true_bhc.global_GMM.covars[1:])
                    self.global_GMM.K += node.true_bhc.global_GMM.K-1

        self.global_GMM.normalise_weights()
        self.global_GMM.set_mean_covar()

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
        # travese tree setting params

        if not self.params_set:
            self.set_params()

        # get mixture model for data point

        # initialise a GMM
        post_GMM = gmm.GMM()

        for level_it in range(len(self.assignments)):
            node_it = self.assignments[level_it][index]

            if node_it >= 0:
                node = self.nodes[level_it][node_it]

                if node.log_rk is not None:
                    weight = node.prev_wk*math.exp(node.log_rk)
                else:       # leaf
                    weight = node.prev_wk

                mu = node.params[0]
                sigma = node.params[1] + node.params[2]

                post_GMM.add_component(weight, mu, sigma)

                # deal with bhc tree children
                if node.tree_terminated and node.nk > 1:

                    # check if single posteriors need finding
                    if node.true_bhc.post_GMMs is None:
                        node.true_bhc.get_single_posteriors()

                    # find index of datum in this tree

                    tree_it = np.nonzero(np.equal(node.true_bhc.data,
                                                  self.data[index])
                                         .all(1))[0][0]
                    tree_GMM = node.true_bhc.post_GMMs[tree_it]

                    tree_prev_wk = (node.prev_wk)

                    post_GMM.weights.extend(tree_prev_wk
                                            * tree_GMM.weights[1:])
                    post_GMM.means.extend(tree_GMM.means[1:])
                    post_GMM.covars.extend(tree_GMM.covars[1:])
                    post_GMM.K += tree_GMM.K-1

        post_GMM.normalise_weights()
        post_GMM.set_mean_covar()

        return post_GMM

    def get_cavity_priors(self):
        """ get_cavity_priors()

            Get the 'cavity priors' the prior implied for each datum
            by removing it from the clusters.
        """
        # travese tree setting params

        if not self.params_set:
            self.set_params()

        self.cavity_GMMs = []

        # get cavity prior mixture models for each data point
        for datum_it in range(self.data.shape[0]):
            # initialise a GMM
            cavity_GMM = gmm.GMM()

            for level_it in range(len(self.assignments)):
                node_it = self.assignments[level_it][datum_it]

                if node_it >= 0:
                    node = self.nodes[level_it][node_it]

                    if node.log_rk is not None:
                        weight = node.prev_wk*math.exp(node.log_rk)
                    else:       # leaf
                        weight = node.prev_wk

                    mu, sigma = node.data_model.cavity_prior(
                                    self.data[datum_it],
                                    self.data_uncerts[datum_it],
                                    node.params)
                    cavity_GMM.add_component(weight, mu, sigma)

                    # deal with bhc tree children
                    if node.tree_terminated and node.nk > 1:

                        # check if single posteriors need finding
                        if node.true_bhc.cavity_GMMs is None:
                            node.true_bhc.get_cavity_priors()

                        # find index of datum in this tree

                        tree_it = np.nonzero(np.equal(node.true_bhc.data,
                                                      self.data[datum_it])
                                             .all(1))[0][0]

                        tree_GMM = node.true_bhc.cavity_GMMs[tree_it]

                        tree_prev_wk = (node.prev_wk)

                        cavity_GMM.weights.extend(tree_prev_wk
                                                  * tree_GMM.weights[1:])
                        cavity_GMM.means.extend(tree_GMM.means[1:])
                        cavity_GMM.covars.extend(tree_GMM.covars[1:])
                        cavity_GMM.K += tree_GMM.K-1

            cavity_GMM.normalise_weights()
            cavity_GMM.set_mean_covar()
            self.cavity_GMMs.append(cavity_GMM)

    def set_params(self):

        for level_it in self.nodes:
            for node in self.nodes[level_it].values():
                node.get_node_params()

        self.params_set = True

    @property
    def root_node(self):
        return self.nodes[0][0]


class noisy_rbhc_Node(object):
    """ A node in the randomised Bayesian hierarchical clustering,
        where there is noise on the observations.
        Attributes
        ----------
        nk : int
            Number of data points assigned to the node
        D : int
            The dimension of the data points
        data : numpy.ndarrary (n, d)
            The data assigned to the Node. Each row is a datum.
        data_uncerts: numpy.ndarray (n, d, d)
            Array of uncertainties on the data, such that the first
            axis is a data point and the second two axes are for the
            covariance matrix.
        data_model : idsteach.CollapsibleDistribution
            The data model used to calcuate marginal likelihoods
        crp_alpha : float
            Chinese restaurant process concentration parameter
        log_rk : float
            The probability of the merged hypothesis for the node.
            Given by eqn 3 of Heller & Ghahrimani (2005).
        prev_wk : float
            The product of the (1-r_k) factors for the nodes leading
            to this node from (and including) the root node. Used in
            eqn 9 of Heller & ghahramani (2005a).
        node_level : int, optional
            The level in the hierarchy at which the node is found.
            The root node lives in level 0 and the level number
            increases down the tree.
        level_index : int, optional
            An index that identifies each node within a level.
        left_allocate : ndarray(bool)
            An array that records if a datum has been allocated
            to the left child (True) or the right(False).
        log_dk : float
            Cached probability variable. Do not define if the node is
            a leaf.
        log_pi : float
            Cached probability variable. Do not define if the node is
            a leaf.
        log_ml : float
            The log marginal likelihood for the tree of the node and
            its children. This is given by eqn 2 of Heller &
            Ghahrimani (2005). Note that this definition is
            recursive.  Do not define if the node is
            a leaf.
        logp : float
            The log marginal likelihood for the particular cluster
            represented by the node. Given by eqn 1 of Heller &
            Ghahramani (2005).
    """
    def __init__(self, data, data_uncerts, data_model, crp_alpha=1.0,
                 prev_wk=1., node_level=0, level_index=0):
        """ __init__(data, data_uncerts, data_model, crp_alpha=1.0
                     prev_wk=1., node_level=0, level_index=)

            Initialise a rBHC node.

            Parameters
            ----------
            data : numpy.ndarrary (n, d)
                The data assigned to the Node. Each row is a datum.
            data_uncerts: numpy.ndarray (n, d, d)
                Array of uncertainties on the data, such that the
                first axis is a data point and the second two axes
                are for the covariance matrix.
            data_model : idsteach.CollapsibleDistribution
                The data model used to calcuate marginal likelihoods
            crp_alpha : float, optional
                Chinese restaurant process concentration parameter
            prev_wk : float
                The product of the (1-r_k) factors for the nodes
                leading to this node from (and including) the root
                node. Used in eqn 9 of Heller & ghahramani (2005a).
            node_level : int, optional
                The level in the hierarchy at which the node is found.
                The root node lives in level 0 and the level number
                increases down the tree.
            level_index : int, optional
                An index that identifies each node within a level.
            verbose : bool
                If true various bits of information, possibly with
                diagnostic uses, will be printed.

        """

        self.data = data
        self.data_uncerts = data_uncerts
        self.data_model = data_model
        self.crp_alpha = crp_alpha
        self.prev_wk = prev_wk
        self.node_level = node_level
        self.level_index = level_index

        self.nk = data.shape[0]
        self.D = data.shape[1]

        self.log_rk = 0

        self.tree_terminated = False
        self.truncation_terminated = False

    def set_rk(self, log_rk):
        """ set_rk(log_rk)

            Set the value of the ln(r_k) The probability of the
            merged hypothesis as given in Eqn 3 of Heller & Ghahramani
            (2005a)

            Parameters
            ----------
            log_rk : float
                The value of log_rk for the node
        """
        self.log_rk = log_rk

    @classmethod
    def as_split(cls, parent_node, sub_size):
        """ as_split(parent_node, subsize)

            Perform a splitting of a rBHC node into two children.
            If the number of data points is large a randomized
            filtered split, as in Fig 4 of Heller & Ghahramani (2005b)
            is performed.
            Otherwise, if the number of points is less than or equal
            to subsize then these are simply subject to a bhc
            clustering.

            Parameters
            ----------
            parent_node : rbhc_Node
                The parent node that is going to be split
            sub_size : int
                The size of the random subset of pooints used to form
                the tree whose top split is employed to filter the
                data.
                Denoted m in Heller & Ghahramani (2005b).

            Returns
            -------
            rBHC_split : bool
                True if the size of data is greater than sub_size and
                so a rBHC split/filtering has occured.
                False if the size of data is less than/equal to
                sub_size and so an bhc clustering that includes all
                the data has been found.
            children : list(rbhc_Node) , bhc
                A clustering of the data, either onto two child
                rbhc_Nodes or as a full bhc tree of all the data
                within parent_node.
            left_allocate : ndarray(bool)
                An array that records if a datum has been allocated
                to the left child (True) or the right(False).
        """

        if (parent_node.prev_wk*parent_node.nk) < 1E-0:
            rBHC_split = False
            parent_node.truncation_terminated = True
            children = []

            # make subsample tree
            if parent_node.nk > sub_size:
                parent_node.subsample_bhc(sub_size)

                # set log_rk from the estimate given by self.sub_bhc
                parent_node.set_rk(parent_node.sub_bhc.root_node.log_rk)
            elif parent_node.nk > 1:
                parent_node.true_bhc = noisy_bhc.from_data(
                                             parent_node.data,
                                             parent_node.data_uncerts,
                                             parent_node.data_model,
                                             parent_node.crp_alpha)
                parent_node.set_rk(parent_node.true_bhc.root_node.log_rk)
                parent_node.tree_terminated = True
            else:
                parent_node.set_rk(0.)
                parent_node.tree_terminated = True

        else:

            if parent_node.nk > sub_size:    # do rBHC filter
                # make subsample tree
                parent_node.subsample_bhc(sub_size)
                print("SS", parent_node.sub_bhc.root_node.left_child.nk,
                      parent_node.sub_bhc.root_node.right_child.nk)

                # set log_rk from the estimate given by self.sub_bhc
                parent_node.set_rk(parent_node.sub_bhc.root_node.log_rk)

                # filter data through top level of subsample_bhc
                parent_node.filter_data()

                # create new nodes

                child_prev_wk = (parent_node.prev_wk
                                 * (1-math.exp(parent_node.log_rk)))
                child_level = parent_node.node_level+1

                left_child = cls(parent_node.left_data,
                                 parent_node.left_data_uncerts,
                                 parent_node.data_model,
                                 parent_node.crp_alpha, child_prev_wk,
                                 child_level,
                                 parent_node.level_index*2)
                right_child = cls(parent_node.right_data,
                                  parent_node.right_data_uncerts,
                                  parent_node.data_model,
                                  parent_node.crp_alpha, child_prev_wk,
                                  child_level,
                                  parent_node.level_index*2+1)
                rBHC_split = True
                children = [left_child, right_child]
                print("TT", left_child.nk, right_child.nk)

            elif parent_node.nk > 1:             # just use the bhc tree
                parent_node.true_bhc = noisy_bhc.from_data(
                                             parent_node.data,
                                             parent_node.data_uncerts,
                                             parent_node.data_model,
                                             parent_node.crp_alpha)
                children = parent_node.true_bhc
                rBHC_split = False
                parent_node.tree_terminated = True

                parent_node.set_rk(children.root_node.log_rk)

            else:                       # only 1 datum
                children = []
                rBHC_split = False
                parent_node.tree_terminated = True

                parent_node.set_rk(0.)

        return (rBHC_split, children)

    def subsample_bhc(self, sub_size):
        """ subsample_bhc(sub_size)

            Produce a subsample of sub_size data points and then
            perform an bhc clustering on it.

            Parameters
            ----------
            sub_size : int
                The size of the random subset of pooints used to form
                the tree whose top split is employed to filter the
                data.
                Denoted m in Heller & Ghahramani (2005b).
        """

        self.sub_indexes = np.random.choice(np.arange(self.nk),
                                            sub_size, replace=False)
        sub_data = self.data[self.sub_indexes]
        sub_data_uncerts = self.data_uncerts[self.sub_indexes]
        self.sub_bhc = noisy_bhc.from_data(sub_data, sub_data_uncerts,
                                           self.data_model, self.crp_alpha,
                                           verbose=False)

    def filter_data(self):
        """ filter_data()

            Filter the data in a rbhc_node onto the two Nodes at the
            second from top layer of a bhc tree.
        """
        # set up data arrays
        self.left_data = np.empty(shape=(0, self.D))
        self.right_data = np.empty(shape=(0, self.D))
        self.left_data_uncerts = np.empty(shape=(0, self.D, self.D))
        self.right_data_uncerts = np.empty(shape=(0, self.D, self.D))

        # get assignemnt for sub_bhc objects
        self.left_allocate = np.zeros(self.nk, dtype=bool)

        # Run through data

        for ind in np.arange(self.nk):

            # check if in subset
            if ind in self.sub_indexes:
                sub_ind = np.argwhere(self.sub_indexes == ind)[0][0]
                if self.sub_bhc.assignments[-2][sub_ind] == 0:
                    self.left_allocate[ind] = True
                    self.left_data = np.vstack((self.left_data,
                                                self.data[ind]))
                    self.left_data_uncerts = np.vstack(
                                   (self.left_data_uncerts,
                                    self.data_uncerts[np.newaxis, ind]))
                else:
                    self.right_data = np.vstack((self.right_data,
                                                 self.data[ind]))
                    self.right_data_uncerts = np.vstack(
                                   (self.right_data_uncerts,
                                    self.data_uncerts[np.newaxis, ind]))

            # non subset data
            else:
                if (self.sub_bhc.root_node.left_child.nk <=
                        self.sub_bhc.root_node.right_child.nk):
                    left_prob = self.sub_bhc.tree_posterior_predictive_prob(
                                    self.sub_bhc.root_node.left_child,
                                    self.data[ind], self.data_uncerts[ind])

                    right_prob = self.sub_bhc.tree_posterior_predictive_prob(
                                    self.sub_bhc.root_node.right_child,
                                    self.data[ind], self.data_uncerts[ind],
                                    target_prob=left_prob)

                else:
                    right_prob = self.sub_bhc.tree_posterior_predictive_prob(
                                    self.sub_bhc.root_node.right_child,
                                    self.data[ind], self.data_uncerts[ind])

                    left_prob = self.sub_bhc.tree_posterior_predictive_prob(
                                    self.sub_bhc.root_node.left_child,
                                    self.data[ind], self.data_uncerts[ind],
                                    target_prob=right_prob)

                if left_prob >= right_prob:
                    # possibly change this to make tupe and vstack at
                    # end if cost is high
                    self.left_allocate[ind] = True
                    self.left_data = np.vstack((self.left_data,
                                                self.data[ind]))
                    self.left_data_uncerts = np.vstack(
                                   (self.left_data_uncerts,
                                    self.data_uncerts[np.newaxis, ind]))
                else:
                    self.right_data = np.vstack((self.right_data,
                                                 self.data[ind]))
                    self.right_data_uncerts = np.vstack(
                                   (self.right_data_uncerts,
                                    self.data_uncerts[np.newaxis, ind]))

    def get_node_params(self):
        self.params = self.data_model.update_parameters(
                                              self.data,
                                              self.data_uncerts,
                                              self.data_model.mu_0,
                                              self.data_model.sigma_0,
                                              self.data_model.S,
                                              self.data_model.d)
