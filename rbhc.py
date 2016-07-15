from __future__ import print_function, division
import math
import numpy as np

from bhc import bhc


class rbhc(object):
    """
    An instance of Randomized Bayesian hierarchical clustering CRP 
    mixture model.
    Attributes
    ----------

    Notes
    -----
    The cost of rBHC scales as O(nlogn) and so should be preferred
    for large data sets.
    """

    def __init__(self, data, data_model, crp_alpha=1.0, sub_size=50):

        """
        Init a rbhc instance and perform the clustering.

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
        sub_size : int
            The size of the random subset of pooints used to form the
            tree whose top split is employed to filter the data.
            Denoted m in the Heller & Ghahramani (2005b).
        """        
        self.data = data
        self.data_model = data_model
        self.crp_alpha = crp_alpha
        self.sub_size = sub_size

        self.nodes = {}

        # initialize the tree

        self.assignments = []

        root_node = rbhc_Node(data, data_model, crp_alpha)
        self.nodes[0] = {0:root_node}

#        self.tree = rbhc_Node.recursive_split(root_node, 50)
        self.recursive_split(root_node)

        self.find_assignments()



    def recursive_split(self, parent_node):

        rBHC_split, children = rbhc_Node.as_split(parent_node,
                                                  self.sub_size)

        if rBHC_split:      # continue recussing down
            if children[0].node_level not in self.nodes:
                self.nodes[children[0].node_level] = {}

            self.nodes[children[0].node_level]\
                      [children[0].level_index] = children[0]
            self.nodes[children[1].node_level]\
                      [children[1].level_index] = children[1]

            self.recursive_split(children[0])
            self.recursive_split(children[1])

        else:               # terminate
            print("reached the leaves")


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

        self.assignments.append(np.zeros(self.data.shape[0]))

        for level_key in self.nodes:
            if level_key!=0:
                self.assignments.append(
                            np.zeros(self.data.shape[0])-1)

                for index_key in self.nodes[level_key]:
                    if index_key%2==0:
                        parent_index = int(index_key/2)
                        write_indexes = (self.assignments[level_key-1]
                                          ==parent_index)

                        self.assignments[level_key][write_indexes] = (
                              parent_index*2+1-
                              self.nodes[level_key-1][parent_index].\
                              left_allocate.astype(int)  )



class rbhc_Node(object):
    """ A node in the randomised Bayesian hierarchical clustering.
        Attributes
        ----------
        nk : int
            Number of data points assigned to the node
        data : numpy.ndarrary (n, d)
            The data assigned to the Node. Each row is a datum.
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
    """
    def __init__(self, data, data_model, crp_alpha=1.0, prev_wk=1.,
                 node_level=0, level_index=0):
        """ __init__(data, data_model, crp_alpha=1.0)

            Initialise a rBHC node.

            Parameters
            ----------
            data : numpy.ndarrary (n, d)
                The data assigned to the Node. Each row is a datum.
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

        """

        self.data = data
        self.data_model = data_model
        self.crp_alpha = crp_alpha
        self.prev_wk = prev_wk
        self.node_level = node_level
        self.level_index = level_index

        self.nk = data.shape[0]


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

        print("\n", parent_node.node_level, parent_node.level_index, parent_node.nk)

        if parent_node.nk>sub_size:    # do rBHC filter
            # make subsample tree
            parent_node.subsample_bhc(sub_size)

            # set log_rk from the estimate given by self.sub_bhc
            parent_node.set_rk(parent_node.sub_bhc.root_node.log_rk)

            # filter data through top level of subsample_bhc
            parent_node.filter_data()

            # create new nodes

            child_prev_wk = (parent_node.prev_wk
                             *(1-math.exp(parent_node.log_rk)))
            child_level = parent_node.node_level+1

            left_child = cls(parent_node.left_data,
                             parent_node.data_model, 
                             parent_node.crp_alpha, child_prev_wk,
                             child_level, parent_node.level_index*2)
            right_child = cls(parent_node.right_data,
                             parent_node.data_model, 
                             parent_node.crp_alpha, child_prev_wk,
                             child_level, parent_node.level_index*2+1)
            rBHC_split = True
            children = [left_child, right_child]

       
        else:               # just use the bhc tree
            children = bhc(parent_node.data, 
                           parent_node.data_model, 
                           parent_node.crp_alpha)
            rBHC_split = False

            parent_node.set_rk(children.root_node.log_rk)

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
        self.sub_bhc = bhc(sub_data, self.data_model, self.crp_alpha)

    def filter_data(self):
        """ filter_data()

            Filter the data in a rbhc_node onto the two Nodes at the
            second from top layer of a bhc tree.
        """
        # set up data arrays with points from sub_bhc
        self.left_data = self.sub_bhc.root_node.left_child.data
        self.right_data = self.sub_bhc.root_node.right_child.data

        # get assignemnt for sub_bhc objects
        self.left_allocate = np.zeros(self.nk, dtype=bool)

        for it in np.arange(self.sub_indexes.size):
            self.left_allocate[self.sub_indexes[it]] = (
                                self.sub_bhc.assignments[-2][it]==0)


        # get non-subset data indices

        notsub_indexes = np.setdiff1d(np.arange(self.nk), 
                                      self.sub_indexes,
                                      assume_unique=True)

        # Run through non-subset data

        for ind in notsub_indexes:
            left_prob = (self.sub_bhc.root_node.left_child.log_pi
                         +self.data_model.log_marginal_likelihood(
                           np.vstack((
                                self.sub_bhc.root_node.left_child.data, 
                                self.data[ind])) ))
            right_prob = (self.sub_bhc.root_node.right_child.log_pi
                         +self.data_model.log_marginal_likelihood(
                           np.vstack((
                                self.sub_bhc.root_node.right_child.data, 
                                self.data[ind])) ))

            if left_prob>=right_prob:
                # possibly change this to make tupe and vstack at 
                # end if cost is high
                self.left_allocate[ind] = True
                self.left_data = np.vstack((self.left_data, 
                                            self.data[ind]))
            else:
                self.right_data = np.vstack((self.right_data, 
                                             self.data[ind]))

#        print(self.left_allocate)
        
