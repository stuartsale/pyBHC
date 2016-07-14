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

    def __init__(self, data, data_model, crp_alpha=1.0):

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
        """        
        # initialize the tree

        self.assignments = [np.zeros(data.size)]

#        self.nodes = []
        root_node = rbhc_Node(data, data_model, crp_alpha)

        self.tree = rbhc_Node.recursive_split(root_node, 50)




class rbhc_Node(object):

    def __init__(self, data, data_model, crp_alpha=1.0, parent=None):

        self.data = data
        self.data_model = data_model
        self.crp_alpha = crp_alpha

        self.nk = data.shape[0]

    @classmethod
    def recursive_split(cls, parent_node, sub_size):

        print(parent_node.nk)

        if parent_node.nk>sub_size:    # do rBHC filter
            # make subsample tree
            parent_node.subsample_bhc(sub_size)

            # filter data through top level of subsample_bhc
            parent_node.filter_data()

            # create new nodes

            left_child = cls(parent_node.left_data,
                             parent_node.data_model, 
                             parent_node.crp_alpha,
                             parent_node)
            right_child = cls(parent_node.right_data,
                             parent_node.data_model, 
                             parent_node.crp_alpha,
                             parent_node)

            # recurse down

            rbhc_Node.recursive_split(left_child, sub_size)
            rbhc_Node.recursive_split(right_child, sub_size)
            
        
        else:               # just use the bhc tree
            parent_node.final_tree = bhc(parent_node.data, 
                                         parent_node.data_model, 
                                         parent_node.crp_alpha)
        


    def subsample_bhc(self, sub_size):

        self.sub_indexes = np.random.choice(np.arange(self.nk), 
                                            sub_size, replace=False)
        sub_data = self.data[self.sub_indexes]
        self.sub_bhc = bhc(sub_data, self.data_model, self.crp_alpha)

    def filter_data(self):

        self.left_data = self.sub_bhc.root_node.left_child.data
        self.right_data = self.sub_bhc.root_node.right_child.data

        # get non-subset data indices

        notsub_indexes = np.setdiff1d(np.arange(self.nk), 
                                      self.sub_indexes,
                                      assume_unique=True)

        # Run through non-subset data

        for ind in notsub_indexes:
            left_prob = (self.sub_bhc.root_node.left_child.log_pi
                         + self.data_model.log_marginal_likelihood(
                            np.vstack((self.left_data, 
                                            self.data[ind])) ))
            right_prob = (self.sub_bhc.root_node.right_child.log_pi
                         + self.data_model.log_marginal_likelihood(
                            np.vstack((self.right_data, 
                                            self.data[ind])) ))

            if left_prob>=right_prob:
                # possibly change this to make tupe and vstack at 
                # end if cost is high
                self.left_data = np.vstack((self.left_data, 
                                            self.data[ind]))
            else:
                self.right_data = np.vstack((self.right_data, 
                                             self.data[ind]))
        
