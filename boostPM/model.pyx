# distutils: language = c++

from .helpers cimport *

from libc.math cimport lgamma, sin, exp, floor, log
from cython.parallel cimport prange
import cython
from numpy.math cimport INFINITY
import time
import numpy as np
import os
os.environ["CC"] = "gcc"

cdef class Node():
    cdef:
        public int depth, split_feature, dim
        public double split_location, theta, log_volume, volume_ratio, split_threshold
        public double[::1] left_coords, right_coords
        public Node left, right, parent
        public bint isLeaf, isRoot
    
    def __init__(self, int depth=1, int split_feature=-1, int dim = 1, double split_location=-1.0, double theta=0.0,\
                 double[::1] left_coords=np.empty(shape=0, dtype=np.double), double[::1] right_coords=np.empty(shape=0, dtype=np.double), \
                 double log_volume=0.0, double volume_ratio=0.0, double split_threshold=-1.0, \
                 left=None, right=None, parent=None, bint isLeaf=False, bint isRoot=False):
        self.depth = depth
        self.split_feature = split_feature
        self.dim = dim
        self.split_location = split_location
        self.theta = theta
        self.left_coords = left_coords
        self.right_coords = right_coords
        self.log_volume = log_volume
        self.split_threshold = split_threshold
        self.volume_ratio = volume_ratio
        self.left = left
        self.right = right
        self.parent = parent
        self.isLeaf = isLeaf
        self.isRoot = isRoot
        
    # not the most elegant way, but it will do
    # REMEMBER TO CALL THESE SETTERS IN ORDER AFTER UPDATING split_feature and split_location!!!!!
    cdef void set_split_threshold(self):
        self.split_threshold = find_threshold(self.left_coords[self.split_feature], self.right_coords[self.split_feature], self.split_location)

    cdef void set_log_volume(self):
        self.log_volume = find_log_volume(self.left_coords, self.right_coords)
    
    cdef void set_volume_ratio(self):
        self.volume_ratio = find_volume_ratio(self.left_coords[self.split_feature], self.right_coords[self.split_feature], self.split_threshold)
    
    cdef inline void mover(self, double[::1] x):
        cdef:
            int split_feature = self.split_feature
            double split_threshold = self.split_threshold
            double theta = self.theta
            double vol_ratio = self.volume_ratio
            double left = self.left_coords[split_feature]
            double right = self.right_coords[split_feature]

        if x[split_feature] < split_threshold:
            x[split_feature] = left + (theta/vol_ratio)*(x[split_feature] - left)
        else:
            x[split_feature] = right + ((1 - theta)/(1 - vol_ratio)) * (x[split_feature] - right)

cdef class BaseDecisionTree:
    cdef:
        public int max_depth, min_obs, num_bins, dim
        public double lr, alpha, beta, gamma
        public bint fitted
        public Node root_node
    
    def __init__(self, int max_depth=5, int dim=1, double lr=0.0, double gamma=0.0, double alpha = 0.5, double beta = 0.0, int min_obs = 10, \
                 int num_bins=128):
        self.max_depth = max_depth
        self.dim = dim
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha # prior probability of splitting node is alpha * (1 + node.depth)^beta
        self.beta = beta
        self.min_obs = min_obs
        self.num_bins = num_bins # number of bins used in histogramming each feature dimension
        self.fitted = False # to check if the tree is fitted, should be set to True only in the fit function
        self.root_node = Node(depth=1, left_coords = np.zeros(dim), right_coords=np.ones(dim), isRoot=True)

    cdef void _split_node(self, Node node, double[:, ::1] data, int[::1] obs_indices):
        cdef:
            int d = self.dim
            int L = self.num_bins
            int n_curr = obs_indices.shape[0]
            int split_feature, split_bin
            int i
            double prior_split_prob = self.alpha * (1 + node.depth)**self.beta
            double post_stop_split_prob
            double prec = (1. + node.depth)**self.gamma
            double modified_lr
            double loc_val, log_loc_val, log_one_minus_loc_val
            double[:, ::1] log_likelihood_mat
            double[::1] prob_vec = np.empty(2)
            double[::1] loc = np.empty(L-1)
            double[::1] left_node_right_coords, right_node_left_coords
            int left_count
            int[:, ::1] CDF
            int feature_idx, loc_idx
            
            double term1, term3 # to save time in nested loop below

        for i in range(L-1):
            loc[i] = (i+1)/(<double> L)
                              
        if n_curr < self.min_obs or node.depth >= self.max_depth:
            node.isLeaf = True
            return

        log_likelihood_mat = np.empty((d, L-1))
        
        CDF = cdf(data, obs_indices, L, node.left_coords, node.right_coords)
        
        term1 = log(prior_split_prob/(<double> d*(L - 1)))
        
        ##################################################################################################################################
        ###                         This part constitutes a major bottleneck, taking up around 40% of runtime.                         ###
        ### Because the computation is repeated (number of total nodes) * (number of bins) * (number of feature dimension) many times. ###
        ###                                                  Can we do better???                                                       ###
        ##################################################################################################################################
        
        # construct the likelihood matrix (p.44 of the paper)
        for loc_idx in prange(L-1, nogil=True, schedule='static', chunksize=1):
        # for loc_idx in range(L - 1):
            loc_val = loc[loc_idx]
            log_loc_val = log(loc_val)
            log_one_minus_loc_val = log(1 - loc_val)
            
            term3 = log_beta(prec * loc_val, prec * (1. - loc_val))
            
            # for feature_idx in prange(d, nogil=True, schedule='static', chunksize=8):
            for feature_idx in range(d):
                if node.right_coords[feature_idx] - node.left_coords[feature_idx] < 1e-10:
                    log_likelihood_mat[feature_idx, :] = 0.0
                else:
                    left_count = CDF[feature_idx, loc_idx]

                    log_likelihood_mat[feature_idx, loc_idx] = \
                        term1 + log_beta(prec * loc_val + <double> left_count, prec * (1. - loc_val) + (<double> (n_curr - left_count))) - term3 - \
                        (<double> left_count) * log_loc_val - (<double> (n_curr - left_count)) * log_one_minus_loc_val   
        
        # posterior probability of *not splitting* the node
        prob_vec[0] = log((1. - prior_split_prob))
        prob_vec[1] = log_sum_mat(log_likelihood_mat)
        post_stop_split_prob = normalize_log_vec(prob_vec)[0]
        
        # post_stop_split_prob = (1. - prior_split_prob) / exp(log_sum_mat(log_likelihood_mat))
        
        if gen_rand() < post_stop_split_prob:
            node.isLeaf = True
            return
        else:
            node.isLeaf = False
            split_feature, split_bin = sample_split_rule(log_likelihood_mat)

            # update node attributes
            node.split_feature = split_feature
            node.split_location = split_bin / L # divide by L since _sample_split_rule returns the bin index that is split
            
            node.set_split_threshold()
            node.set_log_volume()
            node.set_volume_ratio()
            
            left_node_right_coords = np.array(node.right_coords, copy=True)
            right_node_left_coords = np.array(node.left_coords, copy=True)
            left_node_right_coords[split_feature] = node.split_threshold
            right_node_left_coords[split_feature] = node.split_threshold

            left_node_obs_indices, right_node_obs_indices, left_count = children_node_obs_indices(data, obs_indices, split_feature, split_bin, L, \
                                                                                                  node.left_coords[split_feature], node.right_coords[split_feature])
            
            node.left = Node(parent = node, depth = node.depth + 1, left_coords=node.left_coords, right_coords=left_node_right_coords)
            node.right = Node(parent = node, depth = node.depth + 1, left_coords=right_node_left_coords, right_coords=node.right_coords)

            modified_lr = self.lr * (1. - node.log_volume/log(2.))**(-self.gamma)
            node.theta = (1. - modified_lr) * node.split_location + modified_lr * left_count/n_curr

            self._split_node(node.left, data, left_node_obs_indices)
            self._split_node(node.right, data, right_node_obs_indices)
            
            return
    
    cpdef void fit(self, double[:, ::1] data):
        if self.dim != data.shape[1]:
            raise Exception("data dimension does not match with model dimension")
        self._split_node(self.root_node, data, np.arange(data.shape[0], dtype=np.intc))
        self.fitted = True
        
    cdef list _find_terminal_node(self, double[:, ::1] data):
        cdef:
            int i, n = data.shape[0]
            list out = []
            Node current_node
        
        for i in range(n):
            current_node = self.root_node

            while current_node.isLeaf == False:
                if data[i, current_node.split_feature] < current_node.split_threshold:
                    current_node = current_node.left
                else:
                    current_node = current_node.right

            out.append(current_node)
        return out
    
    cpdef int count_leaf(self, Node node):
        if node.isLeaf == False:
            return self.count_leaf(node.left) + self.count_leaf(node.right)
        else:
            return 1
    
    @cython.cdivision(True)
    cpdef void residualize(self, double[:, ::1] data):
        cdef:
            int i, n=data.shape[0]
            int split_feature
            double split_threshold, theta, vol_ratio, left, right
            list terminal_nodes = self._find_terminal_node(data)
            Node current_node, parent
            
        for i in range(n):
            current_node = terminal_nodes[i]
            while not current_node.isRoot:
                parent = current_node.parent
                split_threshold = parent.split_threshold
                split_feature = parent.split_feature
                left = parent.left_coords[split_feature]
                right = parent.right_coords[split_feature]
                theta = parent.theta
                vol_ratio = parent.volume_ratio
                
                # the "local mover" functions
                if data[i, split_feature] < split_threshold:
                    data[i, split_feature] = left + (theta/vol_ratio)*(data[i, split_feature] - left)
                else:
                    data[i, split_feature] = right + ((1. - theta)/(1. - vol_ratio)) * (data[i, split_feature] - right)
                current_node = parent
    
    cpdef double[::1] eval_log_density(self, double[:, ::1] x):
        cdef:
            int i, j, n = x.shape[0], d = x.shape[1]
            list terminal_nodes = self._find_terminal_node(x)
            double[::1] out = np.zeros(n)
            Node current_node, parent
        
        for i in range(n):
            current_node = terminal_nodes[i]
            while current_node.isRoot == False:
                parent = current_node.parent
                if parent.left == current_node:
                    out[i] += log(parent.theta/parent.split_location)
                else:
                    out[i] += log((1. - parent.theta)/(1. - parent.split_location))
                current_node = parent
        return out
    
    cpdef void simulate(self, double[:, ::1] base_samples):
        cdef:
            int i, n = base_samples.shape[0]
            int split_feature
            double z, left, right, theta, threshold
            Node curr_node
        
        for i in range(n):
            curr_node = self.root_node
            while curr_node.isLeaf == False:
                split_feature = curr_node.split_feature
                left = curr_node.left_coords[curr_node.split_feature]
                right = curr_node.right_coords[curr_node.split_feature]
                threshold = curr_node.split_threshold
                theta = curr_node.theta
                z = (base_samples[i, split_feature] - left)/(right - left)
                if base_samples[i, split_feature] <= left + theta * (right - left):
                    base_samples[i, split_feature] = left + (threshold - left) * z / theta
                else:
                    base_samples[i, split_feature] = threshold + ((right - threshold)/(1 - theta)) * (z - theta)
                    
                if base_samples[i, split_feature] < threshold:
                    curr_node = curr_node.left
                else:
                    curr_node = curr_node.right
        

cdef class DensityEstimator:
    cdef:
        public int n_estimators, dim, max_depth, min_obs, num_bins, n_early_stop, n_min_leaves
        public double lr, alpha, beta, gamma, subsample
        public double[:, ::1] data
        public list trees
    
    def __init__(self, int n_estimators=100, int dim=1, int max_depth=10, double lr=0.1, double alpha=0.5, double beta=0.0,\
                 double gamma=0.0, double subsample=1.0, int min_obs=10, int num_bins=128, int n_early_stop=0, int n_min_leaves=1024):
        self.n_estimators = n_estimators
        self.dim = dim
        self.max_depth = max_depth
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.subsample = subsample
        self.min_obs = min_obs
        self.num_bins = num_bins
        self.n_min_leaves = n_min_leaves
        self.n_early_stop = n_early_stop # how many consecutive trees with leaves less than n_min_leaves we accept before we stop boosting
        self.trees = [BaseDecisionTree(max_depth=max_depth, dim=dim, lr=lr, gamma=gamma, alpha=alpha, beta=beta, min_obs=min_obs, num_bins=num_bins)]
    
    cpdef void fit(self, double[:, ::1] data):
        cdef:
            int i, count=0, count_small_trees=0, n = data.shape[0]
            int max_depth=self.max_depth
            double lr = self.lr
            double alpha = self.alpha
            double beta = self.beta
            double gamma = self.gamma
            int min_obs = self.min_obs
            int num_bins = self.num_bins
            int n_early_stop = self.n_early_stop

            double[:, ::1] x = np.copy(data, order='C') # the order keyword is important!!!
            double[:, ::1] x_subsampled
            int[::1] idx
            
            BaseDecisionTree curr_tree
        
        if n_early_stop == 0:
            for i in range(self.n_estimators):
                curr_tree = self.trees[i]
                idx = np.random.default_rng().integers(n, size = <int> floor(n*self.subsample), dtype=np.int32)
                x_subsampled = np.ascontiguousarray(np.array(x)[idx])
                curr_tree.fit(x_subsampled)
                curr_tree.residualize(x)
                if i < self.n_estimators - 1:
                    self.trees.append(BaseDecisionTree(max_depth=max_depth, dim=self.dim, lr=lr, gamma=gamma, alpha=alpha, beta=beta, min_obs=min_obs, num_bins=num_bins))
                    count += 1
        
        else:
            for i in range(self.n_estimators):
                curr_tree = self.trees[i]
                curr_tree.fit(x)
                if curr_tree.count_leaf(curr_tree.root_node) < self.n_min_leaves:
                    count_small_trees += 1
                else:
                    count_small_trees = 0

                if count_small_trees > self.n_early_stop:
                    break

                curr_tree.residualize(x)
                self.trees.append(BaseDecisionTree(max_depth=max_depth, dim=self.dim, lr=lr, gamma=gamma, alpha=alpha, beta=beta, min_obs=min_obs, num_bins=num_bins))
                count += 1

    cpdef void residualize(self, double[:, ::1] data):
        cdef:
            int i, n = len(self.trees)
        
        for i in range(n):
            self.trees[i].residualize(data)
            
    cpdef double[::1] eval_log_density(self, double[:, ::1] data):
        cdef:
            int i 
            double[::1] out = np.zeros(data.shape[0])
            double[:, ::1] x = np.copy(data, order='C')

        for i in range(self.n_estimators):
            out = add(out,self.trees[i].eval_log_density(x))
            self.trees[i].residualize(x)
        
        return out
    
    cpdef double[:, ::1] simulate(self, int n):
        cdef:
            int i
            double[:, ::1] out = np.random.default_rng().random((n, self.dim))
        
        for i in range(self.n_estimators):
            self.trees[self.n_estimators - 1 - i].simulate(out)
        
        return out

    cpdef double[:, ::1] simulate_from_base(self, double[:, ::1] base):
        cdef:
            int i
            double[:, ::1] out = np.copy(base, order='C')
        
        for i in range(self.n_estimators):
            self.trees[self.n_estimators - 1 - i].simulate(out)
        
        return out

cdef class TwoStageDensityEstimator:
    cdef:
        public list models = []
        public int dim
        public dict marginal_params = {""}
    
    def __init__(self, int dim, dict marginal_params, dict copula_params):
        self.models = []

        marginal_n = marginal_params.get('n_estimators', 100)
        marginal_depth = marginal_params.get('max_depth', 10)
        marginal_lr = marginal_params['lr', 0.1]
        marginal_alpha = marginal_params['alpha', 0.5]
        marginal_beta = marginal_params['beta', 0.0]
        marginal_gamma = marginal_params['gamma', 0.0]
        marginal_subsample = marginal_params['subsample', 1.0]
        for i in range(dim):
            self.models.append(DensityEstimator(n_estimators=marginal_params['n_estimators'], dim=1, \
                                                max_depth=marginal_params['max_depth'], lr=marginal_params['lr']))