cdef extern from "rand.h":
    int sample(double* probs, int size)
    double gen_uni_rand()

cdef int rand_choice(double[::1] probs):
    return sample(&probs[0], probs.shape[0])

cdef double gen_rand():
    return gen_uni_rand()

from libc.math cimport sin, exp, floor, log
from cython.parallel cimport prange
import cython
from numpy.math cimport INFINITY
import time
import numpy as np
import os
os.environ["CC"] = "g++"

cdef double find_max(double[::1] vec):
    cdef int i, n=vec.shape[0]
    cdef double curr = -INFINITY
    for i in range(vec.shape[0]):
        if vec[i] >= curr:
            curr = vec[i]
    return curr

cdef double find_max_mat(double[:, ::1] mat):
    cdef int i, j, m=mat.shape[0], n=mat.shape[1]
    cdef double curr = -INFINITY
    
    for i in range(m):
        for j in range(n):
            if mat[i, j] >= curr:
                curr = mat[i, j]
    return curr

cdef double log_sum_vec(double[::1] log_x):
    cdef double log_x_max = find_max(log_x)
    cdef int i, n = log_x.shape[0]
    cdef double out = 0.0
    for i in range(n):
        out += exp(log_x[i] - log_x_max) # to avoid overflow
    return log_x_max + log(out)

cdef double log_sum_mat(double[:, ::1] log_mat):
    cdef double log_mat_max = find_max_mat(log_mat)
    cdef int i, j, m=log_mat.shape[0], n=log_mat.shape[1]
    cdef double out = 0.0
    
    for i in range(m):
        for j in range(n):
            out += exp(log_mat[i, j] - log_mat_max) # to avoid overflow
    return log_mat_max + log(out)

cdef double[::1] normalize_log_vec(double[::1] log_x):
    cdef int i
    cdef double[::1] out = np.empty(log_x.shape[0])
    cdef double log_sum = log_sum_vec(log_x)

    for i in range(log_x.shape[0]):
        out[i] = exp(log_x[i] - log_sum) # to avoid overflow
    return out

cdef double[:, ::1] normalize_log_mat(double[:, ::1] log_mat):
    cdef int i, j
    cdef int m=log_mat.shape[0], n=log_mat.shape[1]
    cdef double[:, ::1] out = np.empty((m, n))
    cdef double log_sum = log_sum_mat(log_mat)
    
    for i in range(m):
        for j in range(n):
            out[i, j] = exp(log_mat[i, j] - log_sum) # to avoid overflow
    return out

@cython.cdivision(True)
cdef int[:, ::1] matrix_hist(double[:, ::1] data, int[::1] obs_indices, int num_bins, double[::1] low, double[::1] high):
    cdef int i, j, n = obs_indices.shape[0], d = data.shape[1]
    cdef int[:, ::1] out = np.empty((d, num_bins), dtype=np.intc)
    cdef double bin_size
    
    out[:, :] = 0
    
    for i in prange(d, nogil=True, schedule='static', chunksize=1):
        bin_size = (high[i] - low[i])/ (<double> num_bins)
        for j in range(n):
            out[i, <int> floor((data[obs_indices[j], i] - low[i])/bin_size)] += 1
    return out

cdef int[:, ::1] cdf(double[:, ::1] data, int[::1] obs_indices, int num_bins, double[::1] low, double[::1] high):
    cdef int[:, ::1] out = matrix_hist(data, obs_indices, num_bins, low, high)
    cdef int d = data.shape[1]
    cdef int i, j
    
    for i in prange(d, nogil=True, schedule='static', chunksize=1):
        for j in range(1, num_bins):
            out[i, j] += out[i, j-1]
    
    return out

cdef tuple children_node_obs_indices(double[:, ::1] data, int[::1] obs_indices, int split_feature, int split_bin, int num_bins, double left, double right):
    cdef:
        int i, j, n = obs_indices.shape[0]
        double bin_size = (right - left)/num_bins
        double threshold = left + split_bin * bin_size
        int[::1] left_node_obs_indices = np.empty(n, dtype=np.intc)
        int[::1] right_node_obs_indices = np.empty(n, dtype=np.intc)
        int curr_left_arr_idx = 0
        int curr_right_arr_idx = 0
    
    for i in range(n):
        if data[obs_indices[i], split_feature] < threshold:
            left_node_obs_indices[curr_left_arr_idx] = obs_indices[i]
            curr_left_arr_idx +=1
        else:
            right_node_obs_indices[curr_right_arr_idx] = obs_indices[i]
            curr_right_arr_idx += 1

    return (left_node_obs_indices[:curr_left_arr_idx], right_node_obs_indices[:curr_right_arr_idx], curr_left_arr_idx)

cdef tuple sample_split_rule(double[:, ::1] log_likelihood_mat):
    cdef:
        double[:, ::1] prob
        int linear_idx, mat_idx1, mat_idx2
    
    prob = normalize_log_mat(log_likelihood_mat)
    linear_idx = rand_choice(np.ravel(prob))
    mat_idx1, mat_idx2 = np.unravel_index(linear_idx, (prob.shape[0], prob.shape[1]))
    return mat_idx1, mat_idx2+1

cdef double find_log_volume(double[::1] left, double[::1] right):
    cdef int i, n = left.shape[0]
    cdef double out = 0.0

    for i in range(n):
        out += log(right[i] - left[i])
    
    return out