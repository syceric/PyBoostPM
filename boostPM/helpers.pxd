import numpy as np
from libc.math cimport lgamma

cdef extern from "rand.h":
    int sample(double* probs, int size)
    double gen_uni_rand()

cdef int rand_choice(double[::1] probs)

cdef double gen_rand()

cdef inline double[::1] add(double[::1] a, double[::1] b):
    cdef:
        int i, n = a.shape[0]
        double[::1] out = np.empty(n)

    for i in range(n):
        out[i] = a[i] + b[i]

    return out

cdef inline double log_beta(double num1, double num2) nogil:
    return lgamma(num1) + lgamma(num2) - lgamma(num1 + num2)

cdef double find_max(double[::1] vec)

cdef double find_max_mat(double[:, ::1] mat)

cdef double log_sum_vec(double[::1] log_x)

cdef double log_sum_mat(double[:, ::1] log_mat)

cdef double[::1] normalize_log_vec(double[::1] log_x)

cdef double[:, ::1] normalize_log_mat(double[:, ::1] log_mat)

cdef int[:, ::1] matrix_hist(double[:, ::1] data, int[::1] obs_indices, int num_bins, double[::1] low, double[::1] high)

cdef int[:, ::1] cdf(double[:, ::1] data, int[::1] obs_indices, int num_bins, double[::1] low, double[::1] high)

cdef tuple children_node_obs_indices(double[:, ::1] data, int[::1] obs_indices, int split_feature, int split_bin, int num_bins, double left, double right)

cdef tuple sample_split_rule(double[:, ::1] log_likelihood_mat)

cdef double find_log_volume(double[::1] left, double[::1] right)

cdef inline double find_volume_ratio(double left, double right, double threshold):
    return (threshold - left) / (right - left)

cdef inline double find_threshold(double left, double right, double location):
    return left + (right - left) * location
