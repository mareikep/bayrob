# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport fabs, INFINITY

cdef class StateCache:
    cdef dict _cache
    cdef int max_size

    def __init__(self, int max_size=50000):
        self._cache = {}
        self.max_size = max_size

    cpdef double get_distance(self, object s1, object s2):
        cdef tuple key = (id(s1), id(s2))
        if key in self._cache:
            return self._cache[key]
        return -1.0

    cpdef void set_distance(self, object s1, object s2, double dist):
        if len(self._cache) < self.max_size:
            self._cache[(id(s1), id(s2))] = dist

cpdef double fast_jaccard_continuous(double lower1, double upper1, double lower2, double upper2):
    """Fast Jaccard similarity for continuous intervals."""
    cdef double inter_lower = max(lower1, lower2)
    cdef double inter_upper = min(upper1, upper2)

    if inter_lower >= inter_upper:
        return 0.0

    cdef double intersection = inter_upper - inter_lower
    cdef double union_val = max(upper1, upper2) - min(lower1, lower2)

    return intersection / union_val if union_val > 0 else 0.0

cpdef double fast_jaccard_sets(set s1, set s2):
    """Fast Jaccard similarity for sets."""
    cdef int inter_size = len(s1.intersection(s2))
    cdef int union_size = len(s1.union(s2))

    return <double>inter_size / <double>union_size if union_size > 0 else 1.0