# distutils: language = c++
# cython: c_string_type=str, c_string_encoding=ascii
cimport cython
cimport libc.stdlib
cimport numpy as np

from libcpp.string cimport string as libcpp_string
from libcpp.vector cimport vector as libcpp_vector
from libcpp.map cimport map as libcpp_map
from libcpp.pair cimport pair as libcpp_pair
from cython.operator cimport dereference as deref, preincrement as inc, address as address
from libcpp cimport bool

# from PeakgroupWrapper cimport c_peakgroup, CyPeakgroupWrapperOnly

cdef class CyPeakgroupWrapperOnly(object):
    """
    See :class:`.PeakGroupBase` for a detailed description.

    This implementation stores a pointer to a C++ object holding the actual
    data. The data access works very similarly as for any :class:`.PeakGroupBase`.
    """

    def __dealloc__(self):
        pass

    def __init__(self):
        pass

    def __richcmp__(self, other, op):
        # op_str = {0: '<', 1: '<=', 2: '==', 3: '!=', 4: '>', 5: '>='}[op]
        """
        Larger than operator, allows sorting in Python 3.x
        """
        if op == 4:
            return self.get_feature_id() > other.get_feature_id()

    # Do not allow setting of any parameters (since data is not stored here)
    def set_fdr_score(self, fdr_score):
        raise Exception("Cannot set in immutable object")

    def set_normalized_retentiontime(self, normalized_retentiontime):
        raise Exception("Cannot set in immutable object")

    def set_feature_id(self, id_):
        raise Exception("Cannot set in immutable object")

    def set_intensity(self, intensity):
        raise Exception("Cannot set in immutable object")

    def get_intensity(self):
        # return deref(self.inst).intensity_
        return 99

    def getPeptide(self):
        return self.peptide

    def get_dscore(self):
        # return deref(self.inst).dscore_
        return 7

    ## Select / De-select peakgroup
    def select_this_peakgroup(self):
        ## self.peptide.select_pg(self.get_feature_id())
        deref(self.inst).cluster_id_ = 1

    ## Select / De-select peakgroup
    def setClusterID(self, id_):
        raise Exception("Not implemented!")
        ### self.cluster_id_ = id_
        ### self.peptide.setClusterID(self.get_feature_id(), id_)

    def get_cluster_id(self):
        return deref(self.inst).cluster_id_
  
    def __str__(self):
        return "PeakGroup %s at %s s in %s with score %s (cluster %s)" % (self.get_feature_id(), self.get_normalized_retentiontime(), "run?", self.get_fdr_score(), self.get_cluster_id())

    def get_value(self, value):
        raise Exception("Needs implementation")

    def set_value(self, key, value):
        raise Exception("Needs implementation")

    def set_fdr_score(self, fdr_score):
        self.fdr_score = fdr_score

    def get_fdr_score(self):
        return deref(self.inst).fdr_score

    def set_normalized_retentiontime(self, float normalized_retentiontime):
        self.normalized_retentiontime = normalized_retentiontime

    def get_normalized_retentiontime(self):
        return deref(self.inst).normalized_retentiontime

    cdef get_normalized_retentiontime_cy(self):
        return deref(self.inst).normalized_retentiontime

    def set_feature_id(self, id_):
        raise Exception("Not implemented!")
        # self.id_ = id_

    def get_feature_id(self):
        return <str>(deref(self.inst).getInternalId())

