# distutils: language = c++
# cython: c_string_type=str, c_string_encoding=ascii
cimport cython
cimport libc.stdlib
cimport numpy as np
from libcpp.vector cimport vector as libcpp_vector

from PeakgroupWrapper cimport c_precursor

from libcpp.string cimport string as libcpp_string

cdef class CyPrecursorGroup(object):
    """See :class:`.PrecursorGroup` for a description.

    This implementation is using a C++ array to store precursors Cython.

    Attributes:
        - self.peptide_group_label_: Identifier or precursor group 
        - self.run_: Reference to the :class:`.Run` where this PrecursorGroup is from
        - self.precursors_: List of :class:`.CyPrecursorWrapperOnly`

    1045 struct __pyx_obj_20msproteomicstoolslib_6cython_14PrecursorGroup_CyPrecursorGroup 
    1046   PyObject_HEAD
    1047   std::string peptide_group_label_;
    1048   PyObject *run_;
    1049   std::vector<c_precursor>  prec_vec_; 

    245316maxresident)k 
    263236maxresident)k # 20 MB for run object map
    276808maxresident)k # 10 MB for empty string
    297808maxresident)k # 15 MB for full string

    """

    # cdef libcpp_string peptide_group_label_ 
    cdef int peptide_group_label_ 
    cdef object run_
    cdef libcpp_vector[ c_precursor ] prec_vec_


