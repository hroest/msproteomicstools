# distutils: language = c++
# cython: c_string_type=str, c_string_encoding=ascii
cimport cython
cimport libc.stdlib
cimport numpy as np
from cython.operator cimport dereference as deref, preincrement as inc, address as address

from PeakgroupWrapper cimport CyPrecursorWrapperOnly

from libcpp.map cimport map as libcpp_map
cdef libcpp_map[int, libcpp_string] group_label_map
cdef libcpp_map[libcpp_string, int] group_label_map_rev

cdef class CyPrecursorGroup(object):
    """See :class:`.PrecursorGroup` for a description.

    This implementation is pure Cython.

    Attributes:
        - self.peptide_group_label_: Identifier or precursor group 
        - self.run_: Reference to the :class:`.Run` where this PrecursorGroup is from
        - self.precursors_: List of :class:`.CyPrecursorWrapperOnly`
    """

    def __init__(self, str peptide_group_label, run):
        cdef libcpp_string s = peptide_group_label
        if (group_label_map_rev.find(s) == group_label_map_rev.end()):
            group_label_map[ group_label_map.size() ] = s
            group_label_map_rev[ s ] = group_label_map.size()

        self.peptide_group_label_ = group_label_map_rev[ s ]

    def __str__(self):
        return "PrecursorGroup %s" % (self.getPeptideGroupLabel())

    def __iter__(self):
        cdef libcpp_vector[ c_precursor ].iterator it
        it = self.prec_vec_.begin()
        while it != self.prec_vec_.end():
            result = CyPrecursorWrapperOnly(None, None, False)
            result.inst = address(deref(it))
            yield result

    # def __classInvariant__(self):
    #     if len(self.precursors_) > 0:
    #         # All precursor sequences should all be equal to the first sequence
    #         assert(all( [precursor.getSequence() == self.precursors_[0].getSequence() for precursor in self.precursors_] )) 
    #     return True

    # @class_invariant(__classInvariant__)
    def getPeptideGroupLabel(self):
        """
        getPeptideGroupLabel(self)
        Get peptide group label
        """
        return group_label_map[ self.peptide_group_label_ ]
  
    # @class_invariant(__classInvariant__)
    def addPrecursor(self, CyPrecursorWrapperOnly precursor):
        """
        addPrecursor(self, precursor)
        Add precursor to peptide group
        """
        ### This by itself (not adding any precursors
        ### reduces it to 212412maxresident)k      
        ### from 304580maxresident)k            
        ### == 90 MB are the precursors themselves (200 bytes)
        ## precursor.set_precursor_group( self )
        # self.precursors_.append(precursor)
        # print(" addPrec, push back")

        self.prec_vec_.push_back(deref(precursor.inst))


    # @class_invariant(__classInvariant__)
    def getPrecursor(self, curr_id):
        """
        getPrecursor(self, curr_id)
        Get the precursor for the given transition group id
        """
        for precursor in self:
            if precursor.get_id() == curr_id:
                return precursor
        return None

    # @class_invariant(__classInvariant__)
    def getAllPrecursors(self):
        """
        getAllPrecursors(self)
        Return a list of all precursors in this precursor group
        """
        return list(self)

    # @class_invariant(__classInvariant__)
    def getAllPeakgroups(self):
        """
        getAllPeakgroups(self)
        Generator of all peakgroups attached to the precursors in this group
        """
        for pr in self:
            for pg in pr.get_all_peakgroups():
                yield pg

    # @class_invariant(__classInvariant__)
    def getOverallBestPeakgroup(self):
        """
        getOverallBestPeakgroup(self)
        Get the best peakgroup (by fdr score) of all precursors contained in this precursor group
        """
        allpg = list(self.getAllPeakgroups())
        if len(allpg) == 0:
            return None

        minscore = min([pg.get_fdr_score() for pg in allpg])
        return [pg for pg in allpg if pg.get_fdr_score() <= minscore][0]

    def get_decoy(self):
        """
        Whether the current peptide is a decoy or not

        Returns:
            decoy(bool): Whether the peptide is decoy or not
        """
        if self.prec_vec_.empty():
            return False

        return self.prec_vec_[0].get_decoy()

