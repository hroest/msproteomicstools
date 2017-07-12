# distutils: language = c++
# cython: c_string_encoding=ascii  # for cython>=0.19
# encoding: latin-1
cimport cython
cimport libc.stdlib
cimport numpy as np

from libcpp.string cimport string as libcpp_string
from libcpp.vector  cimport vector as libcpp_vector
from cython.operator cimport dereference as deref, preincrement as inc, address as address
from libcpp cimport bool

"""
cython -a --cplus msproteomicstoolslib/_optimized.pyx &&  python setup.py  build && cp build/lib.linux-x86_64-2.7/msproteomicstoolslib/_optimized.so msproteomicstoolslib/
"""


import numpy as np
import operator


ctypedef np.float32_t DATA_TYPE

class PeakGroupBase(object):

    __slots__ = ["fdr_score", "normalized_retentiontime", "id_", "intensity_", "cluster_id_"]

    def __init__(self):
        self.fdr_score = None
        self.normalized_retentiontime = None
        self.id_ = None
        self.intensity_ = None
        self.cluster_id_ = -1
  
    def __str__(self):
        return "PeakGroup %s at %s s in %s with score %s (cluster %s)" % (self.get_feature_id(), self.get_normalized_retentiontime(), self.peptide.run, self.get_fdr_score(), self.get_cluster_id())

    def get_value(self, value):
        raise Exception("Needs implementation")

    def set_value(self, key, value):
        raise Exception("Needs implementation")

    def set_fdr_score(self, fdr_score):
        self.fdr_score = fdr_score

    def get_fdr_score(self):
        return self.fdr_score

    def set_normalized_retentiontime(self, normalized_retentiontime):
        self.normalized_retentiontime = float(normalized_retentiontime)

    def get_normalized_retentiontime(self):
        return self.normalized_retentiontime

    def set_feature_id(self, id_):
        self.id_ = id_

    def get_feature_id(self):
        return self.id_

    def set_intensity(self, intensity):
      self.intensity_ = intensity

    def get_intensity(self):
        return self.intensity_

    # Selected
    def is_selected(self):
        return self.cluster_id_ == 1

    def select_this_peakgroup(self):
        self.cluster_id_ = 1

    def get_cluster_id(self):
        return self.cluster_id_

class MinimalPeakGroup(PeakGroupBase):
    """
    A single peakgroup that is defined by a retention time in a chromatogram
    of multiple transitions. Additionally it has an fdr_score and it has an
    aligned RT (e.g. retention time in normalized space).
    A peakgroup can be selected for quantification or not (this is stored as
    having cluster_id == 1).
    
    Note that for performance reasons, the peakgroups are created on-the-fly
    and not stored as objects but rather as tuples in "Peptide".

    Each peak group has a unique id, a score (fdr score usually), a retention
    time as well as a back-reference to the precursor that generated the
    peakgroup.
    In this case, the peak group can also be assigned a cluster id (where the
    cluster 1 is special as the one we will use for quantification).
    """

    def __init__(self, unique_id, fdr_score, assay_rt, selected, cluster_id, peptide, intensity=None, dscore=None):
      super(MinimalPeakGroup, self).__init__()
      self.id_ = unique_id
      self.fdr_score = fdr_score
      self.normalized_retentiontime = assay_rt 
      self.cluster_id_ = cluster_id
      self.peptide = peptide
      self.intensity_ = intensity
      self.dscore_ = dscore

    ### def __lt__(self, other):
    ###     """
    ###     Larger than operator, allows sorting in Python 3.x
    ###     """
    ###     return self.id_ > other.id_
  
    ## Print
    def print_out(self):
        # return self.run.get_id() + "/" + self.get_id() + " " + str(self.get_fdr_score()) + " " + str(self.get_normalized_retentiontime()) + " " + str(self.get_value("RT")) + " " + str(self.get_value("rt_score")) # rt_score = delta iRT
        return self.peptide.run.get_id() + "/" + self.get_feature_id() + " score:" + str(self.get_fdr_score()) + " RT:" + str(self.get_normalized_retentiontime()) # + " " + str(self.get_value("RT")) + " " + str(self.get_value("rt_score")) # rt_score = delta iRT

    # Do not allow setting of any parameters (since data is not stored here)
    def set_fdr_score(self, fdr_score):
        raise Exception("Cannot set in immutable object")

    def set_normalized_retentiontime(self, normalized_retentiontime):
        raise Exception("Cannot set in immutable object")

    def set_feature_id(self, id_):
        raise Exception("Cannot set in immutable object")

    def set_intensity(self, intensity):
        raise Exception("Cannot set in immutable object")

    def get_dscore(self):
        return self.dscore_

    ## Select / De-select peakgroup
    def select_this_peakgroup(self):
        self.peptide.select_pg(self.get_feature_id())

    ## Select / De-select peakgroup
    def setClusterID(self, id_):
        self.cluster_id_ = id_
        self.peptide.setClusterID(self.get_feature_id(), id_)

    def get_cluster_id(self):
        return self.cluster_id_



cdef extern from "peakgroup.h":
    cdef cppclass c_peakgroup:
        c_peakgroup()

        double fdr_score 
        double normalized_retentiontime 
        libcpp_string internal_id_ 
        double intensity_
        double dscore_ 
        int cluster_id_


cdef class CyPeakgroupWrapperOnly:
    """
    """

    cdef c_peakgroup * inst 
    cdef object peptide

    def __dealloc__(self):
        pass

    def __init__(self):
        pass

    # Do not allow setting of any parameters (since data is not stored here)
    def set_fdr_score(self, fdr_score):
        raise Exception("Cannot set in immutable object")

    def set_normalized_retentiontime(self, normalized_retentiontime):
        raise Exception("Cannot set in immutable object")

    def set_feature_id(self, id_):
        raise Exception("Cannot set in immutable object")

    def set_intensity(self, intensity):
        raise Exception("Cannot set in immutable object")

    def getPeptide(self):
        return self.peptide

    def get_dscore(self):
        return self.dscore_

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

    def set_feature_id(self, id_):
        raise Exception("Not implemented!")
        # self.id_ = id_

    def get_feature_id(self):
        return <bytes>(deref(self.inst).internal_id_)

    #### def set_intensity(self, float intensity):
    ####     self.intensity_ = intensity

    #### def get_intensity(self):
    ####     return self.intensity_

    #### # Selected
    #### def is_selected(self):
    ####     return self.cluster_id_ == 1

    #### def select_this_peakgroup(self):
    ####     self.cluster_id_ = 1

    #### def get_cluster_id(self):
    ####     return self.cluster_id_

cdef class CyPeakgroupWrapper:
    """
    A single peakgroup that is defined by a retention time in a chromatogram
    of multiple transitions. Additionally it has an fdr_score and it has an
    aligned RT (e.g. retention time in normalized space).
    A peakgroup can be selected for quantification or not (this is stored as
    having cluster_id == 1).
    
    Note that for performance reasons, the peakgroups are created on-the-fly
    and not stored as objects but rather as tuples in "Peptide".

    Each peak group has a unique id, a score (fdr score usually), a retention
    time as well as a back-reference to the precursor that generated the
    peakgroup.
    In this case, the peak group can also be assigned a cluster id (where the
    cluster 1 is special as the one we will use for quantification).
    """

    cdef double fdr_score 
    cdef double normalized_retentiontime 
    cdef libcpp_string * internal_id_ 
    cdef double intensity_
    cdef double dscore_ 
    cdef int cluster_id_

    def __dealloc__(self):
         del self.internal_id_

    def __init__(self):
        pass

    def __richcmp__(self, other, op):
        """
        op_str = {0: '<', 1: '<=', 2: '==', 3: '!=', 4: '>', 5: '>='}[op]
        """
        if op == 4:
            # __lt__ = larger than
            return self.internal_id_ > other.internal_id_

    # Do not allow setting of any parameters (since data is not stored here)
    def set_fdr_score(self, fdr_score):
        raise Exception("Cannot set in immutable object")

    def set_normalized_retentiontime(self, normalized_retentiontime):
        raise Exception("Cannot set in immutable object")

    def set_feature_id(self, id_):
        raise Exception("Cannot set in immutable object")

    def set_intensity(self, intensity):
        raise Exception("Cannot set in immutable object")

    def get_dscore(self):
        return self.dscore_

    ## Select / De-select peakgroup
    def select_this_peakgroup(self):
        raise Exception("Not implemented!")
        ## self.peptide.select_pg(self.get_feature_id())

    ## Select / De-select peakgroup
    def setClusterID(self, id_):
        raise Exception("Not implemented!")
        ### self.cluster_id_ = id_
        ### self.peptide.setClusterID(self.get_feature_id(), id_)

    def get_cluster_id(self):
        return self.cluster_id_
  
    def __str__(self):
        return "PeakGroup %s at %s s in %s with score %s (cluster %s)" % (self.get_feature_id(), self.get_normalized_retentiontime(), "run?", self.get_fdr_score(), self.get_cluster_id())

    def get_value(self, value):
        raise Exception("Needs implementation")

    def set_value(self, key, value):
        raise Exception("Needs implementation")

    def set_fdr_score(self, fdr_score):
        self.fdr_score = fdr_score

    def get_fdr_score(self):
        return self.fdr_score

    def set_normalized_retentiontime(self, float normalized_retentiontime):
        self.normalized_retentiontime = normalized_retentiontime

    def get_normalized_retentiontime(self):
        return self.normalized_retentiontime

    def set_feature_id(self, id_):
        raise Exception("Not implemented!")
        # self.id_ = id_

    def get_feature_id(self):
        # return self.id_
        ## return "null"
        return <bytes>( deref(self.internal_id_) )

    def set_intensity(self, float intensity):
        self.intensity_ = intensity

    def get_intensity(self):
        return self.intensity_

    # Selected
    def is_selected(self):
        return self.cluster_id_ == 1

    def select_this_peakgroup(self):
        self.cluster_id_ = 1

    def get_cluster_id(self):
        return self.cluster_id_

cdef class CyMinimalPeakGroup:
    """
    A single peakgroup that is defined by a retention time in a chromatogram
    of multiple transitions. Additionally it has an fdr_score and it has an
    aligned RT (e.g. retention time in normalized space).
    A peakgroup can be selected for quantification or not (this is stored as
    having cluster_id == 1).
    
    Note that for performance reasons, the peakgroups are created on-the-fly
    and not stored as objects but rather as tuples in "Peptide".

    Each peak group has a unique id, a score (fdr score usually), a retention
    time as well as a back-reference to the precursor that generated the
    peakgroup.
    In this case, the peak group can also be assigned a cluster id (where the
    cluster 1 is special as the one we will use for quantification).
    """

    # __slots__ = ["fdr_score", "normalized_retentiontime", "id_", "intensity_", "cluster_id_"]
    cdef double fdr_score 
    cdef double normalized_retentiontime 
    cdef libcpp_string * internal_id_ 
    cdef double intensity_
    cdef double dscore_ 
    cdef int cluster_id_

    def __dealloc__(self):
         del self.internal_id_

    def __init__(self, bytes unique_id, float fdr_score, float assay_rt, selected, int cluster_id, peptide, intensity=-1.0, dscore=-1.0):
      # self.id_ = unique_id
      self.fdr_score = fdr_score
      self.normalized_retentiontime = assay_rt 
      self.cluster_id_ = cluster_id
      ## self.peptide = peptide  # TODO!!! 
      self.intensity_ = intensity
      self.dscore_ = dscore

      self.internal_id_ = new libcpp_string(<char*>unique_id)

    def __richcmp__(self, other, op):
        """
        op_str = {0: '<', 1: '<=', 2: '==', 3: '!=', 4: '>', 5: '>='}[op]
        """
        if op == 4:
            # __lt__ = larger than
            return self.internal_id_ > other.internal_id_

    # Do not allow setting of any parameters (since data is not stored here)
    def set_fdr_score(self, fdr_score):
        raise Exception("Cannot set in immutable object")

    def set_normalized_retentiontime(self, normalized_retentiontime):
        raise Exception("Cannot set in immutable object")

    def set_feature_id(self, id_):
        raise Exception("Cannot set in immutable object")

    def set_intensity(self, intensity):
        raise Exception("Cannot set in immutable object")

    def get_dscore(self):
        return self.dscore_

    ## Select / De-select peakgroup
    def select_this_peakgroup(self):
        raise Exception("Not implemented!")
        ## self.peptide.select_pg(self.get_feature_id())
        pass

    ## Select / De-select peakgroup
    def setClusterID(self, id_):
        raise Exception("Not implemented!")
        ### self.cluster_id_ = id_
        ### self.peptide.setClusterID(self.get_feature_id(), id_)

    def get_cluster_id(self):
        return self.cluster_id_
  
    def __str__(self):
        return "PeakGroup %s at %s s in %s with score %s (cluster %s)" % (self.get_feature_id(), self.get_normalized_retentiontime(), "run?", self.get_fdr_score(), self.get_cluster_id())

    def get_value(self, value):
        raise Exception("Needs implementation")

    def set_value(self, key, value):
        raise Exception("Needs implementation")

    def set_fdr_score(self, fdr_score):
        self.fdr_score = fdr_score

    def get_fdr_score(self):
        return self.fdr_score

    def set_normalized_retentiontime(self, float normalized_retentiontime):
        self.normalized_retentiontime = normalized_retentiontime

    def get_normalized_retentiontime(self):
        return self.normalized_retentiontime

    def set_feature_id(self, id_):
        raise Exception("Not implemented!")
        # self.id_ = id_

    def get_feature_id(self):
        # return self.id_
        ## return "null"
        return <bytes>( deref(self.internal_id_) )

    def set_intensity(self, float intensity):
        self.intensity_ = intensity

    def get_intensity(self):
        return self.intensity_

    # Selected
    def is_selected(self):
        return self.cluster_id_ == 1

    def select_this_peakgroup(self):
        self.cluster_id_ = 1

    def get_cluster_id(self):
        return self.cluster_id_

cdef class CyPrecursor:
    """ A set of peakgroups that belong to the same precursor in a single run.

    Each precursor has a backreference to its precursor group (heavy/light
    pair) it belongs to, the run it belongs to as well as its amino acid sequence.
    Furthermore, a unique id for the precursor and the protein name are stored.

    A precursor can return its best transition group, the selected peakgroup,
    or can return the transition group that is closest to a given iRT time.
    Its id is the transition_group_id (e.g. the id of the chromatogram)

    The "selected" peakgroup is represented by the peakgroup that belongs to
    cluster number 1 (cluster_id == 1) which in this case is "special".

    == Implementation details ==
    
    For memory reasons, we store all information about the peakgroup in a
    tuple (invariable). This tuple contains a unique feature id, a score and
    a retention time. Additionally, we also store, in which cluster the
    peakgroup belongs (if the user sets this).

    A peakgroup has the following attributes: 
        - an identifier that is unique among all other precursors 
        - a set of peakgroups 
        - a back-reference to the run it belongs to
    """

    # __slots__ = ["fdr_score", "normalized_retentiontime", "id_", "intensity_", "cluster_id_"]
    cdef bool _decoy
    # cdef int myid_ 
    cdef libcpp_vector[c_peakgroup] cpeakgroups_ 
    cdef libcpp_string curr_id_
    cdef libcpp_string protein_name_ 
    cdef libcpp_string sequence_ 
    cdef object run
    cdef object precursor_group

    def __init__(self, bytes this_id, run):
        cdef libcpp_string * s
        s = new libcpp_string(<char*> this_id)
        self.curr_id_ = deref(s)
        del s
        self.run = run
        ## self.peakgroups_ = []
        ## self.cluster_ids_ = []
        self._decoy = False
        ## self.protein_name = ""
        ## self.sequence = ""
        ## self.precursor_group = None
  
    def set_precursor_group(self, object p):
        self.precursor_group = p

    def set_decoy(self, bytes decoy):
        if decoy in ["FALSE", "False", "0"]:
            self._decoy = False
        elif decoy in ["TRUE", "True", "1"]:
            self._decoy = True
        else:
            raise Exception("Unknown decoy classifier '%s', please check your input data!" % decoy)

    def get_decoy(self):
        return self._decoy

    def setSequence(self, bytes p):
        cdef libcpp_string s
        s = libcpp_string(<char*> p)
        self.sequence_ = s

    def getSequence(self):
        return <bytes>( self.sequence_ )

    def getProteinName(self):
        return <bytes>( self.protein_name_ )

    def setProteinName(self, bytes p):
        cdef libcpp_string s
        s = libcpp_string(<char*> p)
        self.protein_name_ = s

    def __str__(self):
        return "%s (run %s)" % (self.get_id(), self.run)

    def add_peakgroup_tpl(self, pg_tuple, bytes tpl_id, int cluster_id=-1):
        """Adds a peakgroup to this precursor.

        The peakgroup should be a tuple of length 4 with the following components:
            0. id
            1. quality score (FDR)
            2. retention time (normalized)
            3. intensity
            (4. d_score optional)
        """
        # Check that the peak group is added to the correct precursor
        if self.get_id() != tpl_id:
            raise Exception("Cannot add a tuple to this precursor with a different id")

        if len(pg_tuple) == 4:
            pg_tuple = pg_tuple + (None,)

        assert len(pg_tuple) == 5
        cdef c_peakgroup pg
        cdef libcpp_string * s
        cdef libcpp_string q 
        q = libcpp_string(<char*> pg_tuple[0])
        s = new libcpp_string(<char*> pg_tuple[0])
        pg.fdr_score = pg_tuple[1]
        pg.normalized_retentiontime = pg_tuple[2]
        pg.intensity_ = pg_tuple[3]
        pg.dscore_ = pg_tuple[4]
        pg.cluster_id_ = cluster_id
        pg.internal_id_ = q
        self.cpeakgroups_.push_back(pg)
        del s


        ### self.peakgroups_.append(pg_tuple)
        ### self.cluster_ids_.append(cluster_id)

    def get_id(self):
        return <bytes>( self.curr_id_ )
  
    def getPrecursorGroup(self):
        return self.precursor_group 

    def getRun(self):
        return self.run

    def get_run_id(self):
      raise Exception("Not implemented")
      # return self.run.get_id()
    
    # 
    # Peakgroup cluster membership
    # 
    def select_pg(self, bytes this_id):
        self._setClusterID(this_id, 1)

    def unselect_pg(self, bytes this_id):
        self._setClusterID(this_id, -1)

    def setClusterID(self, bytes this_id, int cl_id):
        self._setClusterID(this_id, cl_id)

    cdef _setClusterID(self, bytes this_id, int cl_id):
        cdef libcpp_string s
        s = libcpp_string(<char*> this_id)
        cdef libcpp_vector[c_peakgroup].iterator it = self.cpeakgroups_.begin()
        cdef int nr_hit = 0
        while it != self.cpeakgroups_.end():
            if deref(it).internal_id_ == s:
                deref(it).cluster_id_ = cl_id
                nr_hit += 1
            inc(it)

        if nr_hit != 1:
              raise Exception("Error, found more than one peakgroup")

    def unselect_all(self):
        cdef libcpp_vector[c_peakgroup].iterator it = self.cpeakgroups_.begin()
        while it != self.cpeakgroups_.end():
            deref(it).cluster_id_ = -1
            inc(it)

    # 
    # Peakgroup selection
    # 
    def get_best_peakgroup(self):
        if self.cpeakgroups_.empty():
             return None

        cdef libcpp_vector[c_peakgroup].iterator it = self.cpeakgroups_.begin()
        cdef libcpp_vector[c_peakgroup].iterator best = self.cpeakgroups_.begin()
        cdef double best_score = deref(it).fdr_score
        while it != self.cpeakgroups_.end():
            if deref(it).fdr_score <= best_score:
                best_score = deref(it).fdr_score
                best = it
            inc(it)

        result = CyPeakgroupWrapperOnly()
        result.inst = address(deref(best))
        result.peptide = self
        return result

        ### if len(self.peakgroups_) == 0:
        ###     return None

        ### best_score = self.peakgroups_[0][1]
        ### result = self.peakgroups_[0]
        ### for peakgroup in self.peakgroups_:
        ###     if peakgroup[1] <= best_score:
        ###         best_score = peakgroup[1]
        ###         result = peakgroup
        ### index = [i for i,pg in enumerate(self.peakgroups_) if pg[0] == result[0]][0]
        ### return MinimalPeakGroup(result[0], result[1], result[2], self.cluster_ids_[index] == 1, self.cluster_ids_[index], self, result[3], result[4])

    def _fixSelectedPGError(self, fixMethod="Exception"):
      selected = [i for i,pg in enumerate(self.cluster_ids_) if pg == 1]
      if len(selected) > 1:
          print("Potential error detected in %s:\nWe have more than one selected peakgroup found. Starting error handling by using method '%s'." % (self, fixMethod))
          best_score = self.peakgroups_[0][1]
          best_pg = 0
          for s in selected:
              if best_score > self.peakgroups_[s][1]:
                  best_score = self.peakgroups_[s][1]
                  best_pg = s

          if fixMethod == "Exception":
              raise Exception("More than one selected peakgroup found in %s " % self )
          elif fixMethod == "BestScore":
              # Deselect all, then select the one with the best score...
              for s in selected:
                  self.cluster_ids_[s] = -1
              self.cluster_ids_[best_pg] = 1


    def get_selected_peakgroup(self):
        if self.cpeakgroups_.empty():
             return None

        cdef libcpp_vector[c_peakgroup].iterator it = self.cpeakgroups_.begin()
        cdef libcpp_vector[c_peakgroup].iterator best = self.cpeakgroups_.begin()
        cdef int nr_hit = 0
        while it != self.cpeakgroups_.end():
            if deref(it).cluster_id_ == 1:
                best = it
                nr_hit += 1
            inc(it)

        if nr_hit > 1:
              raise Exception("Error, found more than one peakgroup")
        if nr_hit == 0:
            return None

        result = CyPeakgroupWrapperOnly()
        result.inst = address(deref(best))
        result.peptide = self
        return result


      #### # return the selected peakgroup of this precursor, we can only select 1 or
      #### # zero groups per chromatogram!
      #### selected = [i for i,pg in enumerate(self.cluster_ids_) if pg == 1]
      #### assert len(selected) < 2
      #### if len(selected) == 1:
      ####   index = selected[0]
      ####   result = self.peakgroups_[index]
      ####   return MinimalPeakGroup(result[0], result[1], result[2], self.cluster_ids_[index] == 1, self.cluster_ids_[index], self, result[3], result[4])
      #### else: 
      ####     return None

    def getClusteredPeakgroups(self):
      ## selected = [i for i,pg in enumerate(self.cluster_ids_) if pg != -1]
      ## for index in selected:
      ##   result = self.peakgroups_[index]
      ##   yield MinimalPeakGroup(result[0], result[1], result[2], self.cluster_ids_[index] == 1, self.cluster_ids_[index], self, result[3], result[4])

        cdef libcpp_vector[c_peakgroup].iterator it = self.cpeakgroups_.begin()
        while it != self.cpeakgroups_.end():
            if (deref(it).cluster_id_ == -1): 
                inc(it)
                continue
            result = CyPeakgroupWrapperOnly()
            result.inst = address(deref(it))
            result.peptide = self
            yield result
            inc(it)

    def get_all_peakgroups(self):

        cdef libcpp_vector[c_peakgroup].iterator it = self.cpeakgroups_.begin()
        while it != self.cpeakgroups_.end():
            ### print ("I am iterating!! xx")
            ### print ( deref(it).fdr_score )
            ### # print ( <size_t>( deref(it).internal_id_ ) )
            ### print ( <bytes>(deref(it).internal_id_ ) )
            result = CyPeakgroupWrapperOnly()
            result.inst = address(deref(it))
            result.peptide = self
            yield result
            inc(it)


        ## for index, result in enumerate(self.peakgroups_):
        ##     yield MinimalPeakGroup(result[0], result[1], result[2], self.cluster_ids_[index] == 1, self.cluster_ids_[index], self, result[3], result[4])

    def getAllPeakgroups(self):
        return self.get_all_peakgroups()
  
    def find_closest_in_iRT(self, delta_assay_rt):
      result = min(self.peakgroups_, key=lambda x: abs(float(x[2]) - float(delta_assay_rt)))
      index = [i for i,pg in enumerate(self.peakgroups_) if pg[0] == result[0]][0]
      return MinimalPeakGroup(result[0], result[1], result[2], self.cluster_ids_[index] == 1, self.cluster_ids_[index], self, result[3], result[4])

@cython.boundscheck(False)
@cython.wraparound(False)
def test_optimized(size_t a, size_t b):
    cdef size_t q = 10
    cdef size_t result = 10 + a + b

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def static_cy_findBestPGFromTemplate(double expected_rt, target_peptide, double max_rt_diff,
        already_seen, double aligned_fdr_cutoff, double fdr_cutoff, correctRT_using_pg,
        verbose):
    """Find (best) matching peakgroup in "target" which matches to the source_rt RT.

        Parameters
        ----------
        expected_rt : float
            Expected retention time
        target_peptide: :class:`.PrecursorGroup`
            Precursor group from the target run (contains multiple peak groups)
        max_rt_diff : float
            Maximal retention time difference (parameter)
        already_seen : dict
            list of peakgroups already aligned (e.g. in a previous cluster) and which should be ignored
        aligned_fdr_cutoff : float
        fdr_cutoff : float
        correctRT_using_pg: boolean
        verbose: boolean

    """
    # Select matching peakgroups from the target run (within the user-defined maximal rt deviation)
    matching_peakgroups = [pg_ for pg_ in target_peptide.getAllPeakgroups() 
        if (abs(float(pg_.get_normalized_retentiontime()) - float(expected_rt)) < max_rt_diff) and
            pg_.get_fdr_score() < aligned_fdr_cutoff and 
            pg_.get_feature_id() + pg_.peptide.get_id() not in already_seen]

    cdef double rt
    cdef double fdr_score
    matching_peakgroups = []
    for pg_ in target_peptide.getAllPeakgroups() :
        rt = float(pg_.get_normalized_retentiontime())
        fdr_score = float(pg_.get_fdr_score())
        if fdr_score < aligned_fdr_cutoff:
            if (abs(rt - float(expected_rt)) < max_rt_diff):
                if pg_.get_feature_id() + pg_.peptide.get_id() not in already_seen:
                    matching_peakgroups.append(pg_)


    # If there are no peak groups present in the target run, we simply
    # return the expected retention time.
    if len(matching_peakgroups) == 0:
        return None, expected_rt

    # Select best scoring peakgroup among those in the matching RT window
    bestScoringPG = min(matching_peakgroups, key=lambda x: float(x.get_fdr_score()))

    # Printing for debug mode
    if verbose:
        closestPG = min(matching_peakgroups, key=lambda x: abs(float(x.get_normalized_retentiontime()) - expected_rt))
        print("    closest:", closestPG.print_out(), "diff", abs(closestPG.get_normalized_retentiontime() - expected_rt) )
        print("    bestScoring:", bestScoringPG.print_out(), "diff", abs(bestScoringPG.get_normalized_retentiontime() - expected_rt) )
        print()

    ### if len([pg_ for pg_ in matching_peakgroups if pg_.get_fdr_score() < self._aligned_fdr_cutoff]) > 1:
    ###     self.nr_multiple_align += 1
    ### if len([pg_ for pg_ in matching_peakgroups if pg_.get_fdr_score() < self._fdr_cutoff]) > 1:
    ###     self.nr_ambiguous += 1

    # Decide which retention time to return:
    #  - the threading one based on the alignment
    #  - the one of the best peakgroup
    if correctRT_using_pg:
        return bestScoringPG, bestScoringPG.get_normalized_retentiontime()
    else:
        return bestScoringPG, expected_rt
