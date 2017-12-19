#!/usr/bin/python
# -*- coding: utf-8  -*-
"""
=========================================================================
        msproteomicstools -- Mass Spectrometry Proteomics Tools
=========================================================================

Copyright (c) 2013, ETH Zurich
For a full list of authors, refer to the file AUTHORS.

This software is released under a three-clause BSD license:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of any author or any participating institution
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
--------------------------------------------------------------------------
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL ANY OF THE AUTHORS OR THE CONTRIBUTING
INSTITUTIONS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
--------------------------------------------------------------------------
$Maintainer: Hannes Roest$
$Authors: Hannes Roest$
--------------------------------------------------------------------------
"""

from msproteomicstoolslib.data_structures.PrecursorGroup import PrecursorGroup

kk = 0

prec_label_map = {} # id -> str
prec_label_map_rev = {} # str -> id

class Run():
    """
    A run contains references to identified precursor groups and precursors. 
    
    The run stores a reference to precursor groups (heavy/light pairs) identified in the run.
    It has a unique id and stores the headers from the csv

    A run has the following attributes: 
        - an identifier that is unique to this run
        - a filename where it originally came from
        - a dictionary of precursor groups which are accessible through the following functions
          - getPrecursorGroup 
          - hasPrecursor
          - getPrecursor
          - addPrecursor

    Parameters
    ----------
    header : str
        Run header
    header_dict : dict
        Run header dictionary
    runid : str
        Run header dictionary
    orig_input_filename : str
        Original filname of the csv file
    filename : str
        Original filname of the mzML (e.g. the column "filename")
    aligned_filename : str
        Aligned filename (e.g. the column "align_origfilename")
    """

    def __init__(self, header, header_dict, runid, orig_input_filename=None, filename=None, aligned_filename=None, useCython=False, experiment=None):
        self.header = header
        self.header_dict = header_dict
        self.runid = runid
        self.orig_filename = orig_input_filename # the original input filename
        self.openswath_filename = filename # the original OpenSWATH filename
        self.aligned_filename = aligned_filename # the aligned filename
        # self.all_precursor_groups_ = {}
        ### self.all_precursor_groups_ = []
        self.all_precursor_groups_int_ = {}
        self.k = []
        self.q = []
        self.di = {}
        self.experiment_ = experiment
  
        try:
            # print useCython
            if useCython:
                from msproteomicstoolslib.cython.data_structures import CyPrecursorGroup
                self.PrecursorGroup = CyPrecursorGroup
                # print self.PrecursorGroup
            else:
                self.PrecursorGroup = PrecursorGroup
        except ImportError:
            self.PrecursorGroup = PrecursorGroup

        from msproteomicstoolslib.cython.data_structures import CyPrecursorGroup
        # import asdasdf

    def __str__(self):
        return "Run %s" % (self.get_id())

    def get_id(self):
        return self.runid

    def get_openswath_filename(self):
        return self.openswath_filename

    def get_original_filename(self):
        return self.orig_filename

    def get_aligned_filename(self):
        return self.aligned_filename
  
    def get_best_peaks(self):
        """
        Return the best peakgroup for each peptide precursor
        """
        result = []
        for precursor_group in self.all_precursor_groups_int_.values():
            for peptide in precursor_group:
                result.append(peptide.get_best_peakgroup())
        return result
  
    def get_best_peaks_with_cutoff(self, cutoff):
        """
        Return the best peak per run (with cutoff)
        """
        return [p for p in self.get_best_peaks() if p.get_fdr_score() < cutoff]
  
    def getPrecursorGroup(self, peptide_group_label):
        if self.hasPrecursorGroup(peptide_group_label):
            return self.all_precursor_groups_int_[prec_label_map_rev[peptide_group_label]] 
        # this run has no peakgroup for that peptide
        return None

    def hasPrecursorGroup(self, peptide_group_label):
        return peptide_group_label in prec_label_map_rev and prec_label_map_rev[peptide_group_label] in self.all_precursor_groups_int_

    def hasPrecursor(self, peptide_group_label, trgr_id):
        """
        return peptide_group_label in self.all_precursor_groups_ and \
                -                not self.getPrecursorGroup(peptide_group_label).getPrecursor(trgr_id) is None

        """
        return self.hasPrecursorGroup(peptide_group_label) and \
                not self.getPrecursorGroup(peptide_group_label).getPrecursor(trgr_id) is None

        ## # return peptide_group_label in self.all_precursor_groups_ 
        ## r = peptide_group_label in self.all_precursor_groups_ 
        ## if r:
        ##     xx = self.getPrecursorGroup(peptide_group_label)
        ##     tmp = not xx.getPrecursor(trgr_id) is None
        ##     r = r and tmp
        ## return r

    def getPrecursor(self, peptide_group_label, trgr_id):
        """
        Return precursor corresponding to the given peptide label group and the transition group id
        """
        return self.experiment_.getPrecursor(peptide_group_label, trgr_id)

        ## if self.hasPrecursor(peptide_group_label, trgr_id):
        ##     return self.getPrecursorGroup(peptide_group_label).getPrecursor(trgr_id)
        ## else:
        ##     # print(" we dont have prec", peptide_group_label, trgr_id)
        ##     pass

        ## return None

    def addPrecursor(self, precursor, peptide_group_label):
        """
        Add a new precursor to the run using a specific peptide label.

        If the corresponding precursor group does not yet exist, a new
        precursor group is created. Otherwise the precursor is added to the
        precursor group.

        Parameters
        ----------
        precursor : :class:`.CyPrecursor`,  :class:`.Precursor` or :class:`.GeneralPrecursor`
            Precursor to be added (e.g. PEPT[+98]IDE/2)
        peptide_group_label : str
            Label of the corresponding peptide group (e.g. PEPTIDE)

        if peptide_group_label in self.all_precursor_groups_:
            self.getPrecursorGroup(peptide_group_label).addPrecursor(precursor)
        else:
            prec_gr = self.PrecursorGroup(peptide_group_label, self)
            prec_gr.addPrecursor(precursor) # 90 MB 
            self.all_precursor_groups_[peptide_group_label] = prec_gr # 140 MB

        328440maxresident)k          
        295032maxresident)k
        """

        self.experiment_.addPrecursor(precursor, peptide_group_label)
        return

        #### global kk
        #### global prec_label_map
        #### global prec_label_map_rev

        #### # if peptide_group_label in self.all_precursor_groups_:
        #### if self.hasPrecursorGroup(peptide_group_label):
        ####     self.getPrecursorGroup(peptide_group_label).addPrecursor(precursor)
        #### else:
        ####     prec_gr = self.PrecursorGroup(peptide_group_label, self)
        ####     prec_gr.addPrecursor(precursor) # 90 MB 

        ####     if not peptide_group_label in prec_label_map_rev:
        ####         cid = len (prec_label_map)
        ####         prec_label_map[cid] = peptide_group_label
        ####         prec_label_map_rev[peptide_group_label] = cid

        ####     cid = prec_label_map_rev[peptide_group_label]
        ####     self.all_precursor_groups_int_[ cid ] = prec_gr

        ####     # self.k.append(prec_gr)


        ####     ## print(self.PrecursorGroup)
        ####     ## print(prec_gr)
        ####     # print( len(prec_label_map), len(prec_label_map_rev) )
        ####     # (21261, 21261)
        ####     ### 295032maxresident)k

        ####     # self.k.append(prec_gr)

        ####     # print(" len gr", len(self.all_precursor_groups_) )
        ####     # print(self.all_precursor_groups_) 

        ####     ## down to 241872maxresident)k -> Python list to std::vec
        ####     ## 40 MB for string of peptide group labels --  207980maxresident)k  
        ####     ## 17 MB for ptr to run object




        ####     ## self.di[peptide_group_label] = 0 ## such a dict by itself consumes 30 MB
        ####     ## done here: 304644maxresident)k  


        ####     ## directly storing precursor with label in a list
        ####     # 191024maxresident)k 
        ####     #   removes 110 MB of overhead!
        ####     # self.all_precursor_groups_[peptide_group_label] = precursor

        ####     ## instead store prec_gr in list
        ####     # 250000maxresident)k 
        ####     ## save 50 MB of overhead
        ####     ## self.k.append(prec_gr)


        ####     ## not adding anything: 64716maxresident)k                

        ####     ## only storing the precursor: 190696maxresident) in hash => overhead of 60 MB
        ####     # self.all_precursor_groups_[peptide_group_label] = precursor
        ####     ## only storing the precursor in list: 136164maxresident)k == 70 MB just for the precursor (160 bytes)
        ####     # self.k.append(precursor) ## 70 MB
        ####     # self.q.append(peptide_group_label) ## 168796maxresident == 40 MB just for the strings (90 bytes)
        ####     # kk  = kk + 1
        ####     ## print("len q", len(self.q) )
        ####     ## print("len k", len(self.k) )
        ####     ## print("len kk", kk) ## 437284 = 440k 

    def __iter__(self):
        """
        Iterate through all precursor groups identified in this run.
        """
        for precursor_group in self.all_precursor_groups_int_.values():
            yield precursor_group

