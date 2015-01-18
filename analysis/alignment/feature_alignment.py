#!/usr/bin/env python
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

import os, sys, csv, time
import numpy
import argparse
from msproteomicstoolslib.math.chauvenet import chauvenet
import msproteomicstoolslib.math.Smoothing as smoothing
from msproteomicstoolslib.format.SWATHScoringReader import *
from msproteomicstoolslib.format.TransformationCollection import TransformationCollection, LightTransformationData
from msproteomicstoolslib.algorithms.alignment.Multipeptide import Multipeptide
from msproteomicstoolslib.algorithms.alignment.MRExperiment import MRExperiment
from msproteomicstoolslib.algorithms.alignment.AlignmentAlgorithm import AlignmentAlgorithm
from msproteomicstoolslib.algorithms.alignment.AlignmentMST import getDistanceMatrix, TreeConsensusAlignment
from msproteomicstoolslib.algorithms.alignment.AlignmentHelper import write_out_matrix_file, addDataToTrafo
from msproteomicstoolslib.algorithms.alignment.SplineAligner import SplineAligner
from msproteomicstoolslib.algorithms.alignment.FDRParameterEstimation import ParamEst
from msproteomicstoolslib.algorithms.PADS.MinimumSpanningTree import MinimumSpanningTree

class AlignmentStatistics():

    def __init__(self): 
        self.nr_aligned = 0
        self.nr_changed = 0
        self.nr_quantified = 0
        self.nr_removed = 0

        self.nr_good_peakgroups = 0
        self.nr_good_precursors = 0
        self.good_peptides = set([])
        self.good_proteins = set([])

        self.nr_quant_precursors = 0
        self.quant_peptides = set([])
        self.quant_proteins = set([])

    def count(astats, multipeptides, fdr_cutoff, skipDecoy=True):

        for m in multipeptides:

            if m.get_decoy() and skipDecoy:
                continue

            astats.nr_quantified += len(m.get_selected_peakgroups())

            # Count how many precursors / peptides / proteins fall below the threshold
            if m.find_best_peptide_pg().get_fdr_score() < fdr_cutoff:
                astats.nr_good_precursors += 1
                astats.good_peptides.update([m.getAllPeptides()[0].sequence])
                astats.good_proteins.update([m.getAllPeptides()[0].protein_name])

            # Count how many precursors / peptides / proteins were quantified
            if len(m.get_selected_peakgroups()) > 0:
                astats.nr_quant_precursors += 1
                astats.quant_peptides.update([m.getAllPeptides()[0].sequence])
                astats.quant_proteins.update([m.getAllPeptides()[0].protein_name])

            for p in m.getAllPeptides():

                # Count how many peakgroups simply fall below the threshold
                if p.get_best_peakgroup().get_fdr_score() < fdr_cutoff:
                    astats.nr_good_peakgroups += 1

                if p.get_selected_peakgroup() is not None:

                    # Number of peakgroups that are different from the original
                    if p.get_best_peakgroup().get_feature_id() != p.get_selected_peakgroup().get_feature_id() \
                       and p.get_selected_peakgroup().get_fdr_score() < fdr_cutoff:
                        astats.nr_changed += 1
                    # Number of peakgroups that were added
                    if p.get_best_peakgroup().get_fdr_score() > fdr_cutoff:
                        astats.nr_aligned += 1

                # Best peakgroup exists and is not selected
                elif p.get_best_peakgroup() is not None \
                  and p.get_best_peakgroup().get_fdr_score() < fdr_cutoff:
                    astats.nr_removed += 1

class Experiment(MRExperiment):
    """
    An Experiment is a container for multiple experimental runs - some of which may contain the same precursors.
    """

    def __init__(self):
        super(Experiment, self).__init__()
        self.transformation_collection = TransformationCollection()

    def estimate_real_fdr(self, multipeptides, fraction_needed_selected):
        class DecoyStats():
            def __init__(self):
                self.est_real_fdr = 0.0
                self.nr_decoys = 0
                self.nr_targets = 0
                self.decoy_pcnt = 0.0
                self.est_real_fdr = 0.0

        d = DecoyStats()
        precursors_to_be_used = [m for m in multipeptides if m.more_than_fraction_selected(fraction_needed_selected)]

        # count the decoys
        d.nr_decoys = sum([len(prec.get_selected_peakgroups()) for prec in precursors_to_be_used
                          if prec.find_best_peptide_pg().peptide.get_decoy()])
        d.nr_targets = sum([len(prec.get_selected_peakgroups()) for prec in precursors_to_be_used
                          if not prec.find_best_peptide_pg().peptide.get_decoy()])
        # estimate the real fdr by calculating the decoy ratio and dividing it
        # by the decoy ration obtained at @fdr_cutoff => which gives us the
        # decoy in/decrease realtive to fdr_cutoff. To calculate the absolute
        # value, we multiply by fdr_cutoff again (which was used to obtain the
        # original estimated decoy percentage).
        if self.estimated_decoy_pcnt is None: return d
        if (d.nr_targets + d.nr_decoys) == 0: return d
        d.decoy_pcnt = (d.nr_decoys * 100.0 / (d.nr_targets + d.nr_decoys) )
        d.est_real_fdr = d.decoy_pcnt / self.estimated_decoy_pcnt * self.initial_fdr_cutoff
        return d

    def print_stats(self, multipeptides, fdr_cutoff, fraction_present, min_nrruns):

        alignment = AlignmentStatistics()
        alignment.count(multipeptides, fdr_cutoff)

        # Count presence in all runs (before alignment)
        precursors_in_all_runs_wo_align = len([1 for m in multipeptides if m.all_above_cutoff(fdr_cutoff) and not m.get_decoy()])
        proteins_in_all_runs_wo_align_target = len(set([m.find_best_peptide_pg().peptide.protein_name for m in multipeptides 
                                                        if m.all_above_cutoff(fdr_cutoff) and 
                                                        not m.find_best_peptide_pg().peptide.get_decoy()]))
        peptides_in_all_runs_wo_align_target = len(set([m.find_best_peptide_pg().peptide.sequence for m in multipeptides 
                                                        if m.all_above_cutoff(fdr_cutoff) and 
                                                        not m.find_best_peptide_pg().peptide.get_decoy()]))

        # Count presence in all runs (before alignment)
        precursors_in_all_runs = [m for m in multipeptides if m.all_selected()]
        nr_peptides_target = len(set([prec.find_best_peptide_pg().peptide.sequence for prec in precursors_in_all_runs 
                                      if not prec.find_best_peptide_pg().peptide.get_decoy()]))
        nr_proteins_target = len(set([prec.find_best_peptide_pg().peptide.protein_name for prec in precursors_in_all_runs 
                                      if not prec.find_best_peptide_pg().peptide.get_decoy()]))

        nr_precursors_in_all = len([1 for m in multipeptides if m.all_selected() and not m.get_decoy()])
        max_pg = alignment.nr_good_precursors * len(self.runs)
        dstats = self.estimate_real_fdr(multipeptides, fraction_present)
        dstats_all = self.estimate_real_fdr(multipeptides, 1.0)

        # Get single/multiple hits stats
        from itertools import groupby
        precursors_quantified = [m for m in multipeptides if len(m.get_selected_peakgroups()) > 0]
        target_quant_protein_list = [ prec.find_best_peptide_pg().peptide.protein_name for prec in precursors_quantified 
                                     if not prec.find_best_peptide_pg().peptide.get_decoy()]
        target_quant_protein_list.sort()
        nr_sh_target_proteins = sum( [len(list(group)) == 1 for key, group in groupby(target_quant_protein_list)] )
        nr_mh_target_proteins = sum( [len(list(group)) > 1 for key, group in groupby(target_quant_protein_list)] )

        #
        ###########################################################################
        #
        print "="*75
        print "="*75
        print "Total we have", len(self.runs), "runs with", alignment.nr_good_precursors, \
                "peakgroups quantified in at least %s run(s) below m_score (q-value) %0.4f %%" % (min_nrruns, fdr_cutoff*100) + ", " + \
                "giving maximally nr peakgroups", max_pg
        print "We were able to quantify", alignment.nr_quantified, "/", max_pg, "peakgroups of which we aligned", \
                alignment.nr_aligned
        print "  The order of", alignment.nr_changed, "peakgroups was changed,", max_pg - alignment.nr_quantified, \
                "could not be aligned and %s were removed. Ambigous cases: %s, multiple suitable peakgroups: %s" % (
                    alignment.nr_removed, self.nr_ambiguous, self.nr_multiple_align)
        print "We were able to quantify %s / %s precursors in %s runs, and %s in all runs (up from %s before alignment)" % (
          alignment.nr_quant_precursors, alignment.nr_good_precursors, min_nrruns, nr_precursors_in_all, precursors_in_all_runs_wo_align)
        print "We were able to quantify %s / %s peptides in %s runs, and %s in all runs (up from %s before alignment)" % (
          len(alignment.quant_peptides), len(alignment.good_peptides), min_nrruns, nr_peptides_target, peptides_in_all_runs_wo_align_target)
        print "We were able to quantify %s / %s proteins in %s runs, and %s in all runs (up from %s before alignment)" % (
          len(alignment.quant_proteins), len(alignment.good_proteins), min_nrruns, nr_proteins_target, proteins_in_all_runs_wo_align_target)
        print "  Of these %s proteins, %s were multiple hits and %s were single hits." % (len(alignment.quant_proteins), nr_mh_target_proteins, nr_sh_target_proteins)

        # Get decoy estimates
        decoy_precursors = len([1 for m in multipeptides if len(m.get_selected_peakgroups()) > 0 and m.find_best_peptide_pg().peptide.get_decoy()])
        if len(precursors_in_all_runs) > 0:
            print "Decoy percentage of peakgroups that are fully aligned %0.4f %% (%s out of %s) which roughly corresponds to a peakgroup FDR of %s %%" % (
                dstats_all.decoy_pcnt, dstats_all.nr_decoys, dstats_all.nr_decoys + dstats_all.nr_targets, dstats_all.est_real_fdr*100)

            print "Decoy percentage of peakgroups that are partially aligned %0.4f %% (%s out of %s) which roughly corresponds to a peakgroup FDR of %s %%" % (
                dstats.decoy_pcnt, dstats.nr_decoys, dstats.nr_decoys + dstats.nr_targets, dstats.est_real_fdr*100)

            print "There were", decoy_precursors, "decoy precursors identified out of", \
                    alignment.nr_quant_precursors + decoy_precursors, "precursors which is %0.4f %%" % (
                        decoy_precursors *100.0 / (alignment.nr_quant_precursors + decoy_precursors))

    def _getTrafoFilename(self, current_run, ref_id):
        current_id = current_run.get_id()
        input_basename = os.path.basename(current_run.orig_filename)
        fn = os.path.splitext(input_basename)[0]
        dirname = os.path.dirname(current_run.orig_filename)
        filename = os.path.join(dirname, "%s-%s-%s.tr" % (fn, current_id, ref_id) )
        return filename

    def _write_trafo_files(self):
        # Print out trafo data
        trafo_fnames = []
        for current_run in self.runs:
            current_id = current_run.get_id()
            ref_id = self.transformation_collection.getReferenceRunID()
            filename = self._getTrafoFilename(current_run, ref_id)
            trafo_fnames.append(filename)
            self.transformation_collection.writeTransformationData(filename, current_id, ref_id)
            self.transformation_collection.readTransformationData(filename)

    def write_to_file(self, multipeptides, options, writeTrafoFiles=True):

        infiles = options.infiles
        outfile = options.outfile
        matrix_outfile = options.matrix_outfile
        yaml_outfile = options.yaml_outfile
        ids_outfile = options.ids_outfile
        fraction_needed_selected = options.min_frac_selected
        file_format = options.file_format

        # 1. Collect ids of selected features
        selected_pgs = []
        for m in multipeptides:

            selected_peakgroups = m.get_selected_peakgroups()
            if (len(selected_peakgroups)*1.0 / len(self.runs)) < fraction_needed_selected: 
                continue

            for p in m.getAllPeptides():
                selected_pg = p.get_selected_peakgroup()
                clustered_pg = p.getClusteredPeakgroups()
                for pg in clustered_pg:
                    selected_pgs.append(pg)

        selected_ids_dict = dict( [ (pg.get_feature_id(), pg) for pg in selected_pgs] )

        # 2. Write out the (selected) ids
        if len(ids_outfile) > 0:
            fh = open(ids_outfile, "w")
            id_writer = csv.writer(fh, delimiter="\t")
            for pg in selected_pgs:
                id_writer.writerow([pg.get_feature_id()])
            fh.close()
            del id_writer

        # 3. Write out the matrix outfile
        if len(matrix_outfile) > 0:
            write_out_matrix_file(matrix_outfile, self.runs, multipeptides,
                                  fraction_needed_selected,
                                  style=options.matrix_output_method,
                                  aligner_mscore_treshold=options.fdr_cutoff)

        # 4. Write out the full outfile
        if len(outfile) > 0 and options.readmethod == "full":
            # write out the complete original files 
            writer = csv.writer(open(outfile, "w"), delimiter="\t")
            header_first = self.runs[0].header
            for run in self.runs:
                assert header_first == run.header
            header_first += ["align_runid", "align_origfilename"]
            writer.writerow(header_first)

            for m in multipeptides:

                selected_peakgroups = m.get_selected_peakgroups()
                if (len(selected_peakgroups)*1.0 / len(self.runs)) < fraction_needed_selected:
                    continue

                for p in m.get_peptides():
                    selected_pg = p.get_selected_peakgroup()
                    if selected_pg is None: 
                        continue

                    row_to_write = selected_pg.row
                    row_to_write += [selected_pg.run.get_id(), selected_pg.run.orig_filename]
                    # Replace run_id with the aligned id (align_runid) ->
                    # otherwise the run_id is not guaranteed to be unique 
                    row_to_write[ header_dict["run_id"]] = selected_ids_dict[f_id].peptide.run.get_id()
                    writer.writerow(row_to_write)

        elif len(outfile) > 0 and file_format in ["openswath", "peakview_preprocess"]:

            name_of_id_col_map = { "openswath" : "id" , "peakview_preprocess" : "preprocess_id"}
            name_of_trgr_col_map = { "openswath" : "transition_group_id" , "peakview_preprocess" : "Pep Index"}
            name_of_id_col = name_of_id_col_map[file_format]
            name_of_trgr_col = name_of_trgr_col_map[file_format]

            # Only in openswath we have the ID and can go back to the original file.
            # We can write out the complete original files.

            writer = csv.writer(open(outfile, "w"), delimiter="\t")
            header_first = self.runs[0].header
            for run in self.runs:
                assert header_first == run.header
            header_first += ["align_runid", "align_origfilename", "align_clusterid"]
            writer.writerow(header_first)

            for file_nr, f in enumerate(infiles):
              header_dict = {}
              if f.endswith('.gz'):
                  import gzip
                  filehandler = gzip.open(f,'rb')
              else:
                  filehandler = open(f)

              reader = csv.reader(filehandler, delimiter="\t")
              header = reader.next()
              for i,n in enumerate(header):
                header_dict[n] = i

              for row in reader:
                  f_id = row[ header_dict[name_of_id_col]]
                  if selected_ids_dict.has_key(f_id):
                      # Check the "id" and "transition_group_id" field.
                      # Unfortunately the id can be non-unique, there we check both.
                      trgroup_id = selected_ids_dict[f_id].peptide.get_id()
                      unique_peptide_id = row[ header_dict[name_of_trgr_col]]
                      if unique_peptide_id == trgroup_id:
                          row_to_write = row
                          row_to_write += [selected_ids_dict[f_id].peptide.run.get_id(), f, selected_ids_dict[f_id].get_cluster_id()]
                          # Replace run_id with the aligned id (align_runid) ->
                          # otherwise the run_id is not guaranteed to be unique 
                          if file_format == "openswath" : 
                              row_to_write[ header_dict["run_id"]] = selected_ids_dict[f_id].peptide.run.get_id()
                          writer.writerow(row_to_write)

        # 5. Write out the .tr transformation files
        if writeTrafoFiles:
            self._write_trafo_files()

        # 6. Write out the YAML file
        if len(yaml_outfile) > 0:
            import yaml
            myYaml = {"Commandline" : sys.argv, 
                      "RawData" : [], "PeakGroupData" : [ outfile ],
                      "ReferenceRun" : self.transformation_collection.getReferenceRunID(), 
                      "FeatureAlignment" : 
                      {
                        "RawInputParameters" : options.__dict__,
                        "Parameters" : {}
                      },
                      "Parameters" : {}
                     }
            myYaml["Parameters"]["m_score_cutoff"] = float(options.fdr_cutoff) # deprecated
            myYaml["FeatureAlignment"]["Parameters"]["m_score_cutoff"] = float(options.fdr_cutoff)
            myYaml["FeatureAlignment"]["Parameters"]["fdr_cutoff"] = float(options.fdr_cutoff)
            myYaml["FeatureAlignment"]["Parameters"]["aligned_fdr_cutoff"] = float(options.aligned_fdr_cutoff)
            for current_run in self.runs:
                current_id = current_run.get_id()
                ref_id = self.transformation_collection.getReferenceRunID()
                filename = self._getTrafoFilename(current_run, ref_id)
                dirpath = os.path.dirname(current_run.orig_filename)
                ### Use real path (not very useful when moving data from one computer to another)
                ### filename = os.path.realpath(filename)
                ### dirpath = os.path.realpath(dirpath)
                this = {"id" : current_id, "directory" : dirpath, "trafo_file" : filename}
                myYaml["RawData"].append(this)
            open(yaml_outfile, 'w').write(yaml.dump({"AlignedSwathRuns" : myYaml}))

def estimate_aligned_fdr_cutoff(options, this_exp, multipeptides, fdr_range):
    print "Try to find parameters for target fdr %0.2f %%" % (options.target_fdr * 100)

    for aligned_fdr_cutoff in fdr_range:
        # Do the alignment and annotate chromatograms without identified features 
        # Then perform an outlier detection over multiple runs
        # unselect all
        for m in multipeptides:
            for p in m.get_peptides():
                p.unselect_all()

        # now align
        options.aligned_fdr_cutoff = aligned_fdr_cutoff
        alignment = align_features(multipeptides, options.rt_diff_cutoff, options.fdr_cutoff, options.aligned_fdr_cutoff, options.method)
        est_fdr = this_exp.estimate_real_fdr(multipeptides, options.min_frac_selected).est_real_fdr

        print "Estimated FDR: %0.4f %%" % (est_fdr * 100), "at position aligned fdr cutoff ", aligned_fdr_cutoff
        if est_fdr > options.target_fdr:
            # Unselect the peptides again ...
            for m in multipeptides:
                for p in m.get_peptides():
                    p.unselect_all()
            return aligned_fdr_cutoff

def doBayes_collect_pg_data(mpep, h0, run_likelihood, x, min_rt, max_rt, bins, peak_sd):
    """
    Bayesian alignment step 1:
        - collect the h0 data and the peakgroup data for all peakgroups
    """

    import numpy as np
    import scipy.stats

    # Compute bin width (dt)
    dt = abs(max_rt - min_rt) / bins

    # Collect peakgroup data across runs
    for p in mpep.getAllPeptides(): # loop over runs
        # print "Collect pg for run ", p.run.get_id()
        current_best_pg = p.get_best_peakgroup()
        gaussians = []
        y = np.zeros_like(x)
        ##  print x, y
        # sum_gaussians 
        for pg in p.getAllPeakgroups():
            h0_tmp = float(pg.get_value("h0_score"))
            weight = float(pg.get_value("h_score"))
            gaussians.append( scipy.stats.norm(loc = pg.get_normalized_retentiontime() , scale = peak_sd ))
            y = y + dt * weight * scipy.stats.norm.pdf(x, loc = pg.get_normalized_retentiontime() , scale = peak_sd )

        if False:
            print x, y
            print sum(y)
            print sum(y) + h0_tmp
            print abs(max_rt - min_rt) * 0.2
            print dt

        f_D_r_t = y # f_{D_r}(t) posterior pdf for each run
        run_likelihood[p.run.get_id()] = y
        h0[p.run.get_id()] = h0_tmp
        # print " == Selected peakgroup ", current_best_pg.print_out()

def doBayes_collect_product_data(mpep, tr_data, m, j, h0, run_likelihood, x, peak_sd, bins,
                                ptransfer, transfer_width, verb=False):
    """
    Bayesian computation of the contribution of all other runs to the probability

    Loops over all runs r to compute the probabilities, for each run:
        - (i) RT transfer from source to target r
        - (ii) Compute  p(D_r|B_{jm} ) = \sum_{q=1}^{k} p(D_r | B_{qr} ) * p(B_{qr}|B_{jm})
        - (iii) Compute  transition probability p(B_{qr}|B_{jm} )

    For step (iii), there are different options available how to compute the
    transition probability p(B_{qr}|B_{jm}), see ptransfer option:
        - all: the best bin gets all the probability
        - equal: all bins around the best bin get equal probability
        - gaussian: probability is distributed according to a gaussian

    """

    import scipy.stats

    dt = (max(x) - min(x)) / len(x)
    equal_bins = int(transfer_width / dt) + 1

    prod_acc = 1.0
    # \prod
    # r = 1 \ m to n
    for rloop in mpep.getAllPeptides(): # loop over runs
        r = rloop.run.get_id()
        if r == m:
            continue
        f_D_r = run_likelihood[r]

        # (i) transform the retention time from the source run (m) to the one
        #     of the target run (r) and find the matching bin in run r
        source = m
        target = r
        expected_rt = tr_data.getTrafo(source, target).predict( [ x[j] ] )[0]
        matchbin = int((expected_rt - min(x)) / dt )

        # If verbose
        if verb:
            print "convert from", source, " to ", target
            print "predict for ", x[j]
            print "results in ", expected_rt
            print x[matchbin]
            print min(x)
            print max(x)
            print len(x)
            print "best bin", int((expected_rt - min(x)) / dt )

        # (ii) Compute p(D_r|B_{jm} = \sum_{q=1}^{k} p(D_r | B_{qr} ) * p(B_{qr}|B_{jm}
        #      This is a sum over all bins of the target run r
        p_Dr_Bjm = 0 # p(D_r|B_{jm})
        # \sum 
        # q = 1 to k
        for q in xrange(bins):

            # (iii) Compute transition probability between runs, e.g.
            #       p(B_{qr}|B_{jm} which is the probability of the analyte
            #       being in bin q (of run r) given that the analyte is
            #       actually in bin j (of run m): p_Bqr_Bjm
            #       Initially set to zero
            p_Bqr_Bjm = 0
            if ptransfer == "all":
                if q == matchbin:
                    p_Bqr_Bjm = 1
            elif ptransfer == "equal":
                if abs(q - matchbin) < equal_bins:
                    p_Bqr_Bjm = 0.5 / equal_bins
            elif ptransfer == "bartlett":
                if abs(q - matchbin) < equal_bins:
                    # height of the triangle
                    height = 1.0 / equal_bins
                    # height of normalized window
                    dy = (1.0 * equal_bins - abs(q - matchbin) ) / equal_bins
                    p_Bqr_Bjm = dy * height
            elif ptransfer == "gaussian":
                p_Bqr_Bjm = scipy.stats.norm.pdf(x[q], loc = expected_rt , scale = transfer_width)

            # (iv) multiply f_{D_r}(t_q) with the transition probability

            if verb:
                print "Got here for bin %s a value %s * %s = %s"  %(q, f_D_r[q], p_Bqr_Bjm, f_D_r[q] * p_Bqr_Bjm)
            p_Dr_Bjm += f_D_r[q] * p_Bqr_Bjm

        p_absent = h0[r]
        p_present = 1-h0[r]
        #p_present = 1.0
        #p_absent = 0.0
        # use correct formula from last page
        prod_acc *= p_present * p_Dr_Bjm + p_absent / bins 
        if verb:
            print "all sum", p_Dr_Bjm
            print "h0 here", h0[r]
            print " === add for bin", p_present * p_Dr_Bjm + p_absent / bins 

    return prod_acc

def doPlotStuff(mpep, x, run_likelihood, B_m, m, p_D_no_m, max_prior, max_post):
    """
    Helper function to plot stuff 
    """
    ## print "sum", sum(B_m)
    ## print "sum prior", sum(run_likelihood[m])
    ## print "sum over all other runs", sum(p_D_no_m)
    if False:
        print "B_{%s} forall j" % (m), B_m
        print "(B_{%s} |D) forall j normalized" % (m), B_m
        print "(B_{%s} |D_m) forall j normalized" % (m), run_likelihood[m]
        print "MAP before at ", x[max_prior]
        print "MAP now at ", x[max_post]
        print "  --> ", x[max_post] - 0.5*dt , " to ", x[max_post] + 0.5*dt
        ###
    # Plot ? 
    import pylab
    pepid = mpep.getAllPeptides()[0].get_id()

    pylab.plot(x, run_likelihood[m])
    pylab.savefig('prior_%s.pdf' % m )
    pylab.clf()

    pylab.plot(x, B_m)
    print '%s' % m
    pylab.savefig('post_%s.pdf' % m )
    pylab.clf()

    pylab.plot(x, p_D_no_m, label="likelihood (other runs)")
    pylab.savefig('likelihood_%s.pdf' % m )
    pylab.clf()

    pylab.plot(x, run_likelihood[m], label="prior")
    pylab.plot(x, B_m, label="posterior")
    pylab.plot(x, p_D_no_m, label="likelihood (other runs)")
    #pylab.legend(loc= "upper left")
    pylab.legend(loc= "upper right")
    pylab.title(pepid)
    pylab.xlabel("RT")
    pylab.savefig('both_%s.pdf' % m )
    pylab.clf()


def doBayesianAlignment(exp, multipeptides, max_rt_diff, initial_alignment_cutoff,
                        smoothing_method, doPlot=True, outfile="out"):
    """
    Bayesian alignment
    """
    
    import scipy.stats
    import numpy as np

    print "open outfile", outfile
    fh = open(outfile, "w")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Set parameters
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    ptransfer = "all"
    ptransfer = "equal" # boxcar / rectangle
    ptransfer = "bartlett" #triangular
    ptransfer = "gaussian" # gaussian window

    peak_sd = 15 # 30 seconds peak (2 stdev 95 \% of all signal)
    peak_sd = 10 # 30 seconds peak (3 stdev 99.7 \% of all signal)
    # peak_sd = 7.5
    # peak_sd = 5
    ## equal_bins_mult = 2.0 # two seems reasonable
    ## #equal_bins_mult = 4.0 # two seems reasonable
    ## gaussian_scale = 2.5
    ## gaussian_scale = 3.0
    ## # equal_bins_mult = 2.5 # two seems reasonable
    #equal_bins_mult = 0.25
    # peak_sd = 15 # 30 seconds peak (2 stdev 95 \% of all signal)

    # Increase uncertainty by a factor of 2.5 when transferring probabilities
    # from one run to another
    transfer_width = peak_sd * 2.5

    # Number of bins to obtain reasonable resolution (should be higher than the
    # above gaussian widths).  On a 600 second chromatogram, 100 bins lead to a
    # resolution of ca. 6 seconds.
    bins = 100

    # How much should the RT window extend beyond the peak area (in %) to
    # ensure for smooth peaks when computing gaussians at the end of the
    # chromatogram
    rt_window_ext = 0.2

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Step 1 : Get alignments (all against all)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    start = time.time()
    spl_aligner = SplineAligner(initial_alignment_cutoff)
    tr_data = LightTransformationData()
    for r1 in exp.runs:
        for r2 in exp.runs:
            startx = time.time()
            addDataToTrafo(tr_data, r1, r2,
                           spl_aligner, multipeptides, smoothing_method,
                           max_rt_diff, sd_max_data_length=300)
            print("Add trafo to data took %0.2fs" % (time.time() - startx) )

    print("Compute pairwise alignments took %0.2fs" % (time.time() - start) )
    start = time.time()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Step 2 : Iterate through all peptides
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    for pepcnt,mpep in enumerate(multipeptides):

        pepid = mpep.getAllPeptides()[0].get_id()

        # if False:
        #     for p in mpep.getAllPeptides(): # loop over runs
        #         pgg = [float(pg.get_value("h_score")) for pg in p.getAllPeakgroups()]
        #         pgg.sort(reverse=True)
        #         if len(pgg) > 1:
        #             if pgg[0] / pgg[1] < 10 and float(pg.get_value("h0_score")) < 0.3:
        #                 print "  -> Candiate match", pepcnt, pepid
        #                 #print "  -> Candiate match", pepcnt, pepid, pgg[0] / pgg[1], float(pg.get_value("h0_score")), pgg 

        #     continue

        # for p in mpep.getAllPeptides(): # loop over runs
        #     pgg = [float(pg.get_value("h_score")) for pg in p.getAllPeakgroups()]
        #     pgg.sort(reverse=True)
        #     if len(pgg) > 1:
        #         if pgg[0] / pgg[1] < 10 and float(pg.get_value("h0_score")) < 0.3:
        #             print "  -> Candiate match", pepcnt, pepid, pgg[0] / pgg[1], float(pg.get_value("h0_score")), pgg 


        print "00000000000000000000000000000000000 new peptide (bayes)", mpep.getAllPeptides()[0].get_id(), pepcnt

        # Step 2.1 : Compute the retention time space (min / max)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        rts = [pg.get_normalized_retentiontime()
                for p in mpep.getAllPeptides()
                    for pg in p.getAllPeakgroups() ]

        min_rt = min(rts)
        max_rt = max(rts)
        min_rt -= abs(max_rt - min_rt) * rt_window_ext
        max_rt += abs(max_rt - min_rt) * rt_window_ext

        # Step 2.2 : Collect peakgroup data across runs
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        h0 = {}
        run_likelihood = {}
        x = np.linspace(min_rt, max_rt, bins)
        doBayes_collect_pg_data(mpep, h0, run_likelihood, x, min_rt, max_rt, bins, peak_sd)

        # Step 2.3 : Loop over all runs for this peptide 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        for p in mpep.getAllPeptides():
            m = p.run.get_id()

            # Step 2.3.1 : obtain likelihood f_{D_m}(t) for current run m and prior p(B_{jm})
            f_D_m = run_likelihood[ p.run.get_id() ] # f_{D_m}(t) likelihood pdf for run m
            p_B_jm = 1.0/bins # prior p(B_{jm})

            # Step 2.3.2 : compute product over all runs (obtain likelihood
            #              p(D_r | B_{jm}) for all bins j over all runs r in
            #              the data (except run m).
            #              Store p(D | B_{jm}) in vector B_m for all values of j
            B_m = []
            p_D_no_m = []
            for j in xrange(bins):

                tmp_prod = doBayes_collect_product_data(mpep, tr_data, m, j, h0, run_likelihood, x, peak_sd, bins, ptransfer, transfer_width)

                p_D_no_m.append(tmp_prod)
                B_jm = f_D_m[j] * p_B_jm * tmp_prod # f_{D_m}(t_j) * p(B{jm}) * ... 

                if False:
                    print "tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt"
                    print "Computed bin %s at RT %s" % (j, x[j])
                    print 
                    print "Compute B_jm =  %s * %s * %s = %s " % (f_D_m[j], p_B_jm, tmp_prod, B_jm)
                    print "Got for bin %s a value of %s, will record a result of %s for B_jm" % (j, tmp_prod, B_jm)

                ### TODO
                # correction for h_0 hypothesis according to (35), right before chapter E
                # may be omitted for computational reasons since it does not change the result
                # -> everything gets normalized afterwards anywys ... 
                ### B_jm *= 1-h0[m]

                B_m.append(B_jm)

            # Step 2.3.3 : Compute p(B_{jm} | D) using Bayes formula from the
            #              values p(D| B_{jm}), p(B_{jm}) and p(D). p(D) is
            #              computed by the sum over the array B_m (prior p_B_jm
            #              is already added above).
            B_m /= sum(B_m)
            ### TODO correct here for H0 ? 
            B_m *= 1 - h0[m]

            # Step 2.3.4 : Compute maximal posterior and plot data
            #              
            print "MAP (B_m|D)", max([ [xx,i] for i,xx in enumerate(B_m)])
            print "MAP (B_m|D_m)", max([ [xx,i] for i,xx in enumerate(run_likelihood[m])])
            max_prior = max([ [xx,i] for i,xx in enumerate(run_likelihood[m])])[1]
            max_post = max([ [xx,i] for i,xx in enumerate(B_m)])[1]

            if doPlot:
                p_D_no_m /= sum(p_D_no_m) # for plotting purposes
                p_D_no_m *= 1-h0[m] # for plotting purposes
                # print "to plot p_D_no_m", p_D_no_m
                # print "sum prior", sum(run_likelihood[m])
                # print "sum post", sum(B_m)
                doPlotStuff(mpep, x, run_likelihood, B_m, m, p_D_no_m, max_prior, max_post)

            # Compute bin width (dt)
            dt = abs(max_rt - min_rt) / bins

            # Step 2.3.5 : Select best peakgroup
            #              
            for pg in p.getAllPeakgroups():
                left = float(pg.get_value("leftWidth"))
                right = float(pg.get_value("rightWidth"))
                tmp = [(xx,yy) for xx,yy in zip(x,B_m) if left-0.5*dt < xx and right+0.5*dt > xx]
                pg.set_value("norm_RT", sum([xx[1] for xx in tmp]))
                print "   *", pg, " ", pg.get_value("pg_score"), " / ", pg.get_value("h_score"), " / h0 ", pg.get_value("h0_score")

            # TODO how to transfer this ... 
            # select by maximum probability sum
            best_psum = max([(pg.get_value("norm_RT"), pg) for pg in p.getAllPeakgroups()])
            print "best peak", best_psum[1], "with sum", best_psum[0]
            best_psum[1].select_this_peakgroup()
            fh.write("%s\t%s\n" % (best_psum[1].get_value("id"), best_psum[0]) )
            print "write to fh", "%s\t%s" % (best_psum[1].get_value("id"), best_psum[0]) 
                
        print "peptide (bayes)", mpep.getAllPeptides()[0].get_id()

    fh.close()
    print("Bayesian alignment took %0.2fs" % (time.time() - start) )

def doMSTAlignment(exp, multipeptides, max_rt_diff, rt_diff_isotope, initial_alignment_cutoff,
                   fdr_cutoff, aligned_fdr_cutoff, smoothing_method, method,
                   use_RT_correction, stdev_max_rt_per_run, use_local_stdev):
    """
    Minimum Spanning Tree (MST) based local aligment 
    """

    spl_aligner = SplineAligner(initial_alignment_cutoff)
    tree = MinimumSpanningTree(getDistanceMatrix(exp, multipeptides, spl_aligner))
    print "Computed Tree:", tree
    
    # Get alignments
    tr_data = LightTransformationData()
    for edge in tree:
        addDataToTrafo(tr_data, exp.runs[edge[0]], exp.runs[edge[1]],
                       spl_aligner, multipeptides, smoothing_method,
                       max_rt_diff)

    tree_mapped = [ (exp.runs[a].get_id(), exp.runs[b].get_id()) for a,b in tree]

    # Perform work
    al = TreeConsensusAlignment(max_rt_diff, fdr_cutoff, aligned_fdr_cutoff, 
                                rt_diff_isotope=rt_diff_isotope,
                                correctRT_using_pg=use_RT_correction,
                                stdev_max_rt_per_run=stdev_max_rt_per_run,
                                use_local_stdev=use_local_stdev)

    if method == "LocalMST":
        al.alignBestCluster(multipeptides, tree_mapped, tr_data)
    elif method == "LocalMSTAllCluster":
        al.alignAllCluster(multipeptides, tree_mapped, tr_data)

    # Store number of ambigous cases (e.g. where more than one peakgroup below
    # the strict quality cutoff was found in the RT window) and the number of
    # cases where multiple possibilities were found.
    exp.nr_ambiguous = al.nr_ambiguous
    exp.nr_multiple_align = al.nr_multiple_align

def doParameterEstimation(options, this_exp, multipeptides):
    """
    Perform (q-value) parameter estimation
    """

    start = time.time()
    print "-"*35
    print "Do Parameter estimation"
    p = ParamEst(min_runs=options.nr_high_conf_exp,verbose=True)
    decoy_frac = p.compute_decoy_frac(multipeptides, options.target_fdr)
    print "Found target decoy fraction overall %0.4f%%" % (decoy_frac*100)
    try:
        fdr_cutoff_calculated = p.find_iterate_fdr(multipeptides, decoy_frac)
    except UnboundLocalError:
        raise Exception("Could not estimate FDR accurately!")

    if fdr_cutoff_calculated > options.target_fdr:
        # Re-read the multipeptides with the new cutoff (since the cutoff might
        # be higher than before, we might have to consider more peptides now).
        multipeptides = this_exp.get_all_multipeptides(fdr_cutoff_calculated, verbose=True)
        print "Re-parse the files!"
        try:
            fdr_cutoff_calculated = p.find_iterate_fdr(multipeptides, decoy_frac)
        except UnboundLocalError:
            raise Exception("Could not estimate FDR accurately!")

    options.aligned_fdr_cutoff = float(options.aligned_fdr_cutoff)
    if options.aligned_fdr_cutoff < 0:
        # Estimate the aligned_fdr parameter -> if the new fdr cutoff is
        # lower than the target fdr, we can use the target fdr as aligned
        # cutoff but if its higher we have to guess (here we take
        # 2xcutoff).
        if fdr_cutoff_calculated < options.target_fdr:
            options.aligned_fdr_cutoff = options.target_fdr
        else:
            options.aligned_fdr_cutoff = 2*fdr_cutoff_calculated

    options.fdr_cutoff = fdr_cutoff_calculated
    print "Using an m_score (q-value) cutoff of %0.7f%%" % (fdr_cutoff_calculated*100)
    print "For the aligned values, use a cutoff of %0.7f%%" % (options.aligned_fdr_cutoff*100)
    print("Parameter estimation took %0.2fs" % (time.time() - start) )
    print "-"*35
    return multipeptides

def doReferenceAlignment(options, this_exp, multipeptides):

    # Performing re-alignment using a reference run
    if options.realign_method != "diRT":
        start = time.time()
        spl_aligner = SplineAligner(alignment_fdr_threshold = options.alignment_score, 
                                   smoother=options.realign_method,
                                   external_r_tmpdir = options.tmpdir)
        this_exp.transformation_collection = spl_aligner.rt_align_all_runs(this_exp, multipeptides)
        trafoError = spl_aligner.getTransformationError()
        print("Aligning the runs took %0.2fs" % (time.time() - start) )

    try:
        options.aligned_fdr_cutoff = float(options.aligned_fdr_cutoff)
    except ValueError:
        # We have a range of values to step through. 
        # Since we trust the input, wo dont do error checking.
        exec("fdr_range = numpy.arange(%s)" % options.aligned_fdr_cutoff)
        options.aligned_fdr_cutoff = estimate_aligned_fdr_cutoff(options, this_exp, multipeptides, fdr_range)

    try:
        options.rt_diff_cutoff = float(options.rt_diff_cutoff)
    except ValueError:
        if options.rt_diff_cutoff == "auto_2medianstdev":
            options.rt_diff_cutoff = 2*numpy.median(list(trafoError.getStdev()))
        elif options.rt_diff_cutoff == "auto_3medianstdev":
            options.rt_diff_cutoff = 3*numpy.median(list(trafoError.getStdev()))
        elif options.rt_diff_cutoff == "auto_4medianstdev":
            options.rt_diff_cutoff = 4*numpy.median(list(trafoError.getStdev()))
        elif options.rt_diff_cutoff == "auto_maxstdev":
            options.rt_diff_cutoff = max(list(trafoError.getStdev()))
        else:
            raise Exception("max_rt_diff either needs to be a value in seconds or" + \
                            "one of ('auto_2medianstdev', 'auto_3medianstdev', " + \
                            "'auto_4medianstdev', 'auto_maxstdev'). Found instead: '%s'" % options.rt_diff_cutoff)

    print "Will calculate with aligned_fdr cutoff of", options.aligned_fdr_cutoff, "and an RT difference of", options.rt_diff_cutoff
    start = time.time()
    AlignmentAlgorithm().align_features(multipeptides, 
                    options.rt_diff_cutoff, options.fdr_cutoff,
                    options.aligned_fdr_cutoff, options.method)
    print("Re-aligning peak groups took %0.2fs" % (time.time() - start) )

def handle_args():
    usage = "" #usage: %prog --in \"files1 file2 file3 ...\" [options]" 
    usage += "\nThis program will select all peakgroups below the FDR cutoff in all files and try to align them to each other."
    usage += "\nIf only one file is given, it will act as peakgroup selector (best by m_score)" + \
            "\nand will apply the provided FDR cutoff."

    import ast
    parser = argparse.ArgumentParser(description = usage )
    parser.add_argument('--in', dest="infiles", required=True, nargs = '+', help = 'A list of mProphet output files containing all peakgroups (use quotes around the filenames)')
    parser.add_argument('--file_format', default='openswath', help="Which input file format is used (openswath or peakview)")
    parser.add_argument("--out", dest="outfile", required=True, default="feature_alignment_outfile", help="Output file with filtered peakgroups for quantification (only works for OpenSWATH)")
    parser.add_argument("--out_matrix", dest="matrix_outfile", default="", help="Matrix containing one peak group per row (supports .csv, .tsv or .xlsx)")
    parser.add_argument("--out_ids", dest="ids_outfile", default="", help="Id file only containing the ids")
    parser.add_argument("--out_meta", dest="yaml_outfile", default="", help="Outfile containing meta information, e.g. mapping of runs to original directories")
    parser.add_argument("--fdr_cutoff", dest="fdr_cutoff", default=0.01, type=float, help="Seeding score cutoff", metavar='0.01')
    parser.add_argument("--max_fdr_quality", dest="aligned_fdr_cutoff", default=-1.0, help="Extension score cutoff - during the extension phase of the algorithm, peakgroups of this quality will still be considered for alignment (in FDR) - it is possible to give a range in the format lower,higher+stepsize,stepsize - e.g. 0,0.31,0.01 (-1 will set it to fdr_cutoff)", metavar='-1')
    parser.add_argument("--max_rt_diff", dest="rt_diff_cutoff", default=30, help="Maximal difference in RT for two aligned features", metavar='30')
    parser.add_argument("--iso_max_rt_diff", dest="rt_diff_isotope", default=10, help="Maximal difference in RT for two isotopic channels in the same run", metavar='30')
    parser.add_argument("--frac_selected", dest="min_frac_selected", default=0.0, type=float, help="Do not write peakgroup if selected in less than this fraction of runs (range 0 to 1)", metavar='0')
    parser.add_argument('--method', default='best_overall', help="Which method to use for the clustering (best_overall, best_cluster_score or global_best_cluster_score, global_best_overall, LocalMST, LocalMSTAllCluster). Note that the MST options will perform a local, MST guided alignment while the other options will use a reference-guided alignment. The global option will also move peaks which are below the selected FDR threshold.")
    parser.add_argument("--verbosity", default=0, type=int, help="Verbosity (0 = little)", metavar='0')
    parser.add_argument("--matrix_output_method", dest="matrix_output_method", default='none', help="Which columns are written besides Intensity (none, RT, score, source or full)")
    parser.add_argument('--realign_method', dest='realign_method', default="diRT", help="How to re-align runs in retention time ('diRT': use only deltaiRT from the input file, 'linear': perform a linear regression using best peakgroups, 'splineR': perform a spline fit using R, 'splineR_external': perform a spline fit using R (start an R process using the command line, 'splinePy' use Python native spline from scikits.datasmooth (slow!), 'lowess': use Robust locally weighted regression (lowess smoother), 'nonCVSpline, CVSpline': splines with and without cross-validation, 'earth' : use Multivariate Adaptive Regression Splines using py-earth")

    experimental_parser = parser.add_argument_group('experimental options')

    experimental_parser.add_argument('--disable_isotopic_grouping', action='store_true', default=False, help="Disable grouping of isotopic variants by peptide_group_label, thus disabling matching of isotopic variants of the same peptide across channels. If turned off, each isotopic channel will be matched independently of the other. If enabled, the more certain identification will be used to infer the location of the peak in the other channel.")
    experimental_parser.add_argument('--use_dscore_filter', action='store_true', default=False)
    experimental_parser.add_argument("--dscore_cutoff", default=1.96, type=float, help="Quality cutoff to still consider a feature for alignment using the d_score: everything below this d-score is discarded", metavar='1.96')
    experimental_parser.add_argument("--nr_high_conf_exp", default=1, type=int, help="Number of experiments in which the peptide needs to be identified with high confidence (e.g. above fdr_curoff)", metavar='1')
    experimental_parser.add_argument("--readmethod", dest="readmethod", default="minimal", help="Read full or minimal transition groups (minimal,full)", metavar="minimal")
    experimental_parser.add_argument("--tmpdir", dest="tmpdir", default="/tmp/", help="Temporary directory")
    experimental_parser.add_argument("--alignment_score", dest="alignment_score", default=0.0001, type=float, help="Minimal score needed for a feature to be considered for alignment between runs", metavar='0.0001')
    experimental_parser.add_argument("--mst:useRTCorrection", dest="mst_correct_rt", type=ast.literal_eval, default=False, help="Use aligned peakgroup RT to continue threading in MST algorithm", metavar='False')
    experimental_parser.add_argument("--mst:Stdev_multiplier", dest="mst_stdev_max_per_run", type=float, default=-1.0, help="How many standard deviations the peakgroup can deviate in RT during the alignment (if less than max_rt_diff, then max_rt_diff is used)", metavar='-1.0')
    experimental_parser.add_argument("--mst:useLocalStdev", dest="mst_local_stdev", type=ast.literal_eval, default=False, help="Use standard deviation of local region of the chromatogram", metavar='False')
    experimental_parser.add_argument("--target_fdr", dest="target_fdr", default=-1, type=float, help="If parameter estimation is used, which target FDR should be optimized for. If set to lower than 0, parameter estimation is turned off.", metavar='0.01')

    # deprecated methods
    experimental_parser.add_argument('--realign_runs', action='store_true', default=False, help="Deprecated option (equals '--realign_method external_r')")
    experimental_parser.add_argument('--use_external_r', action='store_true', default=False, help="Deprecated option (equals '--realign_method external_r')")

    args = parser.parse_args(sys.argv[1:])

    # deprecated
    if args.realign_runs or args.use_external_r:
        print "WARNING, deprecated --realign_runs or --use_external_r used! Please use --realign_method instead"
        args.realign_method = "splineR_external"

    if args.min_frac_selected < 0.0 or args.min_frac_selected > 1.0:
        raise Exception("Argument frac_selected needs to be a number between 0 and 1.0")

    if args.target_fdr > 0:
        # Parameter estimation turned on: check user input ...
        if args.fdr_cutoff != 0.01:
            raise Exception("You selected parameter estimation with target_fdr - cannot set fdr_cutoff as well! It does not make sense to ask for estimation of the fdr_cutoff (target_fdr > 0.0) and at the same time specify a certain fdr_cutoff.")
        args.fdr_cutoff = args.target_fdr
        # if args.aligned_fdr_cutoff != -1.0:
        #     raise Exception("You selected parameter estimation with target_fdr - cannot set max_fdr_quality as well!")
        pass
    else:
        # Parameter estimation turned off: Check max fdr quality ...
        try:
            if float(args.aligned_fdr_cutoff) < 0:
                args.aligned_fdr_cutoff = args.fdr_cutoff
                print("Setting max_fdr_quality automatically to fdr_cutoff of", args.fdr_cutoff)
            elif float(args.aligned_fdr_cutoff) < args.fdr_cutoff:
                raise Exception("max_fdr_quality cannot be smaller than fdr_cutoff!")
        except ValueError:
            pass
    return args

def main(options):

    class DReadFilter(object):
        def __init__(self, cutoff):
            self.cutoff = cutoff
        def __call__(self, row, header):
            return float(row[ header["d_score" ] ]) > self.cutoff


    readfilter = ReadFilter()
    if options.use_dscore_filter:
        readfilter = DReadFilter(float(options.dscore_cutoff))

    # Read the files
    start = time.time()
    reader = SWATHScoringReader.newReader(options.infiles, options.file_format,
                                          options.readmethod, readfilter,
                                          enable_isotopic_grouping = not options.disable_isotopic_grouping)
    runs = reader.parse_files(options.realign_method != "diRT", options.verbosity)

    # Create experiment
    this_exp = Experiment()
    this_exp.set_runs(runs)
    print("Reading the input files took %0.2fs" % (time.time() - start) )

    # Map the precursors across multiple runs, determine the number of
    # precursors in all runs without alignment.
    start = time.time()
    multipeptides = this_exp.get_all_multipeptides(options.fdr_cutoff, verbose=False, verbosity=options.verbosity)
    print("Mapping the precursors took %0.2fs" % (time.time() - start) )

    if options.target_fdr > 0:
        multipeptides = doParameterEstimation(options, this_exp, multipeptides)

    if options.method == "Bayesian":
        stdev_max_rt_per_run = None
        start = time.time()
        doBayesianAlignment(this_exp, multipeptides, float(options.rt_diff_cutoff), 
                       float(options.alignment_score), 
                       options.realign_method, doPlot=True, 
                       outfile=options.ids_outfile + "extra") 
        print("Re-aligning peak groups took %0.2fs" % (time.time() - start) )
    elif options.method == "LocalMST" or options.method == "LocalMSTAllCluster":
        start = time.time()
        if options.mst_stdev_max_per_run > 0:
            stdev_max_rt_per_run = options.mst_stdev_max_per_run
        else:
            stdev_max_rt_per_run = None
            
        doMSTAlignment(this_exp, multipeptides, float(options.rt_diff_cutoff), 
                       float(options.rt_diff_isotope),
                       float(options.alignment_score), options.fdr_cutoff,
                       float(options.aligned_fdr_cutoff),
                       options.realign_method, options.method,
                       options.mst_correct_rt, stdev_max_rt_per_run,
                       options.mst_local_stdev)
        print("Re-aligning peak groups took %0.2fs" % (time.time() - start) )
    else:
        doReferenceAlignment(options, this_exp, multipeptides)


    # Filter by high confidence (e.g. keep only those where enough high confidence IDs are present)
    for mpep in multipeptides:
        # check if we have found enough peakgroups which are below the cutoff
        count = 0
        for pg in mpep.get_selected_peakgroups():
            if pg.get_fdr_score() < options.fdr_cutoff:
                count += 1
        if count < options.nr_high_conf_exp:
            for p in mpep.getAllPeptides():
                p.unselect_all()

    # print statistics, write output
    start = time.time()
    this_exp.print_stats(multipeptides, options.fdr_cutoff, options.min_frac_selected, options.nr_high_conf_exp)
    this_exp.write_to_file(multipeptides, options)
    print("Writing output took %0.2fs" % (time.time() - start) )

if __name__=="__main__":
    options = handle_args()
    main(options)

