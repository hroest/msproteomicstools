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

import numpy
import numpy as np
import math
import random
import scipy.stats
import time
from msproteomicstoolslib.algorithms.alignment.Multipeptide import Multipeptide
from msproteomicstoolslib.algorithms.alignment.SplineAligner import SplineAligner
from msproteomicstoolslib.format.TransformationCollection import TransformationCollection, LightTransformationData
from msproteomicstoolslib.algorithms.alignment.AlignmentHelper import addDataToTrafo
from msproteomicstoolslib.algorithms.PADS.MinimumSpanningTree import MinimumSpanningTree
from msproteomicstoolslib.algorithms.alignment.AlignmentMST import getDistanceMatrix, TreeConsensusAlignment
import msproteomicstoolslib.optimized as optimized

def doBayes_collect_pg_data(mpep, h0, run_likelihood, x, min_rt, max_rt, bins, peak_sd):
    """
    Bayesian alignment step 1:
        - collect the h0 data and the peakgroup data for all peakgroups

    For each run, iterate through all peakgroups present and determine the
    gaussian shape (noise distribution) around each peakgroup
    """


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
                                ptransfer, transfer_width, stdev_max_rt_per_run, verb=False):
    """
    Bayesian computation of the contribution of all other runs to the probability

    Loops over all runs r to compute the probabilities, for each run:
        - (i)   RT transfer from source to target r
        - (ii)  Compute p(D_r|B_{jm} ) = \sum_{q=1}^{k} p(D_r | B_{qr} ) * p(B_{qr}|B_{jm})
        - (iii) Compute transition probability p(B_{qr}|B_{jm} )

    For step (iii), there are different options available how to compute the
    transition probability p(B_{qr}|B_{jm}), see ptransfer option:
        - all: the best bin gets all the probability
        - equal: all bins around the best bin get equal probability
        - gaussian: probability is distributed according to a gaussian

    This step usually takes the longest, with the bottleneck either in getTrafo
    or scipy.stats.norm.pdf
    """

    dt = (max(x) - min(x)) / len(x)
    equal_bins = int(transfer_width / dt) + 1
    ## print "equal bins", equal_bins

    local_transfer_width = transfer_width

    verb = True
    verb = False
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
        # TODO: TIME : this fxn call can be rather slow, 13% - 80% of the time (depending on alignment method)
        expected_rt = tr_data.getTrafo(source, target).predict( [ x[j] ] )[0]
        matchbin = int((expected_rt - min(x)) / dt )

        if stdev_max_rt_per_run > 0:
            local_transfer_width = max(transfer_width, stdev_max_rt_per_run * tr_data.getStdev(source, target) )
            equal_bins = int(local_transfer_width / dt) + 1

        # If verbose
        if verb:
            print "---------------------------------------------------------------------------"
            print "use method", ptransfer
            print "convert from", source, " to ", target
            print "predict for", x[j]
            print "results in", expected_rt
            print "transfer_width", transfer_width
            print "stdev per run:", tr_data.getStdev(source, target)
            print "stdev per run: -> mult", stdev_max_rt_per_run * tr_data.getStdev(source, target)
            print x[matchbin]
            print "best bin", int((expected_rt - min(x)) / dt )
            print "eq bins", equal_bins

        # (ii) Compute p(D_r|B_{jm} = \sum_{q=1}^{k} p(D_r | B_{qr} ) * p(B_{qr}|B_{jm}
        #      This is a sum over all bins of the target run r
        p_Dr_Bjm = 0 # p(D_r|B_{jm})
        # \sum 
        # q = 1 to k
        # TODO: TIME : loop can be rather slow (for bartlett even), ca 60% of all time
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
                ### p_Bqr_Bjm = scipy.stats.norm.pdf(x[q], loc = expected_rt , scale = transfer_width)
                p_Bqr_Bjm = optimized.norm_pdf(x[q], expected_rt , local_transfer_width)

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
            print "convert from", source, " to ", target
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
                        smoothing_method, doPlot=True, outfile=None, transfer_fxn="bartlett",nr_bins=100, peak_sd=10):
    """
    Bayesian alignment

    ptransfer = "all"
    ptransfer = "equal" # boxcar / rectangle
    ptransfer = "bartlett" #triangular
    ptransfer = "gaussian" # gaussian window
    ptransfer = "bartlett" #triangular

    peak_sd = 15 # 30 seconds peak (2 stdev 95 \% of all signal)
    peak_sd = 10 # 30 seconds peak (3 stdev 99.7 \% of all signal)
    """

    doPlot = False
    verbose = False
    highlyVerbose = False

    stdev_max_rt_per_run = 2.5
    
    fh = None
    if outfile is not None:
        fh = open(outfile, "w")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Set parameters
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # Set transfer function
    ptransfer = transfer_fxn 

    # Only select any peak in the chromatogram if the chance that any peak is
    # present is better than this probability.
    #  -> To get ROC or precision-recall plots we need to output all of them
    # h0_cutoff = 0.5
    h0_cutoff = 1.0

    # Denotes the width of the peak in RT-domain each peak 
    # peak_sd

    # Increase uncertainty by a factor of when transferring probabilities from
    # one run to another
    transfer_width = peak_sd * stdev_max_rt_per_run 

    # Number of bins to obtain reasonable resolution (should be higher than the
    # above gaussian widths).  On a 600 second chromatogram, 100 bins lead to a
    # resolution of ca. 6 seconds.
    bins = nr_bins

    # How much should the RT window extend beyond the peak area (in %) to
    # ensure for smooth peaks when computing gaussians at the end of the
    # chromatogram
    rt_window_ext = 0.2

    # Use a per-alignment transfer width (punishing bad runs, rewarding good alignments)
    # 
    # stdev_max_rt_per_run = 4.0 ## this works even for linear that is 50 seconds off (with gauss fxn)
    # stdev_max_rt_per_run = 2.5 ## still works, but is cutting it close ... 
    stdev_max_rt_per_run = -1 

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Step 1 : Get alignments (all against all)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    start = time.time()
    spl_aligner = SplineAligner(initial_alignment_cutoff)
    tr_data = LightTransformationData()
    for r1 in exp.runs:
        for r2 in exp.runs:
            addDataToTrafo(tr_data, r1, r2,
                           spl_aligner, multipeptides, smoothing_method,
                           max_rt_diff, sd_max_data_length=300)

    print("Compute pairwise alignments took %0.2fs" % (time.time() - start) )
    start = time.time()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Step 2 : Iterate through all peptides
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    print "Will do %s peptides" % len(multipeptides)
    for pepcnt,mpep in enumerate(multipeptides):

        if verbose:
            print "00000000000000000000000000000000000 new peptide (bayes)", mpep.getAllPeptides()[0].get_id(), pepcnt, len(multipeptides)

        # Step 2.1 : Compute the retention time space (min / max)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        rts = [pg.get_normalized_retentiontime()
                for p in mpep.getAllPeptides()
                    for pg in p.getAllPeakgroups() ]

        min_rt = min(rts)
        max_rt = max(rts)
        min_rt -= abs(max_rt - min_rt) * rt_window_ext
        max_rt += abs(max_rt - min_rt) * rt_window_ext

        # Hack to ensure that the two are never equal
        if min_rt == max_rt:
            min_rt -= peak_sd
            max_rt += peak_sd

        # Step 2.2 : Collect peakgroup data across runs
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        h0 = {}
        run_likelihood = {}
        x = np.linspace(min_rt, max_rt, bins)
        doBayes_collect_pg_data(mpep, h0, run_likelihood, x, min_rt, max_rt, bins, peak_sd)

        # Step 2.3 : Loop over all runs for this peptide 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        for p in mpep.getAllPeptides():

            if verbose:
                print "Do run", p.run.get_id()

            m = p.run.get_id() # current_run id

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

                tmp_prod = optimized.doBayes_collect_product_data(mpep, tr_data, m, j, h0, 
                                run_likelihood, x, peak_sd, bins, ptransfer, transfer_width, stdev_max_rt_per_run)

                p_D_no_m.append(tmp_prod)
                B_jm = f_D_m[j] * p_B_jm * tmp_prod # f_{D_m}(t_j) * p(B{jm}) * ... 

                if False:
                    print "tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt"
                    print "Computed bin %s at RT %s" % (j, x[j])
                    print "Compute B_jm =  %s * %s * %s = %s " % (f_D_m[j], p_B_jm, tmp_prod, B_jm)
                    print "Got for bin %s a value of %s, will record a result of %s for B_jm" % (j, tmp_prod, B_jm)
                    print 

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

            # print "MAP (B_m|D)", max([ [xx,i] for i,xx in enumerate(B_m)])
            # print "MAP (B_m|D_m)", max([ [xx,i] for i,xx in enumerate(run_likelihood[m])])
            max_prior = max([ [xx,i] for i,xx in enumerate(run_likelihood[m])])[1]
            max_post = max([ [xx,i] for i,xx in enumerate(B_m)])[1]

            if doPlot:
                # TODO if zero ...
                p_D_no_m /= sum(p_D_no_m) # for plotting purposes
                p_D_no_m *= 1-h0[m] # for plotting purposes
                doPlotStuff(mpep, x, run_likelihood, B_m, m, p_D_no_m, max_prior, max_post)

            # Compute bin width (dt)
            dt = abs(max_rt - min_rt) / bins

            # Step 2.3.5 : Select best peakgroup
            #              
            for pg in p.getAllPeakgroups():
                left = float(pg.get_value("leftWidth"))
                right = float(pg.get_value("rightWidth"))
                tmp = [(xx,yy) for xx,yy in zip(x,B_m) if left-0.5*dt < xx and right+0.5*dt > xx]
                pg.add_value("accum_p", sum([xx[1] for xx in tmp]))
                if highlyVerbose:
                    print "Got pg", pg, "with value", sum([xx[1] for xx in tmp])

            # select the peak with the maximum probability weight
            best_psum = max([(pg.get_value("accum_p"), pg) for pg in p.getAllPeakgroups()])
            # print "best peak", best_psum[1], "with sum", best_psum[0]
            best_pg = best_psum[1]
            if float(best_pg.get_value("h0_score")) < h0_cutoff: 
                best_pg.select_this_peakgroup()
                if fh is not None:
                    fh.write("%s\t%s\n" % (best_psum[1].get_value("id"), best_psum[0]) )
                
        if verbose:
            print "peptide (bayes)", mpep.getAllPeptides()[0].get_id()

    if fh is not None:
        fh.close()

    print("Bayesian alignment took %0.2fs" % (time.time() - start) )


def getPG(mpep, run, pg):

    if not mpep.hasPrecursorGroup(run):
        if pg == 0:
            # Just a missing one, we only want to have p(H0) which in this
            # case should be 1.0 as we do not have a peakgroup -> 100%
            # chance of missing value
            return None
        else:
            # Bug!
            raise Exception("Bug, requested pg %s from a run that has no peakgroups" % pg)

    prgr = mpep.getPrecursorGroup(run)

    if len( prgr.getAllPrecursors() ) > 1:
        raise Exception("Not implemented for precursor groups...")
    return prgr.getAllPrecursors()[0].getAllPeakgroups()[pg-1]
    # return ( list(list(mpep.getAllPeptides())[ run ].getAllPeakgroups())[pg - 1] )
    

def getScoreAndRT(mpep, run_id, pg, mypghash=None):
    if mypghash is None:
        mypg = getPG(mpep, run_id, pg)
        rt = float(mypg.get_value("RT"))
        current_score = float(mypg.get_value("h_score"))
    else:
        current_score = mypghash[run_id][pg][0]
        score_h0 = mypghash[run_id][pg][1]
        rt = mypghash[run_id][pg][2]

    return (current_score, rt)

def getH0Score(mpep, run_id, mypghash=None):
    if mypghash is None:
        # We have selected H0 here, thus append the h0 score
        mypg = getPG(mpep, run_id, 0)

        # Only return h0 score if we have a pg in this run (otherwise p=1 and we add zero in log-space)
        if mypg is not None:
            return float(mypg.get_value("h0_score"))
        else:
            return 1.0
    else:
        return mypghash[run_id][0]

def evalvec(tree_path, selection_vector_new, tree_start, mpep, tr_data, transfer_width = 30, verbose=False, mypghash=None):

    # starting point
    tree_start
    pg = selection_vector_new[ tree_start ]

    # keep track of all positions of peakgroups from previous runs
    rt_positions = {}

    # keep track of whether something was selected or not
    selected_pg = {}
    H0PENALTY= 1.0

    # Do everything for the first peakgroup !
    current_score = 0.0
    if pg != 0:
        pg_score, rt = getScoreAndRT(mpep, tree_start, pg, mypghash)
        rt_positions[ tree_start ] = rt
        selected_pg[ tree_start ] = pg_score
        current_score += math.log(pg_score)
    else:
        # We have selected H0 here, thus append the h0 score
        current_score += math.log( getH0Score(mpep, tree_start, mypghash) )
        selected_pg[ tree_start ] = -1

    if verbose:
        print "start with score %s at point %s" % (current_score, tree_start) 

    for e in tree_path:
        ####
        #### Compute p(e[1]|e[0])
        ####
        prev_run = e[0]
        curr_run = e[1]
        pg = selection_vector_new[ curr_run ]
        if verbose:
            print "compute p(%s|%s)" % (e[1], e[0]) 

        # get the retention time position in the previous run
        rt_prev = rt_positions.get( prev_run, None)

        if pg != 0:
            pg_score, rt_curr = getScoreAndRT(mpep, curr_run, pg, mypghash)
            current_score += math.log(pg_score)

            selected_pg[ curr_run ] = pg_score

            #
            # p( e[1] ) -> we need to compute p( e[1] | e[0] ) 
            #
            #  -> get the probability that we have a peak in the current run
            #  e[1] eluting at position rt_curr given that there is a peak in
            #  run e[0] eluting at position rt_prev
            #

            if verbose:
                print "Append score for adding peakgroup %s: %s" % (rt_curr, math.log(pg_score))

            if rt_prev is not None:

                source = prev_run
                target = curr_run
                # Try to get fast version of trafo first, then try slow one
                try:
                    mytrafo = tr_data.getTrafo(source, target)
                    expected_rt = mytrafo.internal_interpolation.predictSingleValue(rt_prev)
                except AttributeError:
                    expected_rt = tr_data.getTrafo(source, target).predict(
                        [ float(rt_prev) ] )[0]

                # 
                # Tr to compute p(curr_run | prev_run )
                # 
                #  - compute normalized RT diff (mean, std normalized)
                #  - use fast_score_update to compute probability, then add log(p) to the score
                ###  # The code is equivalent to
                ###  cdf_v = optimized.norm_cdf(norm_rt_diff)
                ###  if (cdf_v > 0.5):
                ###      cdf_v = 1-cdf_v
                ###  
                ###  if cdf_v > 0.0:
                ###      return (current_score + c_log(cdf_v))
                ###      current_score += math.log(cdf_v)
                ###  else:
                ###      current_score += -99999999999999999999999
                norm_rt_diff = (expected_rt-rt_curr) / transfer_width
                current_score = optimized.fast_score_update(norm_rt_diff, current_score)
                if verbose:
                    print "Append score for RT diff from expected (%s): %s" % (norm_rt_diff, math.log( optimized.norm_cdf(norm_rt_diff) ))

            else:
                # no previous run, this means all previous runs were
                # empty and we cannot add any RT so far
                pass

            # store our current position 
            rt_positions[curr_run] = rt_curr

        else:

            selected_pg[ curr_run ] = -1

            # We have selected H0 here, thus append the h0 score
            current_score += math.log( getH0Score(mpep, curr_run, mypghash) / H0PENALTY )
            if verbose:
                print "Selected score H0, append: %s" % (math.log( getH0Score(mpep, curr_run, mypghash) ))
                print "Get prev run al: %s" % (selected_pg[ prev_run])

            # store our current position 
            if rt_prev is not None:

                source = prev_run
                target = curr_run

                # Try to get fast version of trafo first, then try slow one
                try:
                    mytrafo = tr_data.getTrafo(source, target)
                    expected_rt = mytrafo.internal_interpolation.predictSingleValue(rt_prev)
                except AttributeError:
                    expected_rt = tr_data.getTrafo(source, target).predict(
                        [ float(rt_prev) ] )[0]


                # We did not currently select a peakgroup, so this
                # would basically mean that our assumption 
                #
                # p(curr_run| 1,2,..., prev_run, ... n) = p(curr_run|prev_run) 
                #
                # is not really valid any more, we would have to
                # backtrack and figure out what the conditional
                # independence assumpations were for prev_run and thus we would have
                # 
                # p(curr_run| 1,2,..., prev_run, prev_prev_run, ... n) = p(curr_run|prev_run, prev_prev_run) 
                #
                # where prev_prev_run was the run that was successor to prev_run.
                # Currently we dont do this, we simply make a shortcut here
                # We store the expected RT as the current one -> it is not
                # the best idea but it still may do the job
                rt_positions[curr_run] = expected_rt

        if verbose:
            print "  -> have now score %s" % (current_score) 

    # print "use py version", selection_vector_new.values(), current_score
    return current_score

def mcmcrun(nrit, selection_vector, tree_path, tree_start, pg_per_run, mpep,
            tr_data, n_runs, transfer_width, f=1.0, verbose=False, biasSelection=False, usecpp=True):
        """
        f = 2.5 # seems reasonable but still pretty wild
        # 1 seems too tame
        # 5 seems very wild

        """

        # Create hash which can be accessed as:
        #   h[run][pg][0] = score
        #   h[run][pg][1] = h0 score
        #   h[run][pg][2] = rt
        mypghash = {}
        for k,v in pg_per_run.iteritems():
            curr_run = k
            run_hash = mypghash.get(curr_run, {})

            # Default value for null hypothesis (no peakgroup is true) is 100%
            # and the correct value if there are no peakgroups at all. If we
            # have some, we update below
            h0_score = 1.0
            for pg_ in range(v):
                # We count our peakgroups starting with 1
                # 0 means null hypothesis
                pg = pg_ + 1 
                mypg = getPG(mpep, curr_run, pg )
                score = float(mypg.get_value("h_score"))
                h0_score = float(mypg.get_value("h0_score"))
                rt = float(mypg.get_value("RT"))

                pg_hash = run_hash.get(pg, [])
                pg_hash.append(score)
                pg_hash.append(h0_score)
                pg_hash.append(rt)
                run_hash[pg] = pg_hash

            # Set null hypothesis probability
            run_hash[0] = h0_score

            mypghash[curr_run] = run_hash

        burn_in_time = 0
        time_in_best_config = 0

        if usecpp:
            prev_score = optimized.c_evalvec(tree_path, selection_vector, tree_start, mpep, tr_data, transfer_width, False, mypghash)
        else:
            prev_score = evalvec(            tree_path, selection_vector, tree_start, mpep, tr_data, transfer_width=transfer_width, mypghash=mypghash)
        best_score = prev_score
        best_config = selection_vector
        BIAS_PEAKGROUPS=1

        if verbose:
            print "start: ", prev_score, selection_vector

        for i in range(nrit):
            # print "88888888888888888888888888888888888888888888888888888888888888888888888888"
            # print "88888888888888888888888888888888888888888888888888888888888888888888888888"
            # print "88888888888888888888888888888888888888888888888888888888888888888888888888"
            # kprint "selection vector", selection_vector

            # Permute vector
            select_run = pg_per_run.keys()[ random.randint(0, n_runs-1 ) ]
            # print "seelect run", select_run
            select_pg = random.randint(0, pg_per_run[select_run])
            # print "seelect pg", select_pg
            if biasSelection:
                # if we bias our selection, in half of the cases we only
                # propose steps that are either zero or first peakgroups. Make sure that for 
                if random.random() > 0.5:
                    opg = select_pg
                    select_pg = random.randint(0, min(BIAS_PEAKGROUPS, pg_per_run[select_run]))
                    print "bias selection, change pg from %s to %s" %(opg, select_pg)

            if selection_vector[select_run] == select_pg:
                # its equal, no step 
                continue

            #update vector
            ## selection_vector_new = selection_vector[:]
            selection_vector_new = selection_vector.copy()
            selection_vector_new[select_run] = select_pg

            ##
            ## eval vector
            #

            if usecpp:
                score = optimized.c_evalvec(tree_path, selection_vector_new, tree_start, mpep, tr_data, transfer_width, False, mypghash)
            else:
                score = evalvec(            tree_path, selection_vector_new, tree_start, mpep, tr_data, transfer_width=transfer_width, mypghash=mypghash)

            if False:
                score = evalvec(              tree_path, selection_vector_new, tree_start, mpep, tr_data, transfer_width, False, mypghash=mypghash)
                c_score = optimized.c_evalvec(tree_path, selection_vector_new, tree_start, mpep, tr_data, transfer_width, False, mypghash)
                oldscore = evalvec_old_python(tree_path, selection_vector_new, tree_start, mpep, tr_data, transfer_width=transfer_width)
                print oldscore, score, c_score
                if oldscore > -1e20:
                    print "assert"
                    assert ( abs(oldscore - score) < 1e-5)
                    assert ( abs(oldscore - c_score) < 1e-5)

            delta_score = score - prev_score
            if verbose:
                print prev_score, "proposed: ", selection_vector_new.values(), score, " -> delta",  delta_score

            r = random.random()
            accept = (delta_score > 0) or (r < math.exp(delta_score/f))

            if score >= best_score:
                if score > best_score:
                    # we have not seen this score before, it is better!
                    burn_in_time = i
                    time_in_best_config = 0
                    # print 'set time best config to zero', score, best_score
                elif math.fabs(score - best_score) < 1e-6:
                    # we return to the best score
                    time_in_best_config += 1
                    # print 'return to best score ', score, best_score
                else:
                    # print 'WHAT here? ', score, best_score
                    pass
                best_score = score 
                best_config = selection_vector_new
            else:
                # new score is worse, if we do not accept then it means we stay
                # with the same score
                if not accept:
                    time_in_best_config += 1

            # If we accept the new score, change the selection vector and the score
            if accept:
                if verbose:
                    print "accept", r, "exp^%s" %(delta_score/f)
                selection_vector = selection_vector_new
                prev_score = score

        if verbose:
            print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
            print "  MCMC Stats: "
            print "    Nr it: ", nrit
            print "    Burnin time: ", burn_in_time
            print "    Time in best config: ", time_in_best_config
        return best_score, best_config

def doBayesianAlignmentDescrete(exp, multipeptides, max_rt_diff, initial_alignment_cutoff,
                        smoothing_method, doPlot=True, outfile=None, transfer_fxn="bartlett"):
    """
    Bayesian alignment

    smoothing seems to change from 36 seconds to 9 seconds (gain 28 seconds)
    erfc in C seems to change from 26 seconds to 4 seconds (gain 22 seconds)

    79 seconds python / scipy smoothing
    52 seconds cython smoothing
    43 seconds no smoothing

    52 seconds scipy cdf 
    30 seconds using C erfc
    26 seconds using no cdf calculation

    takes ca 14 seconds to setup everything and start computing ... 

    from an initial 65 seconds for 74 peptides * 10 runs  = 10 runpeptides / second
    -> we are down to 16 seconds for 74 peptides * 10 runs = 46 runpeptides / second
    """

    doPlot = False
    verbose = False
    
    fh = None
    if outfile is not None:
        fh = open(outfile, "w")

    #  to compare
    # Perform work
    fdr_cutoff = 0.01
    aligned_fdr_cutoff = 0.05
    rt_diff_isotope = 10
    use_RT_correction = True
    stdev_max_rt_per_run = 3.0
    use_local_stdev = False
    mrt_diff = 30.0
    mst_al = TreeConsensusAlignment(mrt_diff, fdr_cutoff, aligned_fdr_cutoff, 
                                rt_diff_isotope=rt_diff_isotope,
                                correctRT_using_pg=use_RT_correction,
                                stdev_max_rt_per_run=stdev_max_rt_per_run,
                                use_local_stdev=use_local_stdev)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Set parameters
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # Only select any peak in the chromatogram if the chance that any peak is
    # present is better than this probability.
    h0_cutoff = 0.5

    ptransfer = "all"
    ptransfer = "equal" # boxcar / rectangle
    ptransfer = "bartlett" #triangular
    ptransfer = "gaussian" # gaussian window
    # ptransfer = "bartlett" #triangular
    ptransfer = transfer_fxn

    peak_sd = 15 # 30 seconds peak (2 stdev 95 \% of all signal)
    peak_sd = 10 # 30 seconds peak (3 stdev 99.7 \% of all signal)

    # Increase uncertainty by a factor of 2.5 when transferring probabilities
    # from one run to another
    transfer_width = peak_sd * 2.5

    transfer_width = 30.0
    transfer_width = 45.0

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
    ### spl_aligner = SplineAligner(initial_alignment_cutoff)
    ### tr_data = LightTransformationData()
    ### for r1 in exp.runs:
    ###     for r2 in exp.runs:
    ###         addDataToTrafo(tr_data, r1, r2,
    ###                        spl_aligner, multipeptides, smoothing_method,
    ###                        max_rt_diff, sd_max_data_length=300)

    print("Compute pairwise alignments took %0.2fs" % (time.time() - start) )
    start = time.time()

    spl_aligner = SplineAligner(initial_alignment_cutoff)
    tree = MinimumSpanningTree(getDistanceMatrix(exp, multipeptides, spl_aligner))

    # Get alignments (only need edges on the tree!)
    tr_data = LightTransformationData()
    for edge in tree:
        addDataToTrafo(tr_data, exp.runs[edge[0]], exp.runs[edge[1]],
                       spl_aligner, multipeptides, smoothing_method,
                       max_rt_diff)

    tree_mapped = [ (exp.runs[a].get_id(), exp.runs[b].get_id()) for a,b in tree]

    print "Tree"
    print tree
    print tree_mapped

    n_runs = len(exp.runs)

    run_mapping = [r.get_id() for r in exp.runs]

    print "Select path through tree"

    def getOuterNodes(tree_in):
        # Node dictionary, keep track of which nodes have how many connections
        ndict = {} 
        for e in tree_in:
            tmp = ndict.get(e[0], 0)
            tmp += 1
            ndict[ e[0] ] = tmp
            tmp = ndict.get(e[1], 0)
            tmp += 1
            ndict[ e[1] ] = tmp

        print "node dict", ndict
        outer_nodes = [ k for k,v in ndict.iteritems() if v == 1]
        return outer_nodes 

    starting_values  = getOuterNodes(tree_mapped)
    print "Possiblel starting points", starting_values

    def walkTree(tree, start, visited):
        res = []
        for e in tree:
            if e in visited:
                continue

            if start in e:
                # visit edge e, append to result
                visited.append(e)
                if e[0] == start:
                    res.append( [e[0], e[1]] )
                    res.extend( walkTree(tree, e[1], visited) )
                else:
                    res.append( [e[1], e[0]] )
                    res.extend( walkTree(tree, e[0], visited) )
        return res

    ### tree = [ (0,1), (1,2), (1,3), (1,4), (2,5), (2,6)]
    tree_start = starting_values[0]
    print "start ", tree_start

    tmp = []
    tree_path = walkTree(tree_mapped, tree_start, tmp)

    print "path", tree_path

    cnt_better_greedy = 0
    cnt = 0

    greedy_select_zero = 0
    bayes_select_zero = 0

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Step 2 : Iterate through all peptides
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    for pepcnt,mpep in enumerate(multipeptides):

        ### # Step 2.1 : Compute the retention time space (min / max)
        ### # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        ### rts = [pg.get_normalized_retentiontime()
        ###         for p in mpep.getAllPeptides()
        ###             for pg in p.getAllPeakgroups() ]

        ### min_rt = min(rts)
        ### max_rt = max(rts)
        ### min_rt -= abs(max_rt - min_rt) * rt_window_ext
        ### max_rt += abs(max_rt - min_rt) * rt_window_ext

        ### # Hack to ensure that the two are never equal
        ### if min_rt == max_rt:
        ###     min_rt -= peak_sd
        ###     max_rt += peak_sd

        ### # Step 2.2 : Collect peakgroup data across runs
        ### # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        ### h0 = {}
        ### run_likelihood = {}
        ### x = np.linspace(min_rt, max_rt, bins)
        ### ## doBayes_collect_pg_data(mpep, h0, run_likelihood, x, min_rt, max_rt, bins, peak_sd)

        ## if len(p.getAllPeakgroups()) < 2*len(mpep.getAllPeptides()):
        ##     # print "continue"
        ##     continue

        ## print "00000000000000000000000000000000000 new peptide (bayes)", mpep.getAllPeptides()[0].get_id(), pepcnt

        ## pp = "11047_IISYLPDTTYLNENM(UniMod:35)R_3_run0"
        ## verbose = False
        ## if mpep.getAllPeptides()[0].get_id() != pp:
        ##     continue
        ## else:
        ##     verbose = True

        if verbose:
            print "00000000000000000000000000000000000 new peptide (bayes)", mpep.getAllPeptides()[0].get_id(), pepcnt

        # n_runs = len(mpep.getAllPeptides())
        # selection_vector = [0 for i in range(n_runs)]

        # Initialize with zero
        selection_vector = dict([(r.get_id(), 0) for r in exp.runs])

        # print "use nr runs", n_runs
        ## print "n runs", n_runs
        ## print "all peps", len( mpep.getAllPeptides() )

        ### pg_per_run = dict([ [p.run.get_id(), len( p.getAllPeakgroups() )] for p in mpep.getAllPeptides()])
        #### pg_per_run = dict([ [r.get_id(), len( p.getAllPeakgroups() )] for r in exp.runs])
        pg_per_run = {}
        for r in exp.runs:
            if mpep.hasPrecursorGroup(r.get_id()):
                g = mpep.getPrecursorGroup(r.get_id()).getAllPrecursors()
                assert len(g) == 1
                pg_per_run[r.get_id()] = len(list(g[0].getAllPeakgroups()))
            else:
                pg_per_run[r.get_id()] = 0

        if verbose:
            print "pg_per_run", pg_per_run, len(pg_per_run)
            print "sel vec", selection_vector

        nr_comb = 1.0
        for a in pg_per_run.values():
            nr_comb = nr_comb*(a+1)

        """
        Assume that we have three runs and the path is 1 -> 2 -> 3 

        we now assume that p(1|2,3) = p(1|2) e.g that if we know 2 than the
        position of pg in run 1 is fully determined. 

        We can thus write
            p(1,2,3) = p(1|2,3)p(2|3)p(3)
                     = p(1|2)  p(2|3)p(3) #  use cond. indep

        """
        best = mpep.find_best_peptide_pg()

        if float(best.get_value("m_score")) > 0.0012271012721661626:
            continue

        nriterations = 2000
        # minimum value: 100 * nr_runs
        # safe value: 100 * nr_runs * safety = 1000 * nr_runs

        for i in range(0):

            # random starting vectors
            v = {}
            for r in exp.runs:
                if pg_per_run[r.get_id()] != 0:
                    v[r.get_id()] = random.randint(0, pg_per_run[r.get_id()])
                else:
                    v[r.get_id()] = 0

            best_score, best_config = mcmcrun(nriterations, v, tree_path,
                                              tree_start, pg_per_run, mpep,
                                              tr_data, n_runs, transfer_width=transfer_width, verbose=False)
            print "random vec res", best_score, best_config.values()
            print "2222222222222222222  ", 

        for i in range(0):

            # zero starting vector
            v = dict([(r.get_id(), 0) for r in exp.runs])
            best_score, best_config = mcmcrun(nriterations, v, tree_path,
                                              tree_start, pg_per_run, mpep,
                                              tr_data, n_runs, transfer_width=transfer_width, verbose=False)
            print "9999999 zero vec res", best_score, best_config.values()
            print "44444444444444  ", 

        if True:
            best_score, best_config = optimized.c_mcmcrun(nriterations, selection_vector, tree_path,
                                              tree_start, pg_per_run, mpep,
                                              tr_data, n_runs, transfer_width=transfer_width, verbose=False)
        else:
            best_score, best_config = mcmcrun(nriterations, selection_vector, tree_path,
                                              tree_start, pg_per_run, mpep,
                                              tr_data, n_runs, transfer_width=transfer_width, verbose=False, usecpp=True)



        #print "1111111111111111111111111111111111111111111111111111111111111111 "
        greedy = getGreedyVec(mpep, pg_per_run)
        greedy_res = evalvec(tree_path, greedy, tree_start, mpep, tr_data, transfer_width=transfer_width, verbose=False)

        greedy_select_zero += len([v for v in greedy.values() if v == 0])
        bayes_select_zero += len([v for v in best_config.values() if v == 0])

        cnt += 1
        ## print "00000000000000000000000000000000000 new peptide (bayes)", mpep.getAllPeptides()[0].get_id(), pepcnt
        ## print "r:", best_config.values(), best_score, \
        ##         "greedy:", greedy_res, greedy.values(), "%s combs" % nr_comb, "at pcnt %.2f" % (cnt_better_greedy *100.0 / cnt), \
        ##         " zeros selected %.2f" % (greedy_select_zero*100.0 / (bayes_select_zero+1))


        if False:
            # cmp with MST 
            best = mpep.find_best_peptide_pg()
            print "  mst_al, selected best peptide", best
            # Use this peptide to generate a cluster
            mst_al.verbose = False
            mst_pgs = [pg_ for pg_ in mst_al._findAllPGForSeed(tree_mapped, tr_data, mpep, best, {})]
            print "  got %s selected pg" % len(mst_pgs)

            mst_vec = {}
            for r in pg_per_run.keys():
                print "iteratue through runs", r
                mst_vec[r] = 0
                for k in range(pg_per_run[r]):
                    mypg = getPG(mpep, r, k+1)
                    ### print "cmp ", mypg, mst_pgs
                    if mypg in mst_pgs:
                        mst_vec[r] = k+1
                        print "set true for pg %s in run %s" % (k+1, r)

            print "compute MST score:"
            mst_res = evalvec(tree_path, mst_vec, tree_start, mpep, tr_data, transfer_width=transfer_width, verbose=False)
            print "m:", mst_vec.values(), mst_res

        best_score = evalvec(tree_path, best_config, tree_start, mpep, tr_data, transfer_width=transfer_width, verbose=False)

        if pepcnt % 100 == 0:
            print "Progress", pepcnt

        if best_score > greedy_res and False:
            cnt_better_greedy += 1
            print "00000000000000000000000000000000000 new peptide (bayes)", mpep.getAllPeptides()[0].get_id(), pepcnt
            print "r:", best_config.values(), best_score, \
                    "greedy:", greedy_res, greedy.values(), "%s combs" % nr_comb, "at pcnt %.2f" % (cnt_better_greedy *100.0 / cnt), \
                    " zeros selected %.2f" % (greedy_select_zero*100.0 / (bayes_select_zero+1))
            print "g:", greedy.values(), greedy_res
        elif best_score < greedy_res:
            print "8888888888888888888888888888888888888888888888888888"
            print "greedy better!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", mpep.getAllPeptides()[0].get_id(), pepcnt
            print "r:", best_config.values(), best_score, \
                    "greedy:", greedy_res, greedy.values(), "%s combs" % nr_comb, "at pcnt %.2f" % (cnt_better_greedy *100.0 / cnt), \
                    " zeros selected %.2f" % (greedy_select_zero*100.0 / (bayes_select_zero+1))
            # return -1

        idx = 0
        for k,v in pg_per_run.iteritems():
            continue

            print "  I am here in run ", k, "with idx", idx
            idx +=1 
            if v == 0:
                print "    No pg for this run"
            else:
                for pgnr in range(v):
                    pg = getPG(mpep, k, pgnr+1)
                    print "    with pg ", pg, " with prob %s (h0 is %s)" % (float(pg.get_value("h_score")), 
                                                                       float(pg.get_value("h0_score")))

        ## select peakgroups
        for k,v in best_config.iteritems():
            mypg = getPG(mpep, k, v)
            if mypg is not None:
                mypg.select_this_peakgroup()


def getGreedyVec(mpep, pg_per_run):
    # greedy = dict([(r.get_id(), 1) for r in exp.runs])
    greedy = {}
    for k,v in pg_per_run.iteritems():
        if v == 0:
            greedy[k] = 0
        else:
            mypg = getPG(mpep, k, 1)
            if float(mypg.get_value("m_score")) < 0.01:
                greedy[k] = 1
            else:
                greedy[k] = 0
    return greedy

