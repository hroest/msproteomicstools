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
import scipy.stats
import time
from msproteomicstoolslib.algorithms.alignment.Multipeptide import Multipeptide
from msproteomicstoolslib.algorithms.alignment.SplineAligner import SplineAligner
from msproteomicstoolslib.format.TransformationCollection import TransformationCollection, LightTransformationData
from msproteomicstoolslib.algorithms.alignment.AlignmentHelper import addDataToTrafo

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
                                ptransfer, transfer_width, verb=False):
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

        # If verbose
        if verb:
            print "use method", ptransfer
            print "convert from", source, " to ", target
            print "predict for", x[j]
            print "results in", expected_rt
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
                # TODO: TIME : this can be really slow
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
                        smoothing_method, doPlot=True, outfile=None, transfer_fxn="bartlett"):
    """
    Bayesian alignment
    """
    
    fh = None
    if outfile is not None:
        fh = open(outfile, "w")

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
            addDataToTrafo(tr_data, r1, r2,
                           spl_aligner, multipeptides, smoothing_method,
                           max_rt_diff, sd_max_data_length=300)

    print("Compute pairwise alignments took %0.2fs" % (time.time() - start) )
    start = time.time()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Step 2 : Iterate through all peptides
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    for pepcnt,mpep in enumerate(multipeptides):

        # print "00000000000000000000000000000000000 new peptide (bayes)", mpep.getAllPeptides()[0].get_id(), pepcnt

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

                tmp_prod = doBayes_collect_product_data(mpep, tr_data, m, j, h0, 
                                run_likelihood, x, peak_sd, bins, ptransfer, transfer_width)

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

            # select the peak with the maximum probability weight
            best_psum = max([(pg.get_value("accum_p"), pg) for pg in p.getAllPeakgroups()])
            # print "best peak", best_psum[1], "with sum", best_psum[0]
            best_pg = best_psum[1]
            if float(best_pg.get_value("h0_score")) < h0_cutoff: 
                best_pg.select_this_peakgroup()
                if fh is not None:
                    fh.write("%s\t%s\n" % (best_psum[1].get_value("id"), best_psum[0]) )
                
        # print "peptide (bayes)", mpep.getAllPeptides()[0].get_id()

    if fh is not None:
        fh.close()

    print("Bayesian alignment took %0.2fs" % (time.time() - start) )


def evalvec(tree_path, selection_vector_new, tree_start, pg_per_run, mpep, tr_data, verbose=False):
    import math

    if verbose:
        r = 1.0
        for a in pg_per_run:
            r = r*(a+1)
        print "Eval vector", selection_vector_new
        print "combinatorics ", pg_per_run, "total comb", r
        print "follow tree path", tree_path

    # starting point
    tree_start
    pg = selection_vector_new[ tree_start ]
    if verbose:
        print "Get starting point with run", tree_start, " with pg", pg
        # print pg_per_run

    def getPG(run, pg):
        return ( list(list(mpep.getAllPeptides())[ run ].getAllPeakgroups())[pg - 1] )

    # keep track of all positions of peakgroups from previous runs
    rt_positions = {}

    current_score = 0.0
    if pg != 0:
        ## mypg = list(list(mpep.getAllPeptides())[ tree_start ].getAllPeakgroups())[pg - 1]
        mypg = getPG( tree_start, pg)
        ### rt = mypg.get_normalized_retentiontime()
        rt = float(mypg.get_value("RT"))
        ## print "store RT", rt
        ## print "store RT", float(mypg.get_value("RT"))
        rt_positions[ tree_start ] = rt
        current_score += math.log(float(mypg.get_value("h_score")))
    else:
        # mypg = list(list(mpep.getAllPeptides())[ tree_start ].getAllPeakgroups())[0]
        ## if not pg_per_run[tree_start] == 0:
        mypg = getPG( tree_start, 1)
        current_score += math.log(float(mypg.get_value("h0_score")))
    if verbose:
        print mypg, "my pg with p = %s" % (float(mypg.get_value("h_score"))), " start with scoree !! ", current_score
        print " start with scoree !! ", current_score

    for e in tree_path:
        ## print "in tree path", e
        ## print "sel vector len %s vector" % (len(selection_vector_new)), selection_vector_new
        if verbose:
            print "------------------------------"
            print "  heeeeere, compute p(%s|%s)" % (e[1], e[0])
        prev_run = e[0]
        curr_run = e[1]
        pg = selection_vector_new[ curr_run ]
        if pg != 0:
            mypg = getPG( curr_run, pg )
            # p( e[1] ) -> we need p( e[1] | e[0] ) 

            current_score += math.log(float(mypg.get_value("h_score")))
            if verbose:
                print " = ", mypg , "my pg with p = %s" % (float(mypg.get_value("h_score")))
                print "new score:", current_score
            # get the position in the previous run
            rt_prev = rt_positions.get( prev_run, None)

            rt_curr = float(mypg.get_value("RT"))
            if rt_prev is not None:

                source = list( mpep.getAllPeptides() )[prev_run].run.get_id()
                target = list( mpep.getAllPeptides() )[curr_run].run.get_id()

                expected_rt = tr_data.getTrafo(source, target).predict(
                    [ rt_prev ] )[0]
                if verbose:
                    print "try to get ", source, target
                    print "source RT ", rt_prev
                    print "expected RT ", expected_rt
                    print "current_ RT ", rt_curr

                # Tr to compute p(curr_run | prev_run )

                transfer_width = 100
                p_Bqr_Bjm = scipy.stats.norm.pdf(rt_curr,
                                                 loc = expected_rt ,
                                                 scale = transfer_width)
                # print "current location", p_Bqr_Bjm
                p_Bqr_Bjm = scipy.stats.norm.cdf(rt_curr,
                                                 loc = expected_rt ,
                                                 scale = transfer_width)
                # TODO use logcfd
                # print "current cdf", p_Bqr_Bjm
                # print "cdf param cdf", p_Bqr_Bjm
                cdf_v = p_Bqr_Bjm
                if (cdf_v > 0.5):
                    cdf_v = 1-cdf_v


                current_score += math.log(cdf_v)
                # print current_score

                if verbose:
                    print " = "
                    print "add to score cdf"
                    print "current cdf", cdf_v
                    print "new score:", current_score
            else:
                # no previous run, this means all previous runs were
                # empty and we cannot add any RT so far
                pass


            # store our current position 
            rt_positions[curr_run] = rt_curr

        else:
            # We have selected H0 here, thus append the h0 score
            mypg = list(list(mpep.getAllPeptides())[ curr_run ].getAllPeakgroups())[0]
            current_score += math.log(float(mypg.get_value("h0_score")))
            rt_prev = rt_positions.get( prev_run, None)

            if verbose:
                print "------------------------------"
                print "select h0 for run ", curr_run
                print "new score:", current_score


            # store our current position 
            if rt_prev is not None:

                source = list( mpep.getAllPeptides() )[prev_run].run.get_id()
                target = list( mpep.getAllPeptides() )[curr_run].run.get_id()

                expected_rt = tr_data.getTrafo(source, target).predict(
                    [ rt_prev ] )[0]
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

    # print "total score", current_score
    return current_score

def mcmcrun(nrit, selection_vector, tree_path, tree_start, pg_per_run, mpep, tr_data, n_runs, f=2.5,verbose=False):
        """
        f = 1.0
        # 1 seems too tame
        # 5 seems very wild
        f = 2.5
        """

        prev_score = evalvec(tree_path, selection_vector, tree_start, pg_per_run, mpep, tr_data)
        best_score = prev_score
        best_config = selection_vector

        if verbose:
            print "start: ", prev_score, selection_vector

        for i in range(nrit):
            # print "88888888888888888888888888888888888888888888888888888888888888888888888888"
            # print "88888888888888888888888888888888888888888888888888888888888888888888888888"
            # print "88888888888888888888888888888888888888888888888888888888888888888888888888"
            # kprint "selection vector", selection_vector

            # Permute vector
            import random
            select_run = random.randint(0, n_runs-1 )
            # print "seelect run", select_run
            select_pg = random.randint(0, pg_per_run[select_run])
            # print "seelect pg", select_pg

            if selection_vector[select_run] == select_pg:
                # its equal, no step 
                continue

            #update vector
            selection_vector_new = selection_vector[:]
            selection_vector_new[select_run] = select_pg

            ##
            ## eval vector
            #

            score = evalvec(tree_path, selection_vector_new, tree_start, pg_per_run, mpep, tr_data)
            delta_score = score - prev_score
            if verbose:
                print prev_score, "proposed: ", selection_vector_new, score, " -> delta",  delta_score

            if score >= best_score:
                best_score = score 
                best_config = selection_vector_new
    
            r = random.random()
            import math
            #if delta_score > -5.0:
            r = random.random()
            if r < math.exp(delta_score/f):
                if verbose:
                    print "accept", r, math.exp(delta_score/f)
                selection_vector = selection_vector_new
                prev_score = score

        return best_score, best_config

def doBayesianAlignmentDescrete(exp, multipeptides, max_rt_diff, initial_alignment_cutoff,
                        smoothing_method, doPlot=True, outfile=None, transfer_fxn="bartlett"):
    """
    Bayesian alignment
    """

    doPlot = False
    
    fh = None
    if outfile is not None:
        fh = open(outfile, "w")

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

    from msproteomicstoolslib.algorithms.PADS.MinimumSpanningTree import MinimumSpanningTree
    from msproteomicstoolslib.algorithms.alignment.AlignmentMST import getDistanceMatrix, TreeConsensusAlignment
    spl_aligner = SplineAligner(initial_alignment_cutoff)
    tree = MinimumSpanningTree(getDistanceMatrix(exp, multipeptides, spl_aligner))

    # Get alignments (only need edges on the tree!)
    tr_data = LightTransformationData()
    for edge in tree:
        addDataToTrafo(tr_data, exp.runs[edge[0]], exp.runs[edge[1]],
                       spl_aligner, multipeptides, smoothing_method,
                       max_rt_diff)

    tree_mapped = [ (exp.runs[a].get_id(), exp.runs[b].get_id()) for a,b in tree]

    print tree_mapped

    n_runs = len(exp.runs)

    run_mapping = [r.get_id() for r in exp.runs]
    # n_runs = len(exp.runs)

    print tree

    print "Select path through tree"

    ndict = {}
    for e in tree:
        print e
        tmp = ndict.get(e[0], 0)
        tmp += 1
        ndict[ e[0] ] = tmp
        tmp = ndict.get(e[1], 0)
        tmp += 1
        ndict[ e[1] ] = tmp

    print ndict
    starting_values = [ k for k,v in ndict.iteritems() if v == 1]
    print starting_values

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
    tree_path = walkTree(tree, tree_start, tmp)

    print tree_path
    

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Step 2 : Iterate through all peptides
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    for pepcnt,mpep in enumerate(multipeptides):

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
        ## doBayes_collect_pg_data(mpep, h0, run_likelihood, x, min_rt, max_rt, bins, peak_sd)

        ## if len(p.getAllPeakgroups()) < 2*len(mpep.getAllPeptides()):
        ##     # print "continue"
        ##     continue

        print "00000000000000000000000000000000000 new peptide (bayes)", mpep.getAllPeptides()[0].get_id(), pepcnt

        # n_runs = len(mpep.getAllPeptides())
        selection_vector = [0 for i in range(n_runs)]
        # print "use nr runs", n_runs
        print "n runs", n_runs
        print "all peps", len( mpep.getAllPeptides() )

        ## TODO some runs have zero pg !! 
        pg_per_run = [len( p.getAllPeakgroups() ) for p in mpep.getAllPeptides()]
        print pg_per_run
        print len(pg_per_run)
        # print pg_per_run
        prev_score = evalvec(tree_path, selection_vector, tree_start, pg_per_run, mpep, tr_data)

        # print "initial", prev_score
        r = 1.0
        for a in pg_per_run:
            r = r*(a+1)
        print "combinatorics of", pg_per_run, "giving", r, "combinations. initial score", prev_score

        """
        Assume that the path is 1 -> 2 -> 3

        we now assume that p(1|2,3) = p(1|2) e.g that if we know 2 than the
        position of pg in run 1 is fully determined. 

        We can thus write
            p(1,2,3) = p(1|2,3)p(2|3)p(3)
                     = p(1|2)  p(2|3)p(3) #  use cond. indep

        """

        import random
        for i in range(10):

            # random starting vectors
            v = []
            for i in range(n_runs):
                v.append( random.randint(0, pg_per_run[i]) )

            best_score, best_config = mcmcrun(50, v, tree_path,
                                              tree_start, pg_per_run, mpep,
                                              tr_data, n_runs, verbose=False)
            # print "random vec res", best_score, best_config

        best_score, best_config = mcmcrun(500, selection_vector, tree_path,
                                          tree_start, pg_per_run, mpep,
                                          tr_data, n_runs, verbose=False)

        #print "1111111111111111111111111111111111111111111111111111111111111111 "
        print "final result", best_config, best_score, evalvec(tree_path, best_config, tree_start, pg_per_run, mpep, tr_data, False), \
        "cmp with greedy", evalvec(tree_path, [1 for i in range(n_runs)], tree_start, pg_per_run, mpep, tr_data, False)
        continue

        for p in mpep.getAllPeptides():
            m = p.run.get_id() # current_run id


            print "I am here in run ", m
            print "With peptide ", p, "which has %s pg" % (len(p.getAllPeakgroups()))
            for pg in p.getAllPeakgroups():
                print "  with pg ", pg, " with score %s (h0 is %s)" % (float(pg.get_value("h_score")), 
                                                                       float(pg.get_value("h0_score")))
        return


