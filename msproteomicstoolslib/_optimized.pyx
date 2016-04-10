# encoding: latin-1
# distutils: language = c++
cimport cython
from cython.operator cimport dereference as deref, preincrement as inc
cimport libc.stdlib
cimport numpy as np
import numpy as np
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool
from libcpp.string cimport string
import random
from libc.stdlib cimport srand, rand, RAND_MAX


cdef extern from "algorithm":
    vector[double].iterator c_upper_bound "std::upper_bound" (vector[double].iterator, vector[double].iterator, double x)
cdef extern from "iterator":
    size_t c_distance "std::distance" (vector[double].iterator, vector[double].iterator)

cdef extern from "math.h":
    double c_erfc "erfc" (double x)
    double c_fabs "fabs" (double x)
    double c_log "log" (double x)
    double c_exp "exp" (double x)
    double c_floor "floor" (double x)
cdef extern from "time.h":
    double c_time "time" (void*)


# http://stackoverflow.com/questions/28973153/cython-how-to-wrap-a-c-function-that-returns-a-c-object
cdef class Cy_LinearRangeInterpolator(object):

    cdef vector[double] x
    cdef vector[double] y

    def __cinit__(self):
        pass

    def initData(self, vector[double]& x_in, vector[double] y_in):
        self.x.swap(x_in)
        self.y.swap(y_in)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict(self, double value):

        cdef size_t idx
        cdef double x_0, x_1, y_0, y_1

        # // find nearest pair of points
        # std::vector<double>::const_iterator it = std::upper_bound(x_.begin(), x_.end(), x);
        cdef vector[double].iterator it 
        it = c_upper_bound(self.x.begin(), self.x.end(), value)

        # // interpolator is guaranteed to be only evaluated on points x, x_.front() =< x =< x x.back()
        # // see TransformationModelInterpolated::evaluate

        # // compute interpolation
        # // the only point that is > then an element in our series is y_.back()
        # // see call guarantee above
        if it == self.x.end():
            return self.y.back()
        else:
            idx = c_distance(self.x.begin(), it)
            x_0 = self.x[idx - 1]
            x_1 = self.x[idx]
            y_0 = self.y[idx - 1]
            y_1 = self.y[idx]
            return y_0 + (y_1 - y_0) * (value - x_0) / (x_1 - x_0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize(self, np.float64_t[:] x_in, np.float64_t[:] y_in):
        #  data needs to be sorted !
        # data1 is the predictor (e.g. the input) -> x
        # data2 is the response (e.g. what we want to predict) -> y
        cdef Py_ssize_t i
        self.x.clear()
        self.y.clear()
        for i in range(x_in.shape[0]):
             self.x.push_back(x_in[i])
        for i in range(y_in.shape[0]):
            self.y.push_back(y_in[i])

cdef class Cy_LinearInterpolator(object):

    cdef Cy_LinearRangeInterpolator interp_
    cdef double xmin_
    cdef double xmax_
    cdef double ymin_
    cdef double ymax_

    def __cinit__(self):
        self.interp_ = Cy_LinearRangeInterpolator()

    def predictSingleValue(self, double value):
        if value > self.xmin_ and value < self.xmax_:
            return self.interp_.predict(value)
        else:
            return self.ymin_ + (self.ymax_ - self.ymin_) * (value - self.xmin_) / (self.xmax_ - self.xmin_)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict(self, np.float64_t[:] values):
        cdef Py_ssize_t i
        result = np.zeros((values.shape[0],), dtype=np.float64)
        cdef np.float64_t[:] view = result
        for i in range(values.shape[0]):
            view[i] = self.predictSingleValue(values[i])
        return result

    def initialize(self, np.float64_t[:] x_in, np.float64_t[:] y_in):
        self.xmin_ = x_in[0]
        self.xmax_ = x_in[-1]
        self.ymin_ = y_in[0]
        self.ymax_ = y_in[-1]
        self.interp_.initialize(x_in, y_in)

@cython.boundscheck(False)
@cython.wraparound(False)
def erfc(double x):
    return c_erfc(x)

@cython.boundscheck(False)
@cython.wraparound(False)
def norm_cdf(double x):
    # 1 / math.sqrt(2)
    return 0.5 * c_erfc(x * 0.7071067811865475) 

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_score_update(double x, double current_score):
    cdef double cdf_v = norm_cdf(x)

    if (cdf_v > 0.5):
        cdf_v = 1-cdf_v

    # Catch cases where we basically have zero probability
    #  -> simply add a very large negative number to the score to
    #     make it unlikely to ever pick such a combination
    if cdf_v > 0.0:
        return (current_score + c_log(cdf_v))
    else:
        return (current_score - 999999999.0)

cdef c_norm_cdf(double x):
    # 1 / math.sqrt(2)
    return 0.5 * c_erfc(x * 0.7071067811865475) 

cdef c_fast_score_update(double x, double current_score):
    cdef double cdf_v = norm_cdf(x)

    if (cdf_v > 0.5):
        cdf_v = 1-cdf_v

    # Catch cases where we basically have zero probability
    #  -> simply add a very large negative number to the score to
    #     make it unlikely to ever pick such a combination
    if cdf_v > 0.0:
        return (current_score + c_log(cdf_v))
    else:
        return (current_score - 999999999.0)

def getScoreAndRT(mpep, run_id, pg, mypghash=None):
    current_score = mypghash[run_id][pg][0]
    score_h0 = mypghash[run_id][pg][1]
    rt = mypghash[run_id][pg][2]
    return (current_score, rt)

def getH0Score(mpep, run_id, mypghash=None):
    return mypghash[run_id][0]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def c_evalvec(list tree_path, dict selection_vector_new, bytes tree_start, mpep, tr_data, double transfer_width = 30, bool verbose=False, dict mypghash=None):

    cdef int pg
    cdef double current_score
    cdef double rt_curr, pg_score, rt, norm_rt_diff, expected_rt

    # starting point
    pg = selection_vector_new[ tree_start ]

    # keep track of all positions of peakgroups from previous runs
    rt_positions = {}

    # Do everything for the first peakgroup !
    current_score = 0.0
    if pg != 0:
        pg_score, rt = getScoreAndRT(mpep, tree_start, pg, mypghash)
        rt_positions[ tree_start ] = rt
        current_score += c_log(pg_score)
    else:
        # We have selected H0 here, thus append the h0 score
        current_score += c_log( getH0Score(mpep, tree_start, mypghash) )

    for e in tree_path:
        ####
        #### Compute p(e[1]|e[0])
        ####
        prev_run = e[0]
        curr_run = e[1]
        pg = selection_vector_new[ curr_run ]

        # get the retention time position in the previous run
        rt_prev = rt_positions.get( prev_run, None)

        if pg != 0:
            pg_score, rt_curr = getScoreAndRT(mpep, curr_run, pg, mypghash)
            current_score += c_log(pg_score)

            #
            # p( e[1] ) -> we need to compute p( e[1] | e[0] ) 
            #
            #  -> get the probability that we have a peak in the current run
            #  e[1] eluting at position rt_curr given that there is a peak in
            #  run e[0] eluting at position rt_prev
            #

            if rt_prev is not None:

                source = prev_run
                target = curr_run
                # Try to get fast version of trafo first, then try slow one
                try:
                    mytrafo = tr_data.getTrafo(source, target)
                    # expected_rt = mytrafo.internal_interpolation.predictSingleValue(deref(rt_prev_).second)
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
                current_score = c_fast_score_update(norm_rt_diff, current_score)

            else:
                # no previous run, this means all previous runs were
                # empty and we cannot add any RT so far
                pass

            # store our current position 
            rt_positions[curr_run] = rt_curr

        else:

            # We have selected H0 here, thus append the h0 score
            current_score += c_log( getH0Score(mpep, curr_run, mypghash) )

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

    return current_score

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
    return list(prgr.getAllPrecursors()[0].getAllPeakgroups())[pg-1]
    ## return ( list(list(mpep.getAllPeptides())[ run ].getAllPeakgroups())[pg - 1] )

# good random nr
# http://stackoverflow.com/questions/6862844/how-bad-rand-from-stdlib-h-is
# http://eternallyconfuzzled.com/arts/jsw_art_rand.aspx# good random nr
@cython.cdivision(True)
cdef getRandomInt(int a, int b):
    # return random.randint(a, b)
    return a + (rand() % <int>(b - a + 1)) # best choice, may have some bias towards lower values
    # return a + rand() / (RAND_MAX / (b - a) + 1)

@cython.cdivision(True)
cdef getRandomFloat():
    #return random.random()
    return rand() * 1.0 /(RAND_MAX);

def c_mcmcrun(int nrit_, selection_vector, tree_path, bytes tree_start, pg_per_run, mpep,
            tr_data, int n_runs_, transfer_width, double f_=1.0, verbose=False, bool biasSelection_=False):
        """
        MC MC in cython
        http://hplgit.github.io/teamods/MC_cython/main_MC_cython.html

        26 s is the new low
        """
        srand(<unsigned int>c_time(NULL));

        cdef int nrit = nrit_
        cdef int n_runs = n_runs_
        cdef double f = f_
        cdef bool biasSelection = biasSelection_

        cdef double delta_score, score, best_score, prev_score
        cdef int BIAS_PEAKGROUPS=1
        cdef bytes select_run
        cdef int select_pg
        cdef int i
        cdef int pg_, pg
        cdef bool accept
        cdef double tmp_score, h0_score, rt

        cdef map[string, map[int, vector[double] ] ] peakgroup_score_hash
        cdef map[int, vector[double] ] tmp_run_hash
        cdef vector[double] tmp_vec
        cdef string c_tree_start = tree_start

        mypghash = {}

        # Create hash which can be accessed as:
        #   h[run][pg][0] = score
        #   h[run][pg][1] = h0 score
        #   h[run][pg][2] = rt
        if True:
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

        # ca 16.5s for the hash
        cdef int burn_in_time = 0
        cdef int time_in_best_config = 0

        # prev_score = evalvec(          tree_path, selection_vector, tree_start, pg_per_run, mpep, tr_data, transfer_width=transfer_width, mypghash=mypghash)
        best_config = selection_vector
        prev_score = c_evalvec(tree_path, selection_vector, tree_start, mpep, tr_data, transfer_width, False, mypghash)
        best_score = prev_score

        # ca 17.02 for the initial score = takes ca 1 second

        if verbose:
            print "start: ", prev_score, selection_vector

        for i in range(nrit):
            # print "88888888888888888888888888888888888888888888888888888888888888888888888888"
            # print "88888888888888888888888888888888888888888888888888888888888888888888888888"
            # print "88888888888888888888888888888888888888888888888888888888888888888888888888"
            # kprint "selection vector", selection_vector

            # Permute vector
            ### select_run = pg_per_run.keys()[ getRandomInt(0, n_runs-1)  ]
            select_run = pg_per_run.keys()[ rand() % (n_runs-1)  ]
            # print "seelect run", select_run
            if pg_per_run[select_run] == 0:
                select_pg = 0
            else:
                ### select_pg = getRandomInt(0, pg_per_run[select_run])
                select_pg = rand() % pg_per_run[select_run]
            # print "seelect pg", select_pg
            if biasSelection:
                # if we bias our selection, in half of the cases we only
                # propose steps that are either zero or first peakgroups. Make sure that for 
                if getRandomFloat() > 0.5:
                    opg = select_pg
                    select_pg = getRandomInt(0, min(BIAS_PEAKGROUPS, pg_per_run[select_run]))
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

            # score = evalvec(          tree_path, selection_vector_new, tree_start, pg_per_run, mpep, tr_data, transfer_width=transfer_width, mypghash=mypghash)
            # 17 seconds until here -> all below takes 9 seconds!!!
            score = c_evalvec(tree_path, selection_vector_new, tree_start, mpep, tr_data, transfer_width, False, mypghash)
            # 21 seconds until here -> all below takes 5 seconds!!!
            ### print "got c eval", score

            delta_score = score - prev_score
            if verbose:
                print prev_score, "proposed: ", selection_vector_new.values(), score, " -> delta",  delta_score


            r = getRandomFloat()
            accept = delta_score > 0 or r < c_exp(delta_score/f)
            # 21 seconds until here -> all below takes 5 seconds!!!

            if score >= best_score:
                if score > best_score:
                    # we have not seen this score before, it is better!
                    burn_in_time = i
                    time_in_best_config = 0
                    # print 'set time best config to zero', score, best_score
                elif c_fabs(score - best_score) < 1e-6:
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
    
            # 22 seconds until here -> all below takes 4 seconds!!!
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

        # 22.5 seconds
        return best_score, best_config
