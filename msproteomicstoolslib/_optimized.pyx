# encoding: latin-1
# distutils: language = c++
cimport cython
from cython.operator cimport dereference as deref, preincrement as inc
cimport libc.stdlib
cimport numpy as np
import numpy as np
from libcpp.vector cimport vector


cdef extern from "algorithm":
    vector[double].iterator c_upper_bound "std::upper_bound" (vector[double].iterator, vector[double].iterator, double x)
cdef extern from "iterator":
    size_t c_distance "std::distance" (vector[double].iterator, vector[double].iterator)

cdef extern from "math.h":
    double c_erfc "erfc" (double x)
cdef extern from "math.h":
    double c_log "log" (double x)


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
    cdef double cdf_v
    cdf_v = norm_cdf(x)

    if (cdf_v > 0.5):
        cdf_v = 1-cdf_v

    # Catch cases where we basically have zero probability
    #  -> simply add a very large negative number to the score to
    #     make it unlikely to ever pick such a combination
    if cdf_v > 0.0:
        return (current_score + c_log(cdf_v))
    else:
        # current_score += -99999999999999999999999
        return (current_score - 999999999.0)


