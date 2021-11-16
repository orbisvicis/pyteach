#!/usr/bin/env python3

# Type: project
# Teaches: data-analysis, machine-learning, functions, generators, recursion,
#          unit-testing, debugging

import io
import math
import random
import itertools
import textwrap
import statistics
import operator

import jenkspy
import ckwrap
import pandas
import altair
import sympy
import numpy
import scipy
import scipy.stats

from hypothesis import example, given, assume
from hypothesis import strategies as st

import morse_recordings as mrec


# Section 1.1
# This section teaches recursion but can be skipped.

# See:
# https://en.wikipedia.org/wiki/Factorial
def factorial(v):
    if v < 0:
        raise ValueError
    if v <= 1:
        return 1
    else:
        return v * factorial(v-1)

# A good explanation:
# https://betterexplained.com/articles/
#   easy-permutations-and-combinations/
# Return all possible k-length combinations of elements in the given sequence.
# Hint: To avoid calculating combinations between recursion levels, store the
# (partial) solution as a recursion argument. Each recursion then gets you the
# combinations for free.
def combinations(l, k):
    if k < 0:
        raise ValueError
    return _combinations(l, 0, k, ())

def _combinations(l, i, k, c):
    if k <= 0:
        return [c]
    v = []
    for i in range(i, len(l)-k+1):
        v.extend(_combinations(l, i+1, k-1, c+(l[i],)))
    return v


# Section 1.2
# Let's validate our implementations against Python's using the Hypothesis
# unit tester (based on Haskell's QuickCheck). Pytest will collect and run
# all functions starting with "test_".
# $ pytest stats.py

@given(st.integers(min_value=0, max_value=100))
@example(0)
@example(1)
def test_factorial(x):
    assert factorial(x) == math.factorial(x)

# If the requested combination length K exceeds the size of the list the
# possible clusterings is an empty set. We want to test this, so allow K
# to exceed the list length by the given ratio.
@st.composite
def list_choose_combinations(draw, max_size=None, ratio=1.5):
    xs = draw(st.lists(st.integers(), max_size=max_size))
    k = draw(st.integers(min_value=0, max_value=int(len(xs)*ratio)))
    return (xs, k)

# Our version is slower; limit the maximum list
# length so we don't exceed the test deadline.
@given(list_choose_combinations(max_size=15))
def test_combinations(t):
    assert combinations(*t) == list(itertools.combinations(*t))


# Section 1.3

# How many ways can you split a list? This is a simple combinatorics problem;
# given a list of size N into K splits there should be C(N-1,K-1) combinations.
def list_splits(l, k):
    if k > max(1, len(l)):
        raise ValueError
    if k < 1:
        raise ValueError
    ix = range(1, len(l))
    for ss in itertools.combinations(ix, k-1):
        ss = [None] + list(ss) + [None]
        cs = []
        for r in zip(ss, ss[1:]):
            s = slice(*r)
            cs.append(l[s])
        yield cs


# Section 1.4
# Let's validate the expected amount of splits using the
# combinations function from the math module.

@st.composite
def list_choose_splits(draw, max_size=None):
    xs = draw(st.lists(st.integers(), min_size=0, max_size=max_size))
    k = draw(st.integers(min_value=1, max_value=max(1, len(xs))))
    return (xs, k)

@given(list_choose_splits(max_size=15))
def test_list_split(t):
    assert len(list(list_splits(*t))) == math.comb(max(0, len(t[0])-1), t[1]-1)


# Section 2.1

# See:
# https://en.wikipedia.org/wiki/Mean
def mean(l):
    if not l:
        raise ValueError
    return sum(l) / len(l)

# See:
# https://en.wikipedia.org/wiki/Median
def median(l, sort=True):
    if not l:
        raise ValueError
    if sort:
        l = sorted(l)
    d,m = divmod(len(l), 2)
    if m == 0:
        return (l[d-1] + l[d]) / 2
    else:
        return l[d]

# See:
# https://en.wikipedia.org/wiki/Variance
#   #Population_variance_and_sample_variance
def variance(l, m):
    if not l:
        raise ValueError
    return sum((i-m)**2 for i in l) / len(l)

# See:
# https://en.wikipedia.org/wiki/Standard_deviation
def stdev(l, m):
    return math.sqrt(variance(l, m))

# See:
# https://en.wikipedia.org/wiki/Squared_deviations_from_the_mean
def sdm(l, m):
    if not l:
        raise ValueError
    return sum((i-m)**2 for i in l)

# See:
# https://en.wikipedia.org/wiki/Median_absolute_deviation
def mad(l, m):
    if not l:
        raise ValueError
    return median(abs(i-m) for i in l)

# See:
# https://en.wikipedia.org/wiki/Normal_distribution
# Parameters:
#   mu: mean
#   sigma: standard deviation
#   x: data point
# Also note, as this will be important later:
# https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
def normal_pdf(mu, sigma, x):
    if sigma <= 0:
        raise ValueError
    f = 1 / (sigma * math.sqrt(2 * math.pi))
    e = math.e ** (-1/2 * ((x - mu) / sigma)**2)
    return f*e

# Hint: check the math module documentation for 'erf', the error function.
def normal_cdf(mu, sigma, x):
    if sigma <= 0:
        raise ValueError
    return (1 + math.erf((x - mu) / (sigma * math.sqrt(2)))) / 2


# Section 2.2
# Given a list of values $x$, the error between $x$ and a given value $s$ can
# be defined as:
#
# $$
# E = \sum_i \lvert x_i - s \rvert^p
# $$
#
# Where:
#
# $$
# \begin{aligned}
# x = (x_1, x_2, \ldots, x_n) \ &:\ \text{a list of rational numbers} \\
# \lvert x \rvert \ &:\ \text{number of elements in the list if $x$ is a list} \\
# \end{aligned}
# $$
#
# Mode, median and mean are values which minimize $E$ for a given $p$:
#
# $$
# \begin{aligned}
# \text{mode of x} = \arg \min_s \sum_i \lvert x_i - s \rvert^0 \\
# \text{median of x} = \arg \min_s \sum_i \lvert x_i - s \rvert^1 \\
# \text{mean of x} = \arg \min_s \sum_i \lvert x_i - s \rvert^2 \\
# \end{aligned}
# $$
#
# $E$ is also closely related to other values. Variance is the per-element
# error:
#
# $$
# \mathrm{Var}[x] = \frac{E}{\lvert x \rvert}
# $$
#
# The $L^p$-norm is the $p$-th root of $E$ for a given set of vectors $v$
# originating at $s$ such that $s$ is zero:
#
# $$
# \begin{aligned}
# \lvert v \rvert _p &= ( \sum_i \lvert v_i \rvert^p ) ^\frac{1}{p} \\
# \lvert v \rvert _p &= E ^\frac{1}{p}
# \end{aligned}
# $$
#
# For example, the $L^p$-norm when $p=2$ is the Euclidean distance:
#
# $$
# \lvert x \rvert _2 = (x_1^2 + x_2^2 + \ldots + x_n^2) ^\frac{1}{2}
# $$
#
# Here is the derivation of mean from the definition of $E$. We see that it
# corresponds to the standard definition of mean:
#
# $$
# \begin{aligned}
# \arg \min_s \sum_i \lvert x_i -s \rvert^2 &= \text{mean} \\
# \frac{d}{ds} \sum_i (x_i -s)^2 &= 0 \\
# \sum_i \left[ \frac{d}{ds}(x_i-s)^2 \right] &= 0 \\
# \sum_i \left[ 2(x_i-s)(-1) \right] &= 0 \\
# \sum_i \left[ -2x_i + 2s \right] &= 0 \\
# \sum_i -2x_i + \sum_i 2s &= 0 \\
# -2 \sum_i x_i + 2s \lvert x \rvert &= 0 \\
# 2s \lvert x \rvert &= 2 \sum_i x_i \\
# s &= \frac{\sum_i x_i}{\lvert x \rvert}
# \end{aligned}
# $$
#
# Let's extend this to $p=4$, the L4 center:
#
# $$
# \begin{aligned}
# \arg \min_s \sum_i \lvert x_i - s \rvert^4 &= \text{L4 center} \\
# \frac{d}{ds} \sum_i (x_i - s)^4 &= 0 \\
# \sum_i \left[ \frac{d}{ds}(x_i-s)^4 \right] &= 0 \\
# \sum_i \left[ -4(x_i-s)^3 \right] &= 0 \\
# \sum_i (x_i-s)^3 &= 0 \\
# \sum_i \left[ x_i^3 -3x_i^2s + 3x_is^2 - s^3 \right] &= 0 \\
# \sum_i x_i^3 + \sum_i -3x_i^2s + \sum_i 3x_is^2 + \sum_i -s^3 &= 0 \\
# \sum_i x_i^3 -3s \sum_i x_i^2 + 3s^2 \sum_i x_i - s^3 \lvert x \rvert &= 0 \\
# \end{aligned}
# $$
#
# This is a cubic equation in terms of $s$ and cannot be simplified any
# further. Since $\sum_i (x_i - s)^3$ is strictly increasing in $s$ it will
# have a single real root. Therefore it is possible to solve the cubic using
# Cardano's formula without $\textit{casus irreducibilis}$ - intermediate and
# irreducible complex numbers arising from a negative discriminant.
#
# See also:
# http://www.johnmyleswhite.com/notebook/2013/03/22/
#   modes-medians-and-means-an-unifying-perspective/
# http://www.johnmyleswhite.com/notebook/2013/03/22/
#   using-norms-to-understand-linear-regression/
# https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions
# https://mathoverflow.net/questions/413246/
#   can-anything-be-said-about-the-roots-of-the-l4-center
# https://en.wikipedia.org/wiki/Cubic_equation
# https://en.wikipedia.org/wiki/Casus_irreducibilis
# https://www.youtube.com/watch?v=N-KXStupwsc

def l4_error(l, m):
    if not l:
        raise ValueError
    return sum((i-m)**4 for i in l)

# Cardano's solution to:
#   ax^3 + bx^2 + cx + d
# This will raise a ValueError if the discriminant is negative. A general
# solution can be had by switching to Python's complex square root function
# 'cmath.sqrt', but this will result in a complex solution. For our uses the
# discriminant will always be positive.
def cardano(a,b,c,d):
    # Python defaults to the imaginary component
    cbrt = lambda v: math.copysign(abs(v)**(1/3), v)
    p = (3*a*c - b**2) / (3*(a**2))
    q = (2*(b**3) - 9*a*b*c + 27*(a**2)*d) / (27*(a**3))
    d = (q**2)/4 + (p**3)/27
    t = cbrt(-q/2 + math.sqrt(d)) + cbrt(-q/2 - math.sqrt(d))
    x = t - b/(3*a)
    return x

def l4_center(l):
    if not l:
        raise ValueError
    a = -len(l)
    b = 3 * sum(l)
    c = -3 * sum(i**2 for i in l)
    d = sum(i**3 for i in l)
    x = cardano(a,b,c,d)
    return x


# Section 2.2
# Validate our implementations against Python's statistics module.
# This is also a good introduction to floating point error, see:
#   https://docs.python.org/3/tutorial/floatingpoint.html
#   https://blog.demofox.org/2017/11/21/floating-point-precision/
#   https://stackoverflow.com/a/53203428
# As well as the fractions module, see:
#   https://docs.python.org/3/library/fractions.html

def float64_precision(i):
    i = abs(i)
    if i == 0 or math.floor(math.log2(i)) < -1022:
        # subnormals: (smallest exponent) - (fractional bits)
        return 2**-(1022 + 52)
    e = math.floor(math.log2(i))
    r = (2**(e+1) - 2**e)
    return r / 2**52

# Floating-point (FP) errors accumulate over operations. When testing
# mean with integers there is only one floating-point operation, the
# final division by the list's length. However with floats first there
# is the sum of all floats. Internally the statistics module works with
# exact representations - fractions.Fraction - with a variable-type
# output conversion function. We only use floats. If we test only
# integers we know the statistics module will result in either an
# integer or a float, and that a conversion of that integer to a float
# will be equivalent to the result of our own mean. So that given only
# integers, this is valid:
#
#    assert mean(xs) == float(statistics.mean(xs))
#
# There is no valid possible conversion when testing floats. That said,
# we can test fractions instead, which provide an exact representation.
@given(st.lists(st.integers(), min_size=1))
def test_mean(xs):
    assert mean(xs) == float(statistics.mean(xs))

# The implementation of median in the statistics module does no
# conversion and so neither do we.
@given(st.lists(st.integers(), min_size=1))
def test_median(xs):
    assert median(xs) == statistics.median(xs)

# Floating point errors accumulate over operations. Mean has a single
# floating point operation - divions by the list's length. Variance has
# additionally distance from the mean, which is most likely a float.
# Unless the mean is fixed beforehand there is also a possibility that
# floating point deviation in the mean will cause some variance. The best
# approach is to use a type that has an exact representation in the range
# (output) of the function being tested, such as fractions.Fraction.
@given(st.lists(st.fractions(), min_size=1))
def test_variance(xs):
    assert variance(xs, mean(xs)) == statistics.pvariance(xs)

@given(st.lists(st.fractions(), min_size=1))
def test_stdev(xs):
    assert stdev(xs, mean(xs)) == statistics.pstdev(xs)

@given(st.lists(st.fractions(), min_size=1))
def test_sdm(xs):
    m = mean(xs)
    assert sdm(xs, m) == len(xs) * variance(xs, m)

# Testing this is tricky. As before floating-point error may invalidate the
# equality assertion. Unlike before fractions won't help, for two reasons.
# Our inputs are multiplied by floats, and fraction*float -> float.
# Irrational numbers can't be represented by fractions, and the normal PDF
# includes multiple irrational numbers. Standard deviation (sigma parameter)
# and variance must both be greater than zero, yet floating point errors
# given a small enough deviation may force the variance to zero. On the the
# other hand, too large a value can exceed the maximum value of a float,
# causing an OverflowError. One solution is to use floats within a limited
# range and compare to a limited precision.
@given\
    ( st.floats(min_value=-10**6, max_value=10**6)
    , st.floats(min_value=0, max_value=10**6, exclude_min=True)
    , st.floats(min_value=-10**6, max_value=10**6)
    )
def test_normal_pdf(mu, sigma, x):
    assume(sigma**2 > 0)
    n1 = normal_pdf(mu, sigma, x)
    n2 = statistics.NormalDist(mu, sigma).pdf(x)
    m = max(map(abs, (n1, n2)))
    assert abs(n1-n2) <= float64_precision(m) * 2**12

# We both use the same 'erf' function. Floating point error
# should not be an issue.
@given\
    ( st.floats(allow_nan=False, allow_infinity=False)
    , st.floats(min_value=0, exclude_min=True)
    , st.floats(allow_nan=False, allow_infinity=False)
    )
def test_normal_cdf(mu, sigma, x):
    n1 = normal_cdf(mu, sigma, x)
    n2 = statistics.NormalDist(mu, sigma).cdf(x)
    assert n1 == n2


# Section 2.3

# The next several functions are very well detailed at:
# https://medium.com/analytics-vidhya/jenks-natural-breaks-
#   best-range-finder-algorithm-8d1907192051
def jenks(ls, k, metric_distance=sdm, metric_center=mean):
    if not ls:
        raise ValueError
    if k > len(ls):
        raise ValueError
    if k < 1:
        raise ValueError
    ls = sorted(ls)
    ds = []
    for cs in list_splits(ls, k):
        d = 0
        for c in cs:
            d += metric_distance(c, metric_center(c))
        ds.append((d, cs))
    return min(ds, key=lambda i: i[0])

# Convert the result to the standard "breaks" format - a list of bounds. The
# first value is the lower bound (inclusive) of the first cluster. The
# subsequent values are the upper bounds (inclusive) of every cluster.
def breaks(cs):
    bs = []
    for ic,c in enumerate(cs):
        if ic == 0:
            bs.append(c[0])
        bs.append(c[-1])
    return bs

def jenks_breaks(ls, k):
    return breaks(jenks(ls, k)[1])


# Section 2.4

# Calculate the goodness of variance fit (GVF). The original list is
# separate from the clustering in case of removed outliers. Notice
# that the GVF always increases (reflecting a better fit) with the
# number of clusters, so it may not be the best metric.
def gvf(l, c):
    lv = variance(l, mean(l))
    cv = sum(variance(l, mean(l)) for l in c)
    return (lv - cv) / lv

def gvf_c(c):
    return gvf(sum(c, []), c)

# Generate the clustering from the data and the breaks list.
def break_list(l, b, key=None):
    ls = [[] for _ in b[1:]]
    for i in l:
        if key is None:
            ki = i
        else:
            ki = key(i)
        for ij,j in enumerate(b[1:]):
            if ki <= j:
                ls[ij].append(i)
                break
    return ls


# Section 2.5
# Let's validate our implementation of Jenks Natural Breaks Optimization
# algorithm using the Hypothesis unit tester (based on Haskell's QuickCheck).

# The 'jenkspy' module version 0.2.0 segfaults on large integers due to the
# (double*) cast, so here we limit the integer size. This is a good example
# of why unit testing is important, and a good exercise in debugging. Given
# a cluster count K and data of N values, the 'jenkspy' module doesn't
# support K=1 (single cluster) or K=N (every element its own cluster). Our
# implementation does, but for the sake of comparison restrict those cases.
@st.composite
def list_choose_clusters(draw, max_size=None, unique=False):
    xs = draw(st.lists
            ( st.integers(min_value=-10**3, max_value=10**3)
            , min_size=3, max_size=max_size, unique=unique
            ))
    k = draw(st.integers(min_value=2, max_value=len(xs)-1))
    return (xs, k)

# Ensure that the data we put in is the data we get
# out, albeit clustered and sorted.
@given(list_choose_clusters(max_size=15))
def test_jenks(t):
    assert sum(jenks(*t)[1], []) == sorted(t[0])

# The break list format doesn't work well with duplicate values - they might
# be shifted elsewhere, leaving unexpected empty clusters. So we ensure that
# the generated list is unique. There are multiple possible best solutions.
# So we compare variances rather than the lists themselves. Despite all this
# work, the 'jenkspy' module (version 0.2.0) still exhibits some incorrect
# results. I suspect an off-by-one error. Another great example of why
# unit-testing is important. Since this test always fails, it is commented
# out.
#@given(list_choose_clusters(max_size=15, unique=True))
#def test_jenks_breaks(t):
#    tf = lambda cs: sum(variance(c, mean(c)) for c in cs)
#    v1 = tf(break_list(t[0], jenks_breaks(*t)))
#    v2 = tf(break_list(t[0], jenkspy.jenks_breaks(*t)))
#    assert v1 == v2

# That said, we'll export an optimized C/Cython version.
# Make sure this is commented out when running the tests!
jenks_breaks = jenkspy.jenks_breaks


# Section 3.1
# Now that we have a clustering algorithm, we'd like to test it. But first we
# need some test data, and methods of visualizing that data. In the next
# section we'll generate the data by drawing random samples from a mixture of
# normal distributions (a Gaussian Mixture Model). In this section we introduce
# Kernel Density Estimation (KDE) as a means of visualizing that data. While
# Jenk's Natural Breaks is a clustering estimate, KDE is a density estimate.
# You can think of it as a smarter histogram. Visualize an unknown number of
# Normal PDF curves. The GMM is the sum of those curves, and KDE allows us to
# estimate that curve from a large enough sample size.
#
# Here is a great visual overview of KDE:
#   https://mathisonian.github.io/kde/
#
# Here is an applied overview with an excellent comparison to histograms:
#   https://jakevdp.github.io/PythonDataScienceHandbook/
#   05.13-kernel-density-estimation.html
#
# And the wikipedia article we'll use to build our implementation with a
# Gaussian kernel:
#   https://en.wikipedia.org/wiki/Kernel_density_estimation
#   https://en.wikipedia.org/wiki/Kernel_(statistics)
#
# Most interesting is that KDE can also be used for clustering.

# Explore KDE using a library for symbolic mathematics.
def kde_sympy():
    sympy.init_printing(use_unicode=True)
    m,s,x,h,p = sympy.symbols("mu sigma x h pi")
    # The kernel function K must be non-negative, symmetric, and have an
    # area of one. We'll use the standard Gaussian kernel, which in turn
    # is based on the standard Normal PDF, with mu=0 (mean) and sigma=1
    # (deviation). We've already implemented the normal pdf but now we
    # represent it symbolically.
    pdf = 1/(s*sympy.sqrt(2*p)) * sympy.exp((-1/2) * ((x-m)/s)**2)
    k = pdf.subs({m:0, s:1})
    # Derive the scaled kernel Kh = 1/h K(x/h) where h is the smoothing
    # parameter called the bandwidth. Think of the bandwidth as the size
    # of a histogram bucket.
    kh = k.subs({x:x/h})/h
    # Doesn't that look like we directly substituted deviation (sigma)
    # for bandwidth (h)? Let's check.
    kh_1 = pdf.subs({m:0, s:h})
    e = (sympy.simplify(kh - kh_1) == 0)
    # And that makes sense since standard deviation affects the width
    # of the normal PDF.
    print(f"[1] Normal PDF:\n{sympy.pretty(pdf)}")
    print(f"\n\n[2] Standard Normal PDF & Guassian Kernel:\n{sympy.pretty(k)}")
    print(f"\n\n[3] Scaled Gaussian kernel:\n{sympy.pretty(kh)}")
    print\
        ( f"\n\n[4] Standard Normal PDF with {sympy.pretty(h)} as "
          f"{sympy.pretty(s)}:\n{sympy.pretty(kh_1)}"
        )
    print(f"\n\nBandwidth is standard deviation ([3] == [4])? {e}")


# Section 3.2
# Implementations of gaussian KDE.

# Get the KDE at point 'x' given a set of samples 'ss' and a bandwidth 'h'
# defaulting the the standard normal PDF deviation.
def kde_gaussian(x, ss, h=1):
    if not ss:
        raise ValueError
    return sum(normal_pdf(0, h, x-s) for s in ss) / len(ss)

# To be finished once I can build scipy.
#@given
#def test_kde_gaussian():
#    pass

# For fast and efficient arrays you would normally use NumPy. See:
#   https://numpy.org/doc/stable/reference/generated/numpy.linspace
# Remember that, if 'endpoint' is true, the number of inter-element
# spans is one fewer than the number of elements.
def linspace(start, stop, count, endpoint=True, retstep=False):
    step = (stop - start) / max(1, count - (1 if endpoint else 0))
    xs = [start + i*step for i in range(count)]
    if retstep:
        return (xs,step)
    return xs

# Test against NumPy's implementation. The 'retstep' parameter differs
# as I prefer my implementation, so don't test that. Watch out for
# floating-point errors.
@given\
    ( st.integers(), st.integers()
    , st.integers(min_value=1, max_value=1e4)
    , st.booleans()
    )
def test_linspace(start, stop, count, endpoint):
    args = (start, stop, count, endpoint)
    a1 = linspace(*args)
    a2 = numpy.linspace(*args)
    ts = []
    for xi,x in enumerate(a1):
        if xi == 0:
            ts.append(0)
            continue
        px = float64_precision(x)*4
        ps = float64_precision(x-a1[xi-1])*xi
        pp = ts[xi-1]
        ts.append(max(px, ps, pp))
    assert all(abs(a1 - a2) <= ts)

# Given a set of samples, we want to plot the KDE in a range X0...Xn with
# a step size of Xs. The data points for plotting would be:
#   (X0,Y0), (X0+Xs*1,Y1), (X0+Xs*2,Y2), ..., (Xn,Yn).
# Notice that the samples need not lie within the range.
def kde_gaussian_range(start, stop, count, ss, h=1, endpoint=True):
    if h <= 0:
        raise ValueError
    xs = linspace(start, stop, count, endpoint=endpoint)
    ys = [0] * len(xs)
    if not ss:
        return list(zip(xs, ys))
    for xi,x in enumerate(xs):
        ys[xi] = kde_gaussian(x, ss, h)
    return list(zip(xs, ys))

# There could potentially be many data points with negligible KDE, and
# using the function above would be basically busy work. Instead, for
# each sample find the nearest data point, iterate outwards calculating
# the kernel for each adjacent data point until the threshold is reached.
def kde_gaussian_range_fast\
    ( start, stop, count, ss, h=1, endpoint=True, threshold=1e-4
    ):
    if h <= 0:
        raise ValueError
    xs,step = linspace(start, stop, count, endpoint=endpoint, retstep=True)
    if not xs:
        return xs
    ys = [0] * len(xs)
    for s in ss:
        # find the nearest index
        if step:
            xi, r = divmod(s-start, step)
            xi = int(xi)
            if r > step/2:
                xi += 1
            if xi < 0:
                xi = 0
            elif xi >= len(xs):
                xi = len(xs)-1
        else:
            xi = 0
        # run the kernel to either side
        for xj in range(xi, len(xs)):
            p = normal_pdf(0, h, xs[xj]-s) / len(ss)
            if p < threshold:
                break
            ys[xj] += p
        for xj in range(xi-1, -1, -1):
            p = normal_pdf(0, h, xs[xj]-s) / len(ss)
            if p < threshold:
                break
            ys[xj] += p
    return list(zip(xs, ys))

# Given (a, b), to find a centered range of size (b - a)*r:
# b + (b-a)*r/2
# a - (b-a)*r/2
#
# b:
# b + b*r/2 - a*r/2
# b*(1 + r/2) - a*r/2
#
# a:
# a - b*r/2 + a*r/2
# a*(1 + r/2) - b*r/2
def range_scale(a, b, ratio=1):
    ratio -= 1
    a1 = a*(1 + ratio/2) - b*ratio/2
    b2 = b*(1 + ratio/2) - a*ratio/2
    return (a1, b2)

@given(st.integers(), st.integers(), st.fractions())
def test_range_scale(a, b, ratio):
    a1, b1 = range_scale(a, b, ratio)
    assert (b1 - a1) == ratio * (b - a)

@st.composite
def range_choose_samples\
    ( draw
    , min_value=None, max_value=None
    , ratio=1
    , min_size=0, max_size=None
    ):
    if min_value is not None and max_value is not None:
        s1, s2 = range_scale(min_value, max_value, ratio)
        sr = [round(i) for i in sorted((s1,s2))]
    else:
        sr = [min_value, max_value]
    ss = draw(st.lists
        ( st.integers(min_value=sr[0], max_value=sr[1])
        , min_size=min_size, max_size=max_size
        ))
    return ss

@st.composite
def kde_gaussian_range_choose_args\
    ( draw
    , min_value=None, max_value=None
    , max_ssize=None, ratio=1
    , max_dsize=None
    ):
    d1 = draw(st.integers(min_value=min_value, max_value=max_value))
    d2 = draw(st.integers(min_value=min_value, max_value=max_value))
    ss = draw(range_choose_samples
        ( min_value=d1, max_value=d2
        , ratio=ratio
        , max_size=max_ssize
        ))
    count = draw(st.integers(min_value=0, max_value=max_dsize))
    h = draw(st.floats
        ( allow_infinity=False, allow_nan=False, width=16
        , min_value=2**-8, max_value=2**10
        ))
    endpoint = draw(st.booleans())
    return (d1, d2, count, ss, h, endpoint)

@given(kde_gaussian_range_choose_args
    ( -1*10**6, 1*10**6, 1*10**2, 1.5, 1*10**3))
def test_kde_gaussian_range_fast(args):
    r1 = kde_gaussian_range(*args)
    r2 = kde_gaussian_range_fast(*args, threshold=0)
    if not r1 or not r2:
        assert r1 == r2
        return
    x1,y1 = list(zip(*r1))
    x2,y2 = list(zip(*r2))
    assert x1 == x2
    assert all\
        ( abs(y1[i] - y2[i]) <= 2**6 * float64_precision(y1[i])
          for i in range(len(y1))
        )


# Section 3.3
# Now we have the GMM PDF. Notice that the peaks correspond to the centers of
# the individual distributions. Which implies the troughs can be used to split
# the distributions into individual clusters.

# First, we need to locate within our data the extrema: minima and maxima. For
# minima we use 'operator.lt' as comparator; for maxima we use 'operator.gt'.
# Returns indices.
def extrema_v1(ds, comparator, key=None):
    es = []
    for di,d in enumerate(ds[1:-1], start=1):
        ls = [ds[di-1], d, ds[di+1]]
        if key:
            ls = [key(i) for i in ls]
        if comparator(ls[1], ls[0]) and comparator(ls[1], ls[2]):
            es.append(di)
    return es

# Low probability events can exceed floating point precision, leading to
# plateaus rather than local extrema. This is a version of 'extrema' which
# can optionally allow plateaus. Single extrema are returned as 1-tuples
# while plateaus are returned as 2-tuples of (start_index, stop_index).
def extrema(ds, comparator, key=None, plateau=False):
    es = []
    el = len(ds)-1
    f = lambda i: ds[i] if key is None else key(ds[i])
    i = 1
    while i < el:
        if comparator(f(i), f(i-1)):
            j = None
            k = i
            while i < el:
                if comparator(f(i), f(i+1)):
                    j = i
                if not plateau:
                    break
                if f(i) != f(i+1):
                    break
                i += 1
            if j is not None:
                es.append((k,) if k==j else (k,j))
        i += 1
    return es

# Return the nearest index midpoints for the given extrema.
def extrema_midpoints(es):
    return [round(mean(i)) for i in es]

# Now we need to split the samples on the minima. If we sort both first
# the complexity (the n and m terms discarded) is:
#   O(nlogn + mlogm)
# If not the complexity is:
#   O(nm + mlogm)
# This is both simpler and faster for small 'm'. Where 'n' is the length
# of the list to be split and 'm' the length of the list containing the
# split points, and 'nlogn' is the complexity of Python's timsort.
def split_list(ls, ss, lte=True, fe=False):
    ss = sorted(ss)
    cs = [[] for _ in range(len(ss)+1)]
    for l in ls:
        for ji,j in enumerate(ss):
            if l < j or (lte and l == j):
                cs[ji].append(l)
                break
        else:
            cs[-1].append(l)
    if fe:
        cs = [c for c in cs if c]
    return cs

# If splitting the KDE instead of the samples we already have the indices.
# Implement a function to split by index rather than value.
def split_list_ix(ls, ix, lte=False):
    ix = [(i+1 if lte and i is not None else i) for i in ix]
    ix = [None] + ix + [None]
    return [ls[i:j] for i,j in zip(ix, ix[1:])]

# If splitting another list given a clustering, split by the size of each
# cluster. If 'bounded', ensure the total sizes do not exceed the size of
# the list. If 'remainder', return the remaining elements after the split
# even if that would be an empty list.
def split_list_size(ls, ss, bounded=True, remainder=True):
    if bounded and sum(ss) > len(ls):
        raise ValueError
    j = 0
    ss = [j:=(i+j) for i in ss]
    cs = split_list_ix(ls, ss)
    if not remainder:
        cs = cs[:-1]
    return cs

# Split a list given a matching list of zero-indexed labels.
def split_list_label(ds, ls):
    if len(ds) != len(ls):
        raise ValueError
    m = max(ls, default=0)
    cs = [[] for _ in range(m+1)]
    for d,l in zip(ds, ls):
        cs[l].append(d)
    return cs

@st.composite
def list_choose_split_indices(draw, max_size=None, ratio=1, nones=1):
    ls = draw(st.lists(st.integers(), max_size=max_size))
    i1,i2 = range_scale(-len(ls), len(ls), ratio=ratio)
    i1 = math.floor(i1)
    i2 = math.ceil(i1)
    ss = [None]*nones + list(range(i1,i2))
    ix = draw(st.lists(st.sampled_from(ss), max_size=int(len(ss)*ratio)))
    return (ls, ix)

@st.composite
def list_choose_split_values(draw, min_value=None, max_value=None):
    l1 = draw(st.integers(min_value=min_value, max_value=max_value))
    l2 = draw(st.integers(min_value=min_value, max_value=max_value))
    ls = draw(range_choose_samples(l1, l2, ratio=1))
    ss = draw(range_choose_samples(l1, l2, ratio=1.5))
    lte = draw(st.booleans())
    return (ls, ss, lte)

@st.composite
def list_choose_split_sizes(draw, ratio=0, max_min=1):
    ls = draw(st.lists(st.integers()))
    # The sum of the sizes:
    s1 = math.floor(len(ls) * (1-ratio))
    s2 = math.ceil(len(ls) * (1+ratio))
    s2 = max(s2, max_min)
    sr = draw(st.integers(min_value=s1, max_value=s2))
    # The number of sizes:
    s2 = math.ceil(sr * (1+ratio))
    sc = draw(st.integers(min_value=0, max_value=s2))
    # Sizes as indices:
    s2 = max(0, sc-1)
    si = draw(st.lists
        ( st.integers(min_value=0, max_value=sr)
        , min_size=s2, max_size=s2
        ))
    # Indices to sizes
    si = sorted(si + [0,sr])
    ss = [j-i for i,j in zip(si, si[1:])]
    return (ls, ss)

@st.composite
def list_choose_split_labels(draw, max_value=1e2, ratio=1):
    l = draw(st.integers(min_value=0, max_value=max_value))
    k = draw(st.integers(min_value=0, max_value=math.ceil(l*ratio)))
    ls = draw(st.lists
        ( st.integers(min_value=0, max_value=k)
        , min_size=l, max_size=l
        ))
    return ls

@given(list_choose_split_values(-1e3, 1e3))
def test_split_list(args):
    ls, ss, lte = args
    cs = split_list(*args, fe=False)
    assert sorted(sum(cs, [])) == sorted(ls)
    assert len(cs) == len(ss) + 1
    fs = [operator.le if lte else operator.lt] * len(ss)
    fs = fs + [operator.gt if lte else operator.ge]
    ss = sorted(ss)
    ss = ss + ss[-1:]
    for c,s,f in zip(cs, ss, fs):
        assert all(f(i,s) for i in c)

@given(list_choose_split_indices(ratio=1.5), st.booleans())
def test_split_list_ix(t, lte):
    s1 = split_list_ix(*t)
    s2 = [i.tolist() for i in numpy.split(*t)]
    assert s1 == s2

@given(list_choose_split_sizes(ratio=0.5))
def test_split_list_size(t):
    ls, ss = t
    ll = len(ls)
    cs = split_list_size(ls, ss, remainder=False, bounded=False)
    ct = [j for i in cs for j in i]
    assert len(cs) == len(ss)
    assert ct == ls[:sum(ss)]
    for c,s in zip(cs, ss):
        s = min(ll, s)
        ll -= s
        assert len(c) == s
    rs = split_list_size(ls, ss, remainder=True, bounded=False)
    assert cs == rs[:-1]
    assert ct + rs[-1] == ls

@given(list_choose_split_labels(ratio=1.5))
def test_split_list_label(ls):
    ds = list(range(len(ls)))
    cs = split_list_label(ds, ls)
    assert len(cs) == max(ls, default=0) + 1
    assert sorted(sum(cs, [])) == ds
    for i,c in enumerate(cs):
        assert all(ls[j] == i for j in c)


# Section 4.1
# K-means algorithms such as Jenks are susceptible to outliers because
# means are easily influenced by extreme values. The simplest way to
# handle outliers is to remove them beforehand. Removing outliers during
# clustering is an active topic of research not because the algorithms
# are much more complicated, but because they may run much slower. If
# you know enough about the expected clustering, you may be able to
# handle outliers after clustering. This section covers outlier
# detection without clustering.


# Percentiles are a way of analyzing data not by its value but by its
# index. For example, the median is the 50% percentile. Unlike means,
# percentiles are less susceptible to outliers.
#
# When a percentile falls between two data points, use a linear
# interpolation.  Given a two points 'i < j' and the fractional
# part of the index '0 <= r <= 1', define the percentile to be:
#   i + (j-i) * r   # two-point equation of a line
def percentile(xs, p, sort=True):
    if not xs:
        raise ValueError
    if p < 0 or p > 100:
        raise ValueError
    if sort:
        xs = sorted(xs)
    xi = p / 100 * (len(xs) - 1)
    xi,ri = divmod(xi, 1)
    xi = int(xi)
    p = xs[xi]
    if ri:
        p += (xs[xi+1] - xs[xi]) * ri
    return p

@given\
    ( st.lists
        ( st.integers(min_value=-2**31, max_value=2**32-1)
        , min_size=1
        )
    , st.integers(min_value=0, max_value=100)
    )
def test_percentile(xs, p):
    p1 = percentile(xs, p)
    p2 = numpy.percentile(xs, p)
    assert abs(p1-p2) <= 2**6 * float64_precision(p1)

# Given X, a normal random variable, 99.7% of observations are contained
# within +-3 standard deviations. It might be tempting to classify
# everything beyond 3 standard deviations as an outlier. This has two
# downsides:
# 1. 0.3% of all normal observations will be incorrectly classified as an
#    outlier.
# 2. A large outlier may significantly shift the mean and increase the
#    standard deviation. As a worst case the outlier may be included a
#    significant portion of likely observations excluded.
# The interquartile range (IQR) is a method of determing the center of the
# majority of the data without using means. See:
#   https://en.wikipedia.org/wiki/Interquartile_range
# Let's find out where the standard 1.5*IQR factor comes from. We're using
# the inverse CDF of the Normal distribution from the statistics module
# since Python doesn't provide the inverse erf function necessary to write
# our own inverse CDF. While the CDF gets you the probability 'p' of an
# observation less than or equal to 'x', the inverse cdf gets 'x' from 'p'.
#
# Try a random non-standard Normal distribution. The 1.5 factor is designed
# to cover 99.3% of observations given a normal distribution.  There are
# different factors for different distributions. As for why 99.3% rather
# than the more common 99.7%, I think that's because 1.5 is a nice round
# number.
def iqr_explore(mu, sigma):
    n = statistics.NormalDist(mu, sigma)
    # Get the middle 50% in standard deviations.
    q1, q3 = n.inv_cdf(.25), n.inv_cdf(.75)
    # And the IQR
    iqr = q3 - q1
    # Apply the 1.5 multiplier to get the "whiskers" (box plot)
    w1 = q1 - 1.5*iqr
    w3 = q3 + 1.5*iqr
    # What's the probability of everything within the whiskers?
    p = n.cdf(w3) - n.cdf(w1)
    print\
        ( f"Given N(μ={mu}, σ={sigma}):\n\n"
          f"IQR defined as 50% is {iqr:.4f}, "
            f"or {iqr/sigma:.4f} standard deviations\n"
          f"1.5*IQR range is {w3-w1:.4f}, "
            f"or {(w3-w1)/sigma:.4f} standard deviations\n"
          f"1.5*IQR range is {p*100:.2f}% of observations"
        )

def ipr(xs, lower, upper, coverage=None, ratio=None, sort=True):
    if coverage is None and ratio is None:
        raise TypeError
    if coverage is not None and ratio is not None:
        raise TypeError
    if lower > upper:
        raise ValueError
    if coverage is not None and not (0 < coverage < 1):
        raise ValueError
    if coverage is not None and not (0 < lower < 1):
        raise ValueError
    if coverage is not None and not (0 < upper < 1):
        raise ValueError
    if coverage is not None:
        n = statistics.NormalDist(0,1)
        p1 = n.inv_cdf(lower)
        p2 = n.inv_cdf(upper)
        ipr = p2 - p1
        gap = (1 - coverage)/2
        w1 = n.inv_cdf(gap)
        w2 = n.inv_cdf(gap + coverage)
        iwr = w2 - w1
        # ipr + ipr*ratio*2 = iwr
        ratio = (iwr/ipr - 1) / 2
    if sort:
        xs = sorted(xs)
    p1 = percentile(xs, lower, sort=False)
    p2 = percentile(xs, upper, sort=False)
    pr = p2 - p1
    return ((p1 - ratio*pr, p2 + ratio*pr), (p1, p2), ratio)

def iqr(xs, sort=True):
    return ipr(xs, 25, 75, ratio=1.5, sort=sort)


# TODO:
#   KDE: 2 implementations
#   KDE: clustering w/ local minima
#   SD & IQR outlier detection
#   integrate above into GMM plotting
#   clusters_em_var
#       support outliers
#       better name & documentation
#   plotting of real-world morse data
#       visualize outlier detection of extended clusters_em_var
#   Move silhouette functions before first plotting
#   Remove jenks_median


# Section 5.1
# This section uses the optional silhouette methods (below).

# Let's generate some graphs to describe the "elbow" method of selecting
# the cluster count (k). The number of distributions is unknown, lets
# try to derive it. We also integrate KDE and outlier statistics. After
# generating the data, convert it to Pandas dataframes for plotting. For
# plotting shared domains the column names must match.

# Round to a nearest multiple of 'to'. A positive 'to' rounds away from
# zero, a negative 'to' rounds towards zero. Avoids integer division and
# resulting floating-point errors.
def round_dir(n, to):
    if to == 0:
        raise ValueError
    return n - n % (int(math.copysign(1, n)) * -1 * to)

# Round the range to nearest 'to' from the center of the range, outwards
# if positive, inwards if negative.
def range_round(a, b, to):
    a = a - a % to
    b = b - b % -to
    return (a,b)

@given(st.fractions(), st.fractions())
def test_round_dir(n, to):
    assume(to != 0)
    op = math.floor if n<0 else math.ceil
    assert op(n/to)*to == round_dir(n, to)

@given(st.fractions(), st.fractions(), st.fractions())
def test_range_round(a, b, to):
    assume(to != 0)
    a1, b1 = range_round(a, b, to)
    assert math.floor(a/to)*to == a1
    assert math.ceil(b/to)*to == b1

# Embed each chart within the first empty element of class "vega-vis" until
# no more elements are available. The filename is a required keyword-only
# parameter. The style CSS and container HTML may be overridden.
def chart_embed(*charts, filename, style=None, container=None):
    specs = []
    for c in charts:
        s = io.StringIO()
        c.save(s, format="json")
        specs.append(s.getvalue())
    specs = "\n, ".join(specs)
    specs = f"[ {specs}\n]"
    specs = textwrap.indent(specs, " "*4)

    if style is None:
        style = """
        .container {
          overflow-x: hidden;
        }
        .vega-vis {
          width: 90%;
          min-width: 600px;
          margin-left: auto;
          margin-right: auto;
          display: block !important;
        }
        .vega-embed .marks {
          margin-left: 50%;
          transform:
            translateX(-50%)
            translateX(calc(var(--vega-action-padding,38px)/2));
        }
        """
    style = textwrap.indent(textwrap.dedent(style).strip(), " "*2)

    if container is None:
        container = """
        <div class="container"><div class="vega-vis"></div></div>
        """
    container = textwrap.dedent(container).strip()

    template = textwrap.dedent("""
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
      <script src="https://cdn.jsdelivr.net/npm/vega-lite@4"></script>
      <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
      <style>
    {style}
      </style>
    </head>
    <body>

    {container}

    <script type="text/javascript">
      let specs =
    {specs};
      let es = document.querySelectorAll(".vega-vis:empty");
      for (let i = 0; i < Math.min(specs.length, es.length); i++) {{
        vegaEmbed(es[i], specs[i])
          .then((result) => {{ }})
          .catch(console.error);
      }}
    </script>
    </body>
    </html>
    """).strip()

    template = template.format\
        ( style=style
        , container=container
        , specs=specs
        )

    with open(filename, "w") as fp:
        fp.write(template)

# The builtin Vega/Altair boxplot trims the whiskers to the nearest
# enclosed data, which fails to provide a sense of scale. The builtin
# boxplot is also not interactive. Provide our own boxplot.
def chart_box(data_iqr, size, accent_color="#ff7f0e"):
    # Lower layers first.
    chart = (
        # Intersection of whisker range and data range.
          altair.Chart(data_iqr)
            . mark_rule()
            . encode(x="w1d", x2="w2d")
        + altair.Chart(data_iqr)
            . mark_tick()
            . encode(x="w1d")
        + altair.Chart(data_iqr)
            . mark_tick()
            . encode(x="w2d")

        # Difference of data range from whisker range, left side.
        + altair.Chart(data_iqr)
            . mark_rule(color=accent_color, strokeDash=[10,8])
            . encode(x="w1", x2="w1d")
        + altair.Chart(data_iqr)
            . mark_tick(color=accent_color)
            . encode(x="w1")

        # Difference of data range from whisker range, right side.
        + altair.Chart(data_iqr)
            . mark_rule(color=accent_color, strokeDash=[10,8])
            . encode(x="w2d", x2="w2")
        + altair.Chart(data_iqr)
            . mark_tick(color=accent_color)
            . encode(x="w2")

        # Box and median.
        + altair.Chart(data_iqr)
            . mark_bar(size=size)
            . encode(x="b1", x2="b2")
        + altair.Chart(data_iqr)
            . mark_tick(color=accent_color, size=size, thickness=2)
            . encode(x="median")
        )
    chart = chart.encode(tooltip=
        [ altair.Tooltip("max", title="Max")
        , altair.Tooltip("w2", title="Whisker Max")
        , altair.Tooltip("w2d", title="Whisker Data Max")
        , altair.Tooltip("b2", title="Q3")
        , altair.Tooltip("median", title="Median")
        , altair.Tooltip("b1", title="Q1")
        , altair.Tooltip("w1d", title="Whisker Data Min")
        , altair.Tooltip("w1", title="Whisker Min")
        ])
    return chart

def data_kde(data, h, extension=None, step=None):
    # Calculate the gaussian KDE with the given standard deviation or
    # bandwidth. Extend the bounds by 3 standard deviations for aesthetics.
    # This isn't necessary for calculating extrema - as long as all the
    # samples are contained within the range. Derive the step size from the
    # standard deviation. Once again not neccessary, but useful for matching
    # the step size to the level of detail.
    if extension is None:
        extension = h*3
    if step is None:
        step = h // 5
    d_min = min(data)
    d_max = max(data)
    k_min = d_min - extension
    k_max = d_max + extension
    k_count = int((k_max - k_min)//step + 1)
    data_kde = kde_gaussian_range(k_min, k_max, k_count, data, h=h)

    # Calculate the centers - the local maxima.
    kde_cs_ix = extrema_midpoints(extrema
        (data_kde, operator.gt, key=lambda t: t[1]))

    # Calculate the splits - the local minima.
    kde_ss_ix = extrema_midpoints(extrema
        (data_kde, operator.lt, key=lambda t: t[1], plateau=True))

    # Split the KDE.
    data_kde = split_list_ix(data_kde, kde_ss_ix)

    # Convert to Pandas dataframes.
    data_kde = pandas.concat\
        ( ( pandas.DataFrame(c, columns=["x", "KDE"]).assign(h=h)
            for c in data_kde
          )
        , keys=range(1, len(data_kde)+1)
        , names=["k"]
        ).reset_index(level=0)

    kde_cs = pandas.DataFrame(data_kde.iloc[kde_cs_ix]).reset_index(drop=True)
    kde_ss = pandas.DataFrame(data_kde.iloc[kde_ss_ix]).reset_index(drop=True)

    return (data_kde, kde_cs, kde_ss)

def data_iqr(data, sort=True):
    if sort:
        data = sorted(data)

    # Provide some IQR statistics. Much depends on the
    # data already being sorted.
    (w1, w2), (b1, b2), _ = iqr(data, sort=False)
    m = median(data, sort=False)
    wrange = [d for d in data if w1 <= d <= w2]
    data_iqr =\
        { "w1": [w1], "w1d": wrange[:1]
        , "b1": [b1], "b2": [b2]
        , "w2": [w2], "w2d": wrange[-1:]
        , "min": data[:1], "max": data[-1:]
        , "median": [m]
        }

    data_iqr = pandas.DataFrame(data_iqr)

    return data_iqr

def data_cluster_and_score(data, k, sort=True):
    if sort:
        data = sorted(data)

    data_ci = {}
    data_gvf = []
    data_sil = []

    # Cluster the data into 1..k clusters. For each clustering, calculate
    # the available metrics representing the the appropriateness of the
    # clustering. For a single cluster, GVF is known to be zero, and
    # silhouette is undefined but conventionally set to zero.
    for k in range(1, k+1):
        if k == 1:
            c = [data]
        else:
            c = break_list(data, jenks_breaks(data, k))
            # The clusters will later be recombined with the data, so
            # the individual elements must be aligned. The nice thing
            # about Jenks is if the input is sorted, the clusters will
            # be returned in the same order. That said, this is critical
            # enough to double-check.
            assert [j for i in c for j in i] == data

        n = [i[0] for i in c]
        x = [i[-1] for i in c]
        m = [mean(i) for i in c]
        v = [variance(i,j) for i,j in zip(c,m)]
        s = [stdev(i,j) for i,j in zip(c,m)]
        l = [len(i) for i in c]
        i = [i for i in range(1, len(l)+1)]

        data_ci[k] =\
            { "min"     : n
            , "max"     : x
            , "mean"    : m
            , "var"     : v
            , "stdev"   : s
            , "len"     : l
            , "ki"      : i
            }

        if k == 1:
            data_gvf.append((1,0))
            data_sil.append((1,0))
            continue

        t = (k, gvf(data, c))
        data_gvf.append(t)

        t = (k, silhouette_data(silhouette(c)))
        data_sil.append(t)

    # Given the data, for each k-clustering mark the cluster in which the
    # data appears. There is a column per k-clustering, each indexed by
    # the data. Since this incorporates the original data as the index,
    # there is no need for the original data to have its own dataframe.
    data =\
        ( pandas.DataFrame
            ( { k : [e for l,i in zip(ci["len"],ci["ki"]) for e in [i]*l]
                for k,ci in data_ci.items()
              }
            , index=data
            )
        . rename_axis("x")
        . reset_index().reset_index()
        . melt(["index","x"], var_name="k", value_name="ki")
        )

    # To go back to individual clusterings as nested lists, we'd have to
    # run this lengthy chain of operations. So we score the clusters
    # before converting them to dataframes, which is why clustering and
    # scoring happens together.
    #
    # for k in range(1, k+1):
    #     ( data.groupby("k")
    #     . get_group(k)
    #     . groupby("ki")["x"]
    #     . agg(list).tolist()
    #     )

    data_gvf = pandas.DataFrame(data_gvf, columns=["k","GVF"])
    data_sil = pandas.DataFrame(data_sil, columns=["k","Silhouette"])
    data_metrics =\
        ( pandas.merge(data_gvf, data_sil, on="k")
        . melt(id_vars=["k"], var_name="type")
        )

    data_ci = pandas.concat\
        ( pandas.DataFrame(v).assign(k=k)
          for k,v in data_ci.items()
        )

    return (data, data_metrics, data_ci)

def data_gen_gmm():
    data = []

    # Pick a random cluster count within these bounds.
    kb = (2,8)
    kr = random.randint(*kb)
    # Randomly pick the cluster centers from a sequence with a given step
    # (200). The size of the sequence is given in relation to the maximum
    # cluster count (5x) to ensure the random distribution becomes apparent.
    # Randomly shift each cluster by up to a given amount, giving a minimum
    # cluster separation of 'step - amount' (100).
    cs_gap = 200
    cs_max = 5*kb[1]
    cs_off = 100
    cs_min = cs_gap - cs_off
    cs =\
        [ i + random.randint(0,cs_off)
          for i in random.sample(range(0,cs_max*cs_gap,cs_gap), kr)
        ]

    # The number of samples for each cluster is drawn from a range with uniform
    # distribution. For normally distributed clusters to be distinguishable the
    # standard deviation for each should be less than half the minimum cluster
    # separation - the 68/95/99.7 rule above (1/3).
    std = cs_min / 3
    for c in cs:
        for i in range(random.randint(20,100)):
            data.append(int(random.gauss(c, std)))
    data.sort()

    return (data, std, kr, kb)

def gmm_data():
    data, std, kr, kb = data_gen_gmm()

    # Set the bandwidth to the standard deviation of our data. We've
    # glossed over selecting the optimal bandwidth, but we can cheat
    # since we know the standard deviation of the clusters.
    kde, kde_cs, kde_ss = data_kde(data, std)

    # Demonstrate KDEs with different bandwidths.
    kdes = pandas.concat(data_kde(data, std*i)[0] for i in [6,3,1/3,1/6])

    iqr = data_iqr(data, sort=False)

    # Cluster the data into 1...N clusters where N is twice the
    # maximum bound of cluster count.
    data, metrics, ci = data_cluster_and_score(data, 2*kb[1], sort=False)

    return (data, std, kr, kb, kde, kde_cs, kde_ss, kdes, iqr, metrics, ci)

def gmm_chart(filename="chart-gmm.html"):
    ( data, std, kr, kb
    , kde, kde_cs, kde_ss, kdes
    , iqr, metrics, ci
    ) = gmm_data()

    # The initial data domain.
    dstep = 100
    domain = range_round\
        ( *range_scale
            ( min(kde["x"])
            , max(kde["x"])
            , ratio=1.05
            )
        , dstep
        )

    # The interactive data domain selection.
    selection_domain = altair.selection_interval\
        ( bind="scales"
        , encodings=["x", "y"]
        )

    c_freq = chart_freq\
        ( data
        , domain, selection_domain
        , title=f"Frequencies ({kr} distributions, σ={std:.2f})"
        )

    c_clusters = chart_clusters\
        ( ci
        , domain, selection_domain
        , title=f"Clusterings ({kr} distributions)"
        )

    c_iqr = chart_iqr\
        ( data, iqr
        , domain, selection_domain
        )

    c_kde = chart_kde\
        ( kde, kde_cs, kde_ss, kdes
        , domain, selection_domain
        , title=f"KDE ({kr} distributions, h={std:.2f})"
        )

    c_metrics = chart_metrics\
        ( metrics
        , title=
            "Jenk's Natural Breaks - "
            f"Validation Metrics ({kr} distributions)"
        )

    chart =\
        ( altair.vconcat
            ( c_clusters
            , c_freq
            , c_iqr
            , c_kde
            , c_metrics
            )
        . resolve_scale(color="independent")
        . configure_scale(continuousPadding=1)
        )

    if filename is None:
        return chart

    chart_embed(chart, filename=filename)

def chart_freq\
    ( data
    , domain, selection_domain
    , title="Frequencies"
    , color=None
    , mean=None
    ):
    chart_frequencies =\
        ( altair.Chart
            ( data=data
            , title=title
            )
        . mark_tick(height=80)
        . encode
            ( x=altair.X
                ( "x"
                , title="x"
                , scale=altair.Scale
                    ( zero=False
                    , nice=False
                    , domain=domain
                    )
                )
            )
        . properties
            ( width="container"
            , height=100
            )
        . add_selection(selection_domain)
        )

    if color is not None:
        chart_frequencies =\
            ( chart_frequencies
            . encode(color=altair.Color(color))
            )

    if mean is not None:
        chart_mean =\
            ( altair.Chart(data=data)
            . transform_aggregate(mean_x="mean(x)", groupby=[mean])
            . mark_tick(thickness=1, height=40, color="red")
            . encode(x=altair.X("mean_x:Q", title="mean(x)"))
            )
        chart_frequencies = chart_frequencies + chart_mean

    return chart_frequencies

def chart_clusters(data, domain, selection_domain, title="Clusterings"):
    selection_kfocus = altair.selection_multi\
        ( on="click"
        , fields=["k"]
        )

    tooltip =\
        [ "min", "max"
        , altair.Tooltip("len", title="count")
        , altair.Tooltip("ki", title="cluster")
        , altair.Tooltip("mean", format="0.2f")
        , altair.Tooltip("var", format="0.2f")
        , altair.Tooltip("stdev", format="0.2f")
        ]

    opacity = altair.condition\
        ( selection_kfocus
        , altair.value(1.0)
        , altair.value(0.2)
        )

    chart_clusters =\
        ( altair.Chart
            ( data=data
            , title=title
            )
        . transform_calculate
            ( x="datum.min"
            )
        . mark_bar()
        . encode
            ( x=altair.X
                ( "x:Q"
                , title="x"
                , scale=altair.Scale
                    ( nice=False
                    , domain=domain
                    )
                )
            , x2="max"
            , y="k:O"
            , color=altair.Color("ki:N", legend=None)
            , opacity=opacity
            , tooltip=tooltip
            )
        . properties
            ( width="container"
            , height={"step": 30}
            )
        . add_selection(selection_domain)
        )

    chart_singletons =\
        ( altair.Chart(data=data)
        . transform_filter(altair.datum.min == altair.datum.max)
        . transform_calculate(x="datum.min")
        . mark_tick(thickness=2, height=30*0.9)
        . encode
            ( x="x:Q"
            , y="k:O"
            , color="ki:N"
            , opacity=opacity
            , tooltip=tooltip
            )
        . add_selection(selection_kfocus)
        )

    return (chart_clusters + chart_singletons)

def chart_cluster_regression\
    ( data
    , domain, selection_domain
    , title="Cluster Regression"
    , field_x="mean", field_y="stdev"
    , scaletype_y="linear", scaletype_x="linear"
    , rmethod="exp"
    , columns=3
    ):

    data = data[data["k"] >= 3]

    chart_cluster_info =\
        ( altair.Chart(data=data)
        . mark_point()
        . encode
            ( x=altair.X
                ( field_x
                , scale=altair.Scale
                    ( nice=False
                    , domain=domain
                    , type=scaletype_x
                    )
                )
            , y=altair.Y
                ( field_y
                , scale=altair.Scale(type=scaletype_y)
                )
            , color=altair.Color
                ( "k:N"
                , legend=None
                )
            , tooltip=
                [ "ki", "len", "min", "max"
                , altair.Tooltip("mean", format="0.2f")
                , altair.Tooltip("stdev", format="0.2f")
                , altair.Tooltip("var", format="0.2f")
                ]
            )
        )

    # The smallest possible value for standard deviation or variance is zero,
    # for which logarithm is undefined. The workaround for regressions is to
    # add 1 before regressing and subtract 1 afterwards.
    chart_regression =\
        ( altair.Chart(data=data)
        . transform_calculate
            ( var="datum.var + 1"
            , stdev="datum.stdev + 1"
            )
        . transform_regression
            ( on=field_x
            , regression=field_y
            , groupby=["k"]
            , method=rmethod
            , params=False
            )
        . transform_calculate
            ( var="datum.var - 1"
            , stdev="datum.stdev - 1"
            )
        . mark_line()
        . encode
            ( x=field_x
            , y=field_y
            , color="k:N"
            )
        )

    chart_text =\
        ( altair.Chart(data=data)
        . transform_calculate
            ( var="datum.var + 1"
            , stdev="datum.stdev + 1"
            )
        . transform_regression
            ( on=field_x
            , regression=field_y
            , groupby=["k"]
            , method=rmethod
            , params=True
            )
        . transform_calculate
            ( rSquaredText="'R² = ' + round(datum.rSquared*100)/100"
            )
        . mark_text
            ( align="left"
            , baseline="top"
            , fontSize=13
            , color="brown"
            )
        . encode
            ( x=altair.value(10)
            , y=altair.value(10)
            , text="rSquaredText:N"
            )
        )

    # The row title within each facet can be configured with:
    # facet=altair.Facet
    #   ( ...
    #   , title=title
    #   , header=altair.Header(titleFontSize=13)
    #   )
    chart_cluster_info =\
        ( (chart_cluster_info + chart_regression + chart_text)
        . properties
            ( width="container"
            , height=300
            )
        . facet
            ( facet=altair.Facet("k:N")
            , columns=columns
            )
        . add_selection(selection_domain)
        . properties
            ( title={"text":title, "anchor":"middle"}
            )
        )

    # Unfortunately container width sets the width of each facet. We'll
    # have to patch the generated Vega to divide the width by the number
    # of columns.  This is fine as long as long as the autosize type is
    # not "fit", which may overwrite the width signals. The default type
    # is "pad". An example of setting autosize:
    #
    #. properties
    #    ( autosize=altair.AutoSizeParams
    #        ( resize=True
    #        , type="fit"
    #        , contains="padding"
    #        )
    #    )
    #
    # The default spacing between facets is 20 pixels, which also needs to be
    # removed.
    #
    # JSON Pointer, used by JSON Patch, doesn't support array access by child
    # attributes. We don't know the final index of the width signal since the
    # chart may be embedded, so let the user provide it along with extra space.

    def patch(index, extra=0):
        width = "isFinite(containerSize()[0]) ? containerSize()[0] : 400"
        width = "((({}) - {}) / {}) - {}".format\
            (width, 20*(columns-1), columns, extra)

        patch =\
            [ { "op": "replace"
              , "path": f"/signals/{index}/init"
              , "value": width
              }
            , { "op": "replace"
              , "path": f"/signals/{index}/on/0/update"
              , "value": width
              }
            ]

        return patch

    return (chart_cluster_info, patch)

def chart_iqr\
    ( data, data_iqr
    , domain, selection_domain
    , title="IQR @ 25% - 75% with 1.5x whiskers"
    ):
    chart_iqr =\
        ( altair.Chart
            ( data=data[~data["x"].between(*data_iqr.loc[0, ["w1","w2"]])]
            , title=title
            )
        . mark_point()
        . encode
            ( x=altair.X
                ( "x"
                , scale=altair.Scale
                    ( nice=False
                    , domain=domain
                    )
                , title="x"
                )
            )
        . properties
            ( width="container"
            , height=80
            )
        . add_selection(selection_domain)
        + chart_box(data_iqr, size=60)
        )

    return chart_iqr

def chart_kde\
    ( kde, kde_cs, kde_ss, kdes
    , domain, selection_domain
    , title="KDE"
    ):
    chart_kde =\
        ( altair.Chart
            ( data=kde
            , title=title
            )
        . mark_line()
        . encode
            ( x=altair.X
                ( "x"
                , scale=altair.Scale
                    ( nice=False
                    , domain=domain
                    )
                )
            , y=altair.Y("KDE", title="p")
            , color=altair.Color("k:N", legend=altair.Legend
                ( orient="top-right"
                , fillColor="#f9f9f9"
                , padding=10
                ))
            , tooltip=[altair.Tooltip("h", format="0.2f")]
            )
        . properties
            ( width="container"
            , height=300
            )
        . add_selection(selection_domain)
        )

    chart_kde_cs =\
        ( altair.Chart(data=kde_cs)
        . mark_point(size=100, filled=True)
        . encode
            ( x="x"
            , y="KDE"
            , color=altair.Color("k:N")
            , tooltip=[altair.Tooltip("x", format="0.2f", title="μ")]
            )
        )

    chart_kde_ss =\
        ( altair.Chart(data=kde_ss)
        . mark_point
            ( size=50
            , filled=True
            , color="black"
            , opacity=0.5
            )
        . encode
            ( x="x"
            , y="KDE"
            , tooltip=[altair.Tooltip("x", format="0.2f", title="split")]
            )
        )

    chart_kdes =\
        ( altair.Chart(data=kdes)
        . mark_line(opacity=0.35, strokeWidth=1.5)
        . encode\
            ( x="x"
            , y="KDE"
            , color=altair.Color("h:N", legend=None)
            , tooltip=[altair.Tooltip("h", format="0.2f")]
            )
        )

    chart_kde = (chart_kde + chart_kde_cs + chart_kde_ss)
    chart_kde = (chart_kdes + chart_kde).resolve_scale(color="independent")

    return chart_kde

def chart_metrics(metrics, title="Clustering Metrics"):
    chart_metrics_base =\
        ( altair.Chart
            ( data=metrics
            , title=title
            )
        . encode
             ( x=altair.X("k", axis=altair.Axis(tickMinStep=1))
             , y="value"
             , color=altair.Color("type", legend=altair.Legend
                ( orient="bottom-right"
                , fillColor="#f9f9f9"
                , padding=10
                ))
             , tooltip=
                [ "k"
                , altair.Tooltip("value", format="0.4f", title="v")
                ]
             )
        )

    chart_metrics =\
        ( chart_metrics_base.mark_point(size=200, filled=True)
        + chart_metrics_base.mark_line()
        ).properties(width="container").interactive()

    return chart_metrics


# Section 6.1
# Before we examine Morse data in detail we should familiarize ourselves with
# characteristics of the Jenk's Natural Breaks algorithm. The goal of the
# algorithm is to find a clustering which minimizes a given metric. The
# simplest implementation, like ours above, is to enumerate over all possible
# clusterings. This is a slow approach with factorial-time complexity, but it
# does give an optimal answer. Another possibility is to start with an
# arbitrary clustering and use a given heuristic, such as GVF, to move points
# between clusterings until a local minimum is found. While much faster, this
# does not guarantee an optimal solution. This variant is known as the
# Jenks-Caspall algorithm and was introduced in 1971 [1] for chloropleth maps.
# In 1958 W.D. Fisher demonstrated that an optimal solution can be calculated
# in quadratic time without examining all possible clusterings [2]. This was
# popularized for cartography by Jenks in 1977 and is now known as the
# Fisher-Jenks algorithm [3]. This has since seen various improvements leading
# to the optimal and linear-time Ckmeans algorithm [4].
#
# Jenks clustering can be seen as a 1-dimensional specializing of K-means
# clustering [5]. Only 1-dimensional data can be sorted; Jenks takes advantage
# of this to run faster. In turn k-means can be interpreted as a specialization
# of expectation-maximization for Guassian Mixture Models (GMM EM), where the
# weight and variance of each Gaussian is fixed and equal [6]. As such it
# inherits all the limitations of the k-means algorithm [7]:
#
# 1. Clusters are spherical (not much of a restriction in 1D).
# 2. Clusters all have the same variance.
# 3. Clusters all have the same weight; the same number of elements.

# This is incredibly restrictive; however it runs much faster than GMM EM. The
# challenge will be to see if we can adapt it to analyze Morse data in real
# time. There is no hard and fast value to these limitations. The further
# removed a point from its mean the greater the cost to include it in the
# cluster. This ensures clusters are islands of similarity within a sea of
# dissimilar "gaps". The restrictions only become apparent with the division
# between clusters is borderline. Otherwise they can be fudged.

# [1] Jenks, G.F. & Caspall, F.C. (1971).
#     Error on Choroplethic Maps: Definition, Measurement, Reduction
# [2] Fisher, W. D. (1958).
#     On grouping for maximum homogeneity
# [3] Jenks, G.F. (1977).
#     Optimal Data Classification for Choropleth Maps
# [4] Wang, H. & Song, M. (2011).
#     Ckmeans.1d.dp: Optimal k-means Clustering in One Dimension by Dynamic
#     Programming
# [5] https://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization#See_also
# [6] https://en.wikipedia.org/wiki/K-means_clustering#Gaussian_mixture_model
# [7] http://varianceexplained.org/r/kmeans-free-lunch/
#
# For more information, see:
# * http://ces.iisc.ernet.in/hpg/envis/doc98html/ecoty9952.html
# * https://www.geodms.nl/
#   Fisher%27s_Natural_Breaks_Classification_complexity_proof
# * https://stackoverflow.com/questions/45555371/
#   is-jenks-natural-breaks-deterministic
# * https://stats.stackexchange.com/questions/133656/
#   how-to-understand-the-drawbacks-of-k-means/249288#249288

# Now that we know the history of the Jenks algorithms, let's implement a
# clustering interface to ckmeans. The ckmeans function returns some very
# useful statistics as well as a list of labels, but does not separate the
# data. We will require a non-standard distance metric (below) so are unlikely
# to use this directly. In the future it might be worthwhile to modify the
# quadratic ckmeans implementation to use our selected distance metric.
def ckmeans_cluster(ls, k):
    r = ckwrap.ckmeans(ls, k)
    r.clusters = split_list_label(ls, r.labels)
    return r

# First we must generate idealized normally-distributed data such that each
# point is evenly spaced in terms of probability. For this we turn to the
# normal quantile function, then fit the result to the desired standard
# deviation.
def normal_perfect(n, mu, sigma):
    nd = statistics.NormalDist(0,1)
    ls = linspace(0,1,n+2)[1:-1]
    ls = [nd.inv_cdf(i) for i in ls]
    mn = mean(ls)
    sd = stdev(ls, mn)
    ls = [(i/sd*sigma)+mu for i in ls]
    return ls

# Generate and cluster perfect normal distributions with the given features.
# The first parameter is a collection of argument sets to "normal_perfect",
# one set per distribution.
def jenks_data(args_nms, metric_distance, metric_center, step=10):
    cs = sorted(c for a in args_nms for c in normal_perfect(*a))
    dm = range_round(*range_scale(cs[0], cs[-1], ratio=1.05), step)
    cs = jenks\
        ( cs, len(args_nms)
        , metric_distance=metric_distance
        , metric_center=metric_center
        )[1]
    jd = pandas.concat\
        ( pandas.DataFrame(c, columns=["x"]).assign(ki=i)
          for i,c in enumerate(cs)
        )
    return jd, dm

# More graphs! Let's see what happens when each condition is violated.
def jenks_chart(filename="chart-jenks.html"):
    selection_domain = altair.selection_interval\
        ( bind="scales"
        , encodings=["x", "y"]
        )

    charts = []

    # The clusters are spherical and have the same number of elements but they
    # have different standard deviations and therefore variances. On average
    # each element has the specified variance; this is because variance is
    # defined as the SDM per element. Therefore the sum of the variances gives
    # the SDM, though elements further from the mean have a larger variance
    # while closer elements have a smaller variance. What's happening here? The
    # left cluster has a much larger variance than the right cluster. The
    # peripheral elements of the left cluster which border the right cluster
    # are actually nearer the mean of the right cluster than the left cluster.
    # Therefore the total SDM would be reduced by moving these elements to the
    # right cluster. This cannot be predicted only with a ruler as each cluster
    # adjustment shifts the mean.
    jd, dm = jenks_data(((1000, 1500, 160), (1000, 2300, 32)), sdm, mean)
    charts.append(chart_freq
        ( jd, dm, selection_domain, color="ki:N", mean="ki"
        , title="Frequencies: n=1000 σ=160,32 distance=SDM center=mean"
        ))

    # There isn't much to be done about this situation except reduce the cost
    # to keep the elements in their original (left) cluster by changing the
    # distance metric. For example, the the square root of the SDM reduces the
    # cost of peripheral elements more than central elements. It would also be
    # possible to use the standard deviation as a metric for distance.
    distance = lambda l,m: math.sqrt(sdm(l,m))
    jd, dm = jenks_data(((1000, 1500, 160), (1000, 2300, 32)), distance, mean)
    charts.append(chart_freq
        ( jd, dm, selection_domain, color="ki:N", mean="ki"
        , title="Frequencies: n=1000 σ=160,32 distance=sqrt(SDM) center=mean"
        ))

    # The cluster remain spherical; they now have the same standard deviation
    # but a differing number of elements. The right cluster has many more
    # elements each contributing the the SDM and eventually the SDM grows large
    # enough to split the cluster. Splitting a cluster significantly reduces
    # the SDM as the distance from the mean for each element shrinks. Here
    # splitting the right cluster reduces the SDM by such a large amount that
    # it is able to absorb the left cluster for a net reduction in SDM. While
    # the variance of each element in the left cluster increases significantly
    # after being absorbed, there are so few elements in the left cluster that
    # the increase in SDM is unable to offset the net SDM reduction resulting
    # from the split.
    jd, dm = jenks_data(((26, 100, 10), (2000, 180, 10)), sdm, mean)
    charts.append(chart_freq
        ( jd, dm, selection_domain, color="ki:N", mean="ki"
        , title="Frequencies: n=26,2000 σ=10 distance=SDM center=mean"
        ))

    # Absorbing the left cluster shifts the mean towards the left cluster,
    # slightly reducing the cost of absorption. Switching to a metric less
    # affected by outliers, such as median, may help.
    jd, dm = jenks_data(((26, 100, 10), (2000, 180, 10)), sdm, median)
    charts.append(chart_freq
        ( jd, dm, selection_domain, color="ki:N", mean="ki"
        , title="Frequencies: n=26,2000 σ=10 distance=SDM center=median"
        ))

    # When the right cluster gains enough elements, using the median is not
    # enough to offset the loss in SDM from the splitting of the right cluster.
    jd, dm = jenks_data(((26, 100, 10), (4000, 180, 10)), sdm, median)
    charts.append(chart_freq
        ( jd, dm, selection_domain, color="ki:N", mean="ki"
        , title="Frequencies: n=26,4000 σ=10 distance=SDM center=median"
        ))

    # A distance metric that more severely penalizes outliers is more effective
    # than using the median. Here we use the error from the L4 norm as the
    # distance metric. Given enough elements, splitting the right cluster will
    # eventually offset the increased cost of absorption even with the new
    # distance. While median with this configuration allows a size disparity of
    # ~80:1, the L4 distance allows ~520:1. Technically the L4 distance should
    # be used in conjunction with the L4 center. However the L4 center can be
    # difficult to work with, involving intermediate complex numbers. The L4
    # center is also more sensitive to outliers and that's not a feature we
    # want.
    distance = lambda l,m: sum((i-m)**4 for i in l)
    jd, dm = jenks_data(((26, 100, 10), (4000, 180, 10)), distance, mean)
    charts.append(chart_freq
        ( jd, dm, selection_domain, color="ki:N", mean="ki"
        , title="Frequencies: n=26,4000 σ=10 distance=L4 center=mean"
        ))

    # What if we use a distance metric for which the number of elements is an
    # invariant - such as variance, the per-element SDM? Increasing the number
    # of elements in the cluster won't change the variance because the
    # distribution is preserved. Unfortunately k-means doesn't model the
    # underlying cluster distributions so the algorithm is free to reassign
    # elements in ways that breaks the normal distribution. This results in a
    # problem that is roughly the inverse of when using the SDM as a distance
    # metric. Adding an element when using SDM increases the cluster score
    # regardless of the size of the cluster. When using variance the size of
    # the cluster matters since variance is the mean of SDM. Large clusters are
    # stable; adding or removing an element does little to affect the variance.
    # Small clusters are sensitive; adding or removing an element significantly
    # changes the variance. The issue with SDM is large clusters increasing the
    # score. The issue with variance is small clusters decreasing the score,
    # which is what's happening here. The large cluster is free to absorb
    # elements from the small cluster; the large cluster's score hardly
    # increases while the small cluster's score drops significantly.
    jd, dm = jenks_data(((26, 100, 10), (2000, 180, 10)), variance, mean)
    charts.append(chart_freq
        ( jd, dm, selection_domain, color="ki:N", mean="ki"
        , title="Frequencies: n=26,2000 σ=10 distance=variance center=mean"
        ))

    # Just as with SDM, a split will result in a net loss of variance even if
    # neither resulting clusters are normally distributed. The problem is
    # compounded because there is less of a penalty for absorbing outliers.
    jd, dm = jenks_data(((26, 100, 10), (4000, 180, 10)), variance, mean)
    charts.append(chart_freq
        ( jd, dm, selection_domain, color="ki:N", mean="ki"
        , title="Frequencies: n=26,4000 σ=10 distance=variance center=mean"
        ))

    # The fix for variance is the same as for SDM; increase the penalty for
    # outliers. Since variance is less sensitive to outliers, the penalty must
    # be increased a bit more. However this solution is more robust, allowing a
    # size disparity even greater than SDM with L4 distance.
    distance = lambda l,m: sum((i-m)**4 for i in l) / math.sqrt(len(l))
    jd, dm = jenks_data(((26, 100, 10), (4000, 180, 10)), distance, mean)
    charts.append(chart_freq
        ( jd, dm, selection_domain, color="ki:N", mean="ki"
        , title="Frequencies: n=26,4000 σ=10 distance=L4/sqrt(len) center=mean"
        ))

    # Large clusters are penalized with SDM and small clusters penalized with
    # variance. There is no straightforward solution. The last thing you want
    # is to equalize the penalties for removing an item from the correct
    # cluster and adding it to another cluster. Instead of using a distance
    # metric, let's try using a test of normality. Note that the center metric
    # is no longer used; this approach no longer qualifies as k-means. Any
    # k-means or Jenks or ckmeans optimizations are probably no longer
    # applicable. However this approach performs exeedingly well.
    #
    # The Shapiro-Wilk tests generates a p-value and a W statistic. W has a
    # range of (0,1]; 1 for a perfect match to the normal distribution. Given
    # the number of points n, the test is only defined for n >= 3 though W
    # approaches 1 as n decreases. The test is roughly the inverse of the
    # 'normal_perfect' function. It standardizes the mean and deviation then
    # checks if the quantile values are evenly spaced. Our goal is to maximize
    # W; we can ignore the p-value which is anyway only defined when n <= 5000.
    # Since Jenks is a minimization algorithm we take 1-W. For n < 3 we define
    # W to be 1.
    #
    # https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test
    metric = (lambda l,m:
        1 - (1 if len(l) < 3 else scipy.stats.shapiro(l).statistic))
    jd, dm = jenks_data(((26, 100, 10), (4000, 180, 10)), metric, mean)
    charts.append(chart_freq
        ( jd, dm, selection_domain, color="ki:N", mean="ki"
        , title="Frequencies: n=26,4000 σ=10 distance=Shapiro-Wilk center=N/A"
        ))

    # Out of curiosity let's test L4 error with an L4 center. As expected it
    # performs slightly worse than L4 error with L2 center, and significantly
    # worse than a solution based on L4/variance or Shapiro-Wilks.
    jd, dm = jenks_data(((26, 100, 10), (4000, 180, 10)), l4_error, l4_center)
    charts.append(chart_freq
        ( jd, dm, selection_domain, color="ki:N", mean="ki"
        , title="Frequencies: n=26,4000 σ=10 distance=L4 center=L4"
        ))

    chart = altair.vconcat(*charts)
    chart_embed(chart, filename=filename)


# Section 7.1

# Given a list of clusters and a corresponding list of
# expected relative means, return the variance.

# Latex formula:
# vem(C, E)\ =\ \frac{1}{|C|}\sum_{i=1}^{|C|}
#   \left[\sum_{j=1}^{|C|}
#     Var\left(c_j,\ \frac{\overline{c_i}}{e_i} e_j\right)
#   \right] \\
# where: \\
# \begin{aligned}
#              C \ &=\ \text{sequence of clusters, $c_i \in C$} \\
#              E \ &=\ \text{sequence of expected means,
#                            $e_i \in E$; $e_i$ corresponds to $c_i$} \\
#            |C| \ &=\ \text{size of sequence} \\
# \overline{c_i} \ &=\ \text{mean of $c_i$} \\
#       Var(c,m) \ &=\ \text{variance of c from given mean m} \\
# \end{aligned}
#
# Given a clustering of clusters and corresponding ratios of expected
# means. For each cluster, anchor and scale the expected means from
# the actual means of that cluster then calculate the variance of
# every cluster from its expected mean. The final variance is the mean
# of all variances.
def clusters_em_var(cs, ems):
    if len(cs) != len(ems):
        raise ValueError
    if len(cs) == 1:
        return variance(cs[0], mean(cs[0]))
    vs = []
    for ic1,c1 in enumerate(cs):
        m = mean(c1)
        v = variance(c1, m)
        m /= ems[ic1]
        for ic2,c2 in enumerate(cs):
            if ic2 == ic1:
                continue
            v += variance(c2, m * ems[ic2])
        vs.append(v)
    return mean(vs)

# Latex formula:
# allvem(S, E)\ =\ \min_{C \in S}
#   \left[\min_{E_c \in \binom{E}{|C|}}vem(C, E_c)
#   \right] \\
# where: \\
# \begin{aligned}
#            S \ &=\ \text{set of C} \\
#            C \ &=\ \text{sequence of clusters, sorted by means} \\
#            E \ &=\ \text{sequence of expected means, sorted by means} \\
#          |C| \ &=\ \text{size of C} \\
# \binom{S}{k} \ &=\ \text{k-combinations of sequence S} \\
# \end{aligned}
#
# Now that we've defined the variance of a clustering given the expected means
# of each cluster, we can rank all possible combinations of clusterings and
# expected means. For example, given a 2-cluster and expected means [1,3,7],
# we can rank the cluster with these expected means: [1,3], [1,7], [3,7]. We
# do this for all given clusterings. No clustering can have more clusters than
# expected means - in other words, no outliers. It follows from the
# pidgeonhole principle, that there can not be more clusterings than expected
# means. We also assume the clusters and means are already sorted and so use
# combinations rather than permutations.
def all_clusters_em_var(css, ems, collapse=False):
    if len(css) > len(ems):
        raise ValueError
    cvs = []
    for ics,cs in enumerate(css):
        if len(cs) > len(ems):
            raise ValueError
        if len(cs) == 1:
            cvs.append([ics, variance(cs[0], mean(cs[0])), ()])
            continue
        vs = []
        for em in itertools.combinations(ems, len(cs)):
            vs.append([ics, clusters_em_var(cs, em), em])
        if collapse:
            cvs.append(min(vs, key=lambda t: t[1]))
        else:
            cvs.extend(vs)
    return cvs


# Section 7.1
# K-means algorithms such as Jenks are susceptible to outliers because
# means are easily influenced by extreme values. Let's take a look at the
# implications of outliers on real-world morse code recordings. Break the
# 'outliers' recording into four sets of 40 and analyze each set with the
# tools developed so far.

# ???
# We can see there's no good method to detect or remove
# outliers, even standard inter-qaurtile methods. While we know and can
# minimize towards the expected relative means, outliers have unknown
# means. We can modify Jenks to discard outliers during the processing
# phase (complicated) but first let's switch Jenks to a median calculation.

# We only care about the durations. Let's first get an overview of the entire
# data for comparison to the filtered data without any outliers. Then divide
# the data with outliers into groups of 40 for analysis.
# Return:
#   [ data_with_outliers
#   , data_without_outliers
#   , [grouped data with outliers]
#   ]
def morse_data():
    mo = [i[2] for i in mrec.outliers]
    mr = [i[2] for i in mrec.outliers_removed]
    mo_i = (0, len(mo))
    mr_i = (0, len(mr))
    mg_i = [(i*40,(i+1)*40) for i in range(4)]
    mg = [sorted(mo[slice(*i)]) for i in mg_i]
    mo.sort()
    mr.sort()

    mstep = 10
    mclusters = 5

    mss = ([(mo_i, mo)], [(mr_i, mr)], zip(mg_i,mg))
    mss_data = []
    for ms in mss:
        d = []
        for i,m in ms:
            m_domain = range_round\
                ( *range_scale(m[0], m[-1], ratio=1.05)
                , mstep
                )
            m_iqr = data_iqr(m, sort=False)
            m, m_metrics, m_ci = data_cluster_and_score\
                (m, mclusters, sort=False)
            d.append((m, m_iqr, m_metrics, m_ci, m_domain, i))
        mss_data.append(d)

    return [(m[0] if len(m) == 1 else m) for m in mss_data]

def morse_chart(filename="chart-morse.html"):
    tm = "Morse Outliers [{}:{}]"

    ( (mo, mo_iqr, _, _, mo_domain, mo_ix)
    , (mr, mr_iqr, mr_metrics, mr_ci, mr_domain, mr_ix)
    , mg_data
    ) = morse_data()

    # First the full data with outliers.
    selection_mo_domain = altair.selection_interval\
        ( bind="scales"
        , encodings=["x", "y"]
        )

    co_freq = chart_freq(mo, mo_domain, selection_mo_domain)
    co_iqr = chart_iqr(mo, mo_iqr, mo_domain, selection_mo_domain)

    co = altair.vconcat(co_freq, co_iqr)
    to = tm.format(*mo_ix)

    # Then the full data without outliers in more detail.
    selection_mr_domain = altair.selection_interval\
        ( bind="scales"
        , encodings=["x", "y"]
        )

    cr_freq = chart_freq(mr, mr_domain, selection_mr_domain)
    cr_clusters = chart_clusters(mr_ci, mr_domain, selection_mr_domain)
    cr_iqr = chart_iqr(mr, mr_iqr, mr_domain, selection_mr_domain)
    cr_metrics = chart_metrics(mr_metrics)

    # We know the relative means of each group. We can estimate the relative
    # frequencies of each group. We don't know the relative variance of each
    # group. I suspect the variance follows Weber's Law, so let's plot the means
    # against the standard deviations and apply a logarithmic regression. We use
    # standard deviations rather than variance since standard deviations has the
    # same units as the data. I am inverting Weber's law to account for output
    # rather than input, so we use an exponential regression rather than a
    # logarithmic regression. We only have three data points which means any
    # higher-order regression can fit, though the exponential regression has a
    # hair lower variance than the quadratic regression. To confirm this theory
    # we would need data from multiple people.
    # https://en.wikipedia.org/wiki/Weber%E2%80%93Fechner_law
    #
    # Any intrinsic feature of the model can be used for estimating the cluster
    # count, including the logarithmic/exponential relation. However it is only
    # useful for k >= 3, and marginally useful for outlier detection.
    cr_regression, cr_patch = chart_cluster_regression\
        ( mr_ci, mr_domain, selection_mr_domain
        , "Weber's Law: Logarithmic Standard Deviation vs Mean"
        )

    cr =\
        ( altair.vconcat
            ( cr_clusters
            , cr_freq
            , cr_iqr
            , cr_metrics
            , cr_regression
            )
        . resolve_scale(color="independent")
        . configure_scale(continuousPadding=1)
        )
    cr.usermeta = {"embedOptions": {"patch": cr_patch(9)}}
    tr = "Morse Without Outliers [{}:{}]".format(*mr_ix)

    # And finally the groups.
    tss = []
    css = []
    for (mg, mg_iqr, mg_metrics, mg_ci, mg_domain, mg_ix) in mg_data:
        selection_mg_domain = altair.selection_interval\
            ( bind="scales"
            , encodings=["x", "y"]
            )

        cs_freq = chart_freq(mg, mg_domain, selection_mg_domain)
        cs_clusters = chart_clusters(mg_ci, mg_domain, selection_mg_domain)
        cs_iqr = chart_iqr(mg, mg_iqr, mg_domain, selection_mg_domain)
        cs_metrics = chart_metrics(mg_metrics)

        cs =\
            ( altair.vconcat(cs_clusters, cs_freq, cs_iqr, cs_metrics)
            . resolve_scale(color="independent")
            . configure_scale(continuousPadding=1)
            )
        ts = tm.format(*mg_ix)

        css.append(cs)
        tss.append(ts)

    # It seems that hconcat charts with "container" width are not sized
    # properly, so build the HTML containers and CSS styling manually.
    css = [co, cr, *css]
    tss = [[to, tr], *split_list_ix(tss, [len(tss)//2])]

    if filename is None:
        return (css, tss)

    morse_chart_embed(css, *tss, filename)

def morse_chart_embed\
    ( charts
    , titles_before, titles_left, titles_right
    , filename
    ):
    titles = [titles_before, titles_left, titles_right]

    details = textwrap.dedent("""
    <details class="details-chart" open>
      <summary>{}</summary>
      <div><div class="vega-vis"></div></div>
    </details>
    """).strip()

    details = ["\n".join(details.format(t) for t in ts) for ts in titles]
    details[1:] = [textwrap.indent(d, " "*4) for d in details[1:]]

    container = textwrap.dedent("""
    {}

    <section class="section-columns">
      <div>
    {}
      </div>

      <div>
    {}
      </div>
    </section>
    """).strip()

    container = container.format(*details)

    style = """
    :root {
      --border: 4px solid black;
      --margin: 1em;
      --mwidth: 400px;
    }
    .details-chart {
      overflow-x: hidden;
      border: var(--border);
      border-radius: 1rem;
      margin: var(--margin);
      margin-bottom: 0;
    }
    .details-chart > summary {
      font-family: sans-serif;
      font-weight: bold;
      font-size: 100%;
      padding: 1rem;
      cursor: pointer;
      text-align: center;
      background: #eee;
      border-bottom: var(--border);
    }
    .details-chart:not([open]) > summary {
      border-bottom: none;
    }
    .details-chart > div {
      margin: 2em;
      margin-top: 1em;
    }
    .section-columns {
      display: flex;
      flex-flow: row nowrap;
      align-items: stretch;
    }
    .section-columns > div {
      flex: 0 50%;
      min-width: var(--mwidth);
    }
    .section-columns > div:not(:last-child) > .details-chart {
      margin-right: calc(var(--margin)/2);
    }
    .section-columns > div:not(:first-child) > .details-chart {
      margin-left: calc(var(--margin)/2);
    }
    .vega-vis {
      width: 100%;
      margin-left: auto;
      margin-right: auto;
      display: block !important;
    }
    .section-columns .vega-vis {
      width: calc(100% - 20px);
    }
    .vega-embed .marks {
      margin-left: 50%;
      transform:
        translateX(-50%)
        translateX(calc(var(--vega-action-padding,38px)/2));
    }
    """

    return chart_embed\
        ( *charts
        , filename=filename
        , style=style
        , container=container
        )

def jenks_median(ls, k):
    if not ls:
        raise ValueError
    if k > len(ls):
        raise ValueError
    if k < 1:
        raise ValueError
    ls = sorted(ls)
    vs = []
    for cs in list_splits(ls, k):
        v = 0
        for c in cs:
            v += mad(c, median(c))
        vs.append((v, cs))
    return min(vs, key=lambda i: i[0])

def jenks_median_breaks(ls, k):
    return breaks(jenks_median(ls, k)[1])


# Section 8.1
# This covers an optional validation metric.

# This is very well described at:
# https://en.wikipedia.org/wiki/Silhouette_(clustering)
def silhouette(ls):
    if len(ls) == 0:
        raise ValueError
    if len(ls) == 1:
        return 0
    ss = [[] for _ in ls]
    for il1,l1 in enumerate(ls):
        for ii,i in enumerate(l1):
            if len(l1) == 1:
                a = 0
            else:
                a = sum(abs(i-j) for ij,j in enumerate(l1) if ij != ii)
                a *= (1 / (len(l1) -1))
            bs = []
            for il2,l2 in enumerate(ls):
                if il2 == il1:
                    continue
                b = sum(abs(i-j) for j in l2)
                b *= (1 / len(l2))
                bs.append(b)
            b = min(bs)
            if a < b:
                s = 1 - a/b
            elif a > b:
                s = b/a - 1
            else:
                s = 0
            ss[il1].append([a,b,s])
    return ss

def silhouette_data(ss):
    s = 0
    n = 0
    for l in ss:
        for i in l:
            s += i[2]
            n += 1
    return s/n


# Section 9.1
# A module that can be both imported and run.

if __name__ == "__main__":
    #gmm_chart()
    morse_chart()
