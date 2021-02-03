import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from scipy.special import gammainc
import time


# Read data from datafile:
# - bins_low = lower histogram boundary;    defailt = 1e-4
# - bins_high = higher histogram boundary;  default = 5
# - nbins = number of bins to use;          default = 20
class Read_data:
    def __init__(self, datafile, bins_low=1e-4, bins_high=5, nbins=20):
        file = open(datafile, 'r')
        lines = file.readlines()
        lines = [lines[i][:-1] for i in range(len(lines))]  # remove \n at the end of each line
        file.close()

        self.nhaloes = int(lines[3])  # extract the no. of haloes - 3rd line
        self.nsats = np.zeros(self.nhaloes)  # to hold the no. of satellites of each halo

        i = 0
        # Count the number of satellites per halo:
        for l in lines[5:]:
            if l == '#':  # new halo encountered
                i += 1
            else:  # new satellite encountered: +1 to halo satellite count
                self.nsats[i] += 1

        self.Nsat = np.average(self.nsats)  # this is <Nsat>

        # read in the satellite coordinates:
        self.coords = np.genfromtxt(datafile, skip_header=4)
        # Make the histogram:
        self.bins = np.logspace(np.log10(bins_low), np.log10(bins_high), nbins)
        self.counts, self.bin_edges = np.histogram(self.coords[:, 0], bins=self.bins)
        self.Ni = self.counts / self.nhaloes    # average no. of satellites per halo in each bin
        self.Ni_poisson = []  # to hold the Poisson equivalent of Ni
        self.nbins = nbins
        self.bins_low = bins_low
        self.bins_high = bins_high
        self.A = []     # Integral normalisation: to be recomputed at every (a, b, c) evaluation
        self.chi = []


# Given the data 'data' and parameters (a, b, c),
# 1) compute normalisation A
# 2) compute Ni_poisson = tilde(Ni) - Poisson expectation counts value
class Get_chi_squared():
    def __init__(self, data, a, b, c):
        args = [a, b, c]
        data.A = 1/(4*np.pi*self.simpsons(self.n_not_normalised, data.bins_low, data.bins_high, 100000, args))
        data.Ni_poisson = [4*np.pi*self.simpsons(self.n_normalised_poisson,
                                                 a=data.bin_edges[i], b=data.bin_edges[i+1], N=10000,
                                                 args=[a, b, c, data.Nsat, data.A]) for i in range(data.nbins-1)]

    # Simpson's rule:
    # integrate a function f in the interval from a to b, using N points
    def simpsons(self, func, a, b, N, args=[]):
        h = (b - a) / N
        s = func(a, *args) + func(b, *args)
        for i in range(1, N, 2):
            s += 4 * func(a + h * i, *args)
        for i in range(2, N - 1, 2):
            s += 2 * func(a + h * i, *args)
        return s * h / 3

    # General chi squared
    def chi_squared(self, x, mu, sigma):
        return sum((x - mu) * (x - mu) / sigma / sigma)

    # Expression to integrate, as function of x = r/r-vir
    def n_not_normalised(self, x, *args):
        [a, b, c] = args
        return x ** (a - 1) * b ** (3 - a) * np.exp(-(x / b) ** c)

    def n_normalised_poisson(self, x, a, b, c, nsat, A):
        return A*nsat*(x/b)**(a-3)*np.exp(-(x/b)**c)*x*x

    def get_chi(self):
        return self.chi_squared(data.Ni, mu=data.Ni_poisson, sigma=data.Ni_poisson)

    # Poisson log-likelihood for the data observation (Ni) and expectation (Ni_poisson)
    def get_poisson_likelihood(self):
        n = len(data.Ni)
        s = np.zeros(n)
        for i in range(n):
            s[i] = data.Ni_poisson[i] - data.Ni[i]*np.log(data.Ni_poisson[i])
            for j in range(int(data.Ni[i]))[::-1]:  # the factorial
                s[i] += np.log(j + 1)
        return sum(s)


# Get chi squared value at a point in parameter space
def chi_minimise(point):
    a = point[0]
    b = point[1]
    c = point[2]
    return Get_chi_squared(data, a, b, c).get_chi()


# Get log-likelihood value at a point in parameter space
def likelihood_minimise(point):
    a = point[0]
    b = point[1]
    c = point[2]
    return Get_chi_squared(data, a, b, c).get_poisson_likelihood()


# Swap function
def swap(a, b):
    swap = a
    a = b
    b = swap
    return a, b


# Simple selection sort algorithm; return sorted indices
def selection(a):
    n = len(a)
    b = np.copy(a)
    ind_array = np.arange(n)
    for i in range(n):
        m = min(b[i:])
        # print(m)
        ind = np.argmin(b[i:])
        swap = b[i]
        b[i] = m
        b[ind + i] = swap
        # swap = i
        ind_array[i] = i + ind
        # ind_array[ind+i] = swap
        # print(a)
    return b, ind_array


# N-dimensional Downhill simplex
def simplex_nd(func, x, maxiter=8, accuracy=1e-3):
    iteration = 0
    n = len(x)
    f = np.empty(n)
    for i in range(n):
        f[i] = func(x[i])
    # -------------------- Downhill simplex implementation: -------------------
    while iteration <= maxiter:
        # sort the points:
        for i in range(n):
            f[i] = func(x[i])
        # print(f)
        f, ind = selection(f)
        # print(f, ind)
        x = x[ind]

        xbar = np.empty(x.shape[1])
        for i in range(int(x.shape[1])):
            xbar[i] = np.average(x[:, i])

        frac_range = abs(f[-1]-f[0])/abs(f[-1]+f[0])*2

        # Follow the algorithm until accuracy is met: for more algorithm details see e.g. the book, the slides
        if frac_range < accuracy:
            result = xbar
            print('\nAccuracy reached in ', iteration, '  iterations.')
            break
        else:
            xtry = 2*xbar - x[-1]
            ftry = func(xtry)

            if f[0] < ftry < f[-1]:
                # print('case0')
                x[-1] = xtry
            elif ftry < f[0]:
                # print('expand')
                xexp = 2*xtry - xbar
                fexp = func(xexp)
                if fexp < ftry:
                    x[-1] = xexp
                else:
                    x[-1] = xtry
            elif ftry >= f[-1]:
                # print('contract')
                xnew = 0.5*(xbar+x[-1])
                fnew = func(xnew)
                if fnew < f[-1]:
                    x[-1] = xnew
                else:
                    x = 0.5*(xbar+x)

        iteration += 1
        print(iteration, end='')
    if int(iteration) == int(maxiter):
        print("Max iterations reached (", str(maxiter), '), best guess is: ', xbar)
    return xbar, func(xbar)


# Standard G-test implementation, expression from the slides:
def Gtest(observed, expected):
    r = observed/expected
    a = observed*np.log(r)
    a = a[~np.isnan(a)]
    G = 2*sum(a)
    return G


# Significance Q of a variable following a chi-squared distribution;
# expression from the slides.
def Q(k, x):
    p = gammainc(k/2, x/2)/math.gamma(k/2)
    return 1 - p


# Function that performs KS test: is the sample 'observed' consistent
# with 'expected'? NOTE: assumes the data is binned in THE SAME bins
# Returns KS statistic D and probability P (significance Q).
# See report and 'Numerical Recipes' for more info.
def ks_test(observed, expected):
    # Numerically optimal implementation of P, the CDF:
    def Q(z):
        if z == 0:
            return 1
        elif z < 1.18:
            v = np.exp(-np.pi*np.pi/8/z/z)
            P = np.sqrt(2*np.pi)/z*(v + v**9 + v**25)
            return 1 - P
        elif z >= 1.18:
            v = np.exp(-2*z*z)
            P = 1 - 2*(v-v**4+v**9)
            return 1-P

    n = len(observed)
    c = sum(observed)   # normalisation factor
    dist = np.array([abs(sum(observed[:i])/c-sum(expected[:i])/c) for i in range(n)])

    D = max(abs(dist))
    z = D*(np.sqrt(n) + 0.12 + 0.11/np.sqrt(n))
    return D, Q(z)


datafiles = ['satgals_m11.txt', 'satgals_m12.txt', 'satgals_m13.txt', 'satgals_m14.txt', 'satgals_m15.txt']
for datafile in datafiles[:2]:
    beg = time.time()
    # Read data in data class:
    data = Read_data(datafile=datafile, bins_low=1e-4, bins_high=5, nbins=20)

    # Initial tetrahedron:
    x0 = np.array([1, 0.5, 1])
    x1 = np.array([1.1, 0.7, 1.5])
    x2 = np.array([0.8, 1.2, 1.2])
    x3 = np.array([1, 1.3, 1.2])

    n = 3   # number of dimensions

    x = np.vstack((x0, x1, x2, x3))
    # Minimise, starting at initial tetrahedron:
    [[a, b, c], chi_min] = simplex_nd(chi_minimise, x)
    Get_chi_squared(data, a, b, c)
    g = Gtest(observed=data.Ni, expected=data.Ni_poisson)
    # degrees of freedom k: fixed parameter number is the dimension of a point in param space (x0),
    # i.e. 3 here (a, b, c) plus the total number of satellites observed - Nsat. The number of free
    # parameters is the number of bins (each bin = observation of a ~random variable).
    k = data.nbins - len(x0) - 1
    q = Q(k=k, x=g)
    d, q_ks = ks_test(observed=data.Ni, expected=data.Ni_poisson)

    # -------------- Report results --------------
    print('\nResults for datafile ', datafile, ':')
    print('<Nsat> =', data.Nsat.round(2))
    print('-- Best fit parameters: --')
    print('a = \t', a.round(3))
    print('b = \t', b.round(3))
    print('c = \t', c.round(3))
    print('Min chi-squared = ', chi_min.round(3), '\n')
    print('G value =\t', g.round(5))
    print('Q value =\t', q)
    print('KS statistic D =', d.round(3))
    print('KS significance Q =', q_ks.round(3), '\n')

    bin_centres = np.array([0.5*(data.bin_edges[i+1] + data.bin_edges[i]) for i in range(len(data.bin_edges)-1)])
    bin_width = np.array([data.bin_edges[i+1]-data.bin_edges[i] for i in range(len(data.bin_edges)-1)])
    plt.figure()
    plt.loglog()
    plt.step(bin_centres, data.Ni, where='mid', c='grey', label='Data')
    plt.step(bin_centres, data.Ni_poisson, where='mid', c='royalblue', label='Min chi-squared')
    plt.xlabel('x')
    plt.ylabel('Counts N')

    [[a2, b2, c2], lmin] = simplex_nd(likelihood_minimise, x)
    Get_chi_squared(data, a2, b2, c2)
    g = Gtest(observed=data.Ni, expected=data.Ni_poisson)
    k = data.nbins - len(x0) - 1
    q = Q(k=k, x=g)
    d, q_ks = ks_test(observed=data.Ni, expected=data.Ni_poisson)
    # -------------- Report results --------------
    print('\n')
    print('<Nsat> =', data.Nsat.round(2))
    print('-- Best fit parameters: --')
    print('a = \t', a2.round(3))
    print('b = \t', b2.round(3))
    print('c = \t', c2.round(3))
    print('Min log-likelihood = ', lmin, '\n')
    print('G value =\t', g.round(5))
    print('Q value =\t', q)
    print('KS statistic D =', d.round(3))
    print('KS significance Q =', q_ks.round(3), '\n')

    plt.step(bin_centres, data.Ni_poisson, where='mid', c='maroon', label='Min log-likelihood')
    plt.legend(frameon=False)
    plt.title(datafile)
    plt.savefig('plots/'+datafile[:11]+'-hist.png', dpi=300)

    end = time.time()
    print(datafile, ' took ', round(end-beg), 's')

