import numpy as np
import math
import cmath

def gaussian(x, t, sigma):
    """A Gaussian curve.
    x = Variable
    t = time shift
    sigma = standard deviation"""
    return np.exp(-((x - t) ** 2) / (2 * sigma ** 2))


def free(npts):
    "Free particle."
    return np.zeros(npts)


def step(npts, v0):
    "Potential step"
    v = free(npts)
    v[npts // 2 :] = v0
    return v


def barrier(npts, v0, thickness):
    "Barrier potential"
    # v = free(npts)
    # v[npts // 2 : npts // 2 + thickness] = v0
    v = np.array(
            [v0 if 0.0 < x < thickness else 0.0 for x in npts])
    return v

def theta(x):
    """
    theta function :
      returns 0 if x<=0, and 1 if x>0
    """
    x = np.asarray(x)
    y = np.zeros(x.shape)
    y[x > 0] = 1.0
    return y

def square_barrier(x, width, height):
    return height * (theta(x) - theta(x - width))


def tunneling_probability(v0, E, m, h, a):
    f1 = 1 + (v0**2)/(4*E*(v0-E))
    f2 = math.sin(h**2)
    return 1/(f1 * f2 *((cmath.sqrt(2*m*(v0-E))*a)/(h)))