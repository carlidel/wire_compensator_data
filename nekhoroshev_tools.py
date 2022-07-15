"""Some code elements that will be extremely useful in this analysis...
"""

import itertools

import crank_nicolson_numba.generic as cn
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.integrate

# For parallelization
from joblib import Parallel, delayed
from numpy.lib.shape_base import expand_dims
from tqdm.notebook import tqdm


def D(I, I_star, exponent, c=1.0, halved=False):
    """Generate Nekhoroshev diffusion

    Parameters
    ----------
    I : ndarray
        action position
    I_star : float
        coefficient
    exponent : float
        coefficient
    c : float (default=1.0)
        normalization factor
    halved : bool (default=False)
        multiply by 0.5? you need to do that in certain crank-nicolson scenarios...

    Returns
    -------
    ndarray
        diffusion value
    """    
    return c * np.exp(-2*np.power(I_star/I, exponent)) * (0.5 if halved else 1.0)


def D_exp(I, k, c=1.0, halved=False):
    return c * np.exp(k * I) * (0.5 if halved else 1.0)


def D_poly(I, a, c=1.0, halved=False):
    return c * np.power(I, a) * (0.5 if halved else 1.0)


def standard_c(I_min, I_max, I_star, exponent):
    result = scipy.integrate.quad(lambda x: D(x, I_star,
                                              exponent), I_min, I_max)[0]
    if result == 0.0:
        return np.inf
    else:
        return 1/result


def afpt(I_min, I_max, I_star, exponent, c=None):
    if c is None:
        c = standard_c(0.0, I_max, I_star, exponent)
    
    if not hasattr(I_min, "__iter__"):
        return scipy.integrate.quad(
            lambda x: 2*x/D(x, I_star, exponent, c=c),
            I_min,
            I_max
        )[0]
    else:
        return np.array([
            scipy.integrate.quad(
                lambda x: 2*x/D(x, I_star, exponent, c=c),
                i,
                I_max
            )[0] for i in I_min
        ])


def stationary_dist(I_min, I_max, I_star, exponent, c=None):
    if c is None:
        c = standard_c(0.0, I_max, I_star, exponent)

    if not hasattr(I_min, "__iter__"):
        return scipy.integrate.quad(
            lambda x: 1/D(x, I_star, exponent, c=c),
            I_min,
            I_max
        )[0]
    else:
        return np.array([
            scipy.integrate.quad(
                lambda x: 1/D(x, I_star, exponent, c=c),
                i,
                I_max
            )[0] for i in I_min
        ])


def single_x(I, I_max, I_star, exponent, c):
    """Return the corresponding variable x(I)

    Parameters
    ----------
    I : float
        action value
    I_max : float
        absorbing barrier
    I_star : float
        coefficient
    exponent : coefficient
        coefficient
    c : float
        coefficient

    Returns
    -------
    float
        x value
    """    
    return -scipy.integrate.quad(lambda x: 1/np.sqrt(D(x, I_star, exponent, c)), I, I_max)[0]

x = np.vectorize(single_x, excluded=["I_star", "exponent", "c"], doc=
    """Return the corresponding variable x(I)

    Parameters
    ----------
    I : ndarray
        action value
    I_max : ndarray
        absorbing barrier
    I_star : float
        coefficient
    exponent : coefficient
        coefficient
    c : float
        coefficient

    Returns
    -------
    ndarray
        x value
    """
)


def nu(I, I_star, exponent, c):
    """compute nu(I)

    Parameters
    ----------
    I : ndarray
        action position
    I_star : float
        coefficient
    exponent : float
        coefficient
    c : float
        coefficient

    Returns
    -------
    ndarray
        nu values
    """    
    return (np.sqrt(c) * exponent / I) * np.power(I_star/I, exponent) * np.exp(-np.power(I_star/I, exponent))


def current_peak_time(I_0, I_max, I_star, exponent, c):
    """Return timing of current peak (analytical)

    Parameters
    ----------
    I_0 : ndarray
        action values
    I_max : ndarray
        max values
    I_star : float
        coefficient
    exponent : float
        coefficient
    c : float
        coefficient

    Returns
    -------
    ndarray
        current timing
    """    
    return 2*(np.sqrt(nu(I_0, I_star, exponent, c)**2 * x(I_0, I_max, I_star, exponent, c)**2 + 9) - 3) / nu(I_0, I_star, exponent, c)**2


def current_peak_value(I_0, I_max, I_star, exponent, c):
    """Return value of current peak (analytical)

    Parameters
    ----------
    I_0 : ndarray
        action values
    I_max : ndarray
        max values
    I_star : float
        coefficient
    exponent : float
        coefficient
    c : float
        coefficient

    Returns
    -------
    ndarray
        current value
    """
    return -x(I_0, I_max, I_star, exponent, c)*np.exp(-nu(I_0, I_star, exponent, c)**2*(x(I_0, I_max, I_star, exponent, c) + (np.sqrt(nu(I_0, I_star, exponent, c)**2*x(I_0, I_max, I_star, exponent, c)**2 + 9) - 3)/nu(I_0, I_star, exponent, c))**2/(4*(np.sqrt(nu(I_0, I_star, exponent, c)**2*x(I_0, I_max, I_star, exponent, c)**2 + 9) - 3)))/(4*np.sqrt(np.pi)*((np.sqrt(nu(I_0, I_star, exponent, c)**2*x(I_0, I_max, I_star, exponent, c)**2 + 9) - 3)/nu(I_0, I_star, exponent, c)**2)**(3/2))


def current_derivative(t, I, I_max, I_star, exponent, c):
    return -np.sqrt(2)*x(I, I_max, I_star, exponent, c)*(-nu(I, I_star, exponent, c)*(nu(I, I_star, exponent, c)*t/2 + x(I, I_max, I_star, exponent, c))/(2*t) + (nu(I, I_star, exponent, c)*t/2 + x(I, I_max, I_star, exponent, c))**2/(2*t**2))*np.exp(-(nu(I, I_star, exponent, c)*t/2 + x(I, I_max, I_star, exponent, c))**2/(2*t))/(2*np.sqrt(np.pi)*t**(3/2)) + 3*np.sqrt(2)*x(I, I_max, I_star, exponent, c)*np.exp(-(nu(I, I_star, exponent, c)*t/2 + x(I, I_max, I_star, exponent, c))**2/(2*t))/(4*np.sqrt(np.pi)*t**(5/2))


def current_second_derivative(t, I, I_max, I_star, exponent, c):
    return np.sqrt(2)*x(I, I_max, I_star, exponent, c)*(-nu(I, I_star, exponent, c)**4*t**4 - 24*nu(I, I_star, exponent, c)**2*t**3 + 8*nu(I, I_star, exponent, c)**2*t**2*x(I, I_max, I_star, exponent, c)**2 - 240*t**2 + 160*t*x(I, I_max, I_star, exponent, c)**2 - 16*x(I, I_max, I_star, exponent, c)**4)*np.exp(-nu(I, I_star, exponent, c)**2*t/8 - nu(I, I_star, exponent, c)*x(I, I_max, I_star, exponent, c)/2 - x(I, I_max, I_star, exponent, c)**2/(2*t))/(128*np.sqrt(np.pi)*t**(11/2))


def current_point(t, I, I_max, I_star, exponent, c):
    """Compute the value of the current for a specific point for a dirac delta distribution

    Parameters
    ----------
    t : ndarray
        times
    I : float
        action
    I_max : float
        maximum
    I_star : float
        coefficient
    exponent : float
        coefficient
    c : float
        coefficient

    Returns
    -------
    list
        current
    """    
    if t == 0:
        return 0.0
    return -x(I, I_max, I_star, exponent, c) / (t * np.sqrt(2*np.pi*t)) * np.exp(-(x(I, I_max, I_star, exponent, c)+((nu(I, I_star, exponent, c)/2)*t))**2/(2*t))


def current_uniform(t, I, I_max, I_int_min, I_star, exponent, c):
    """Compute the value of the current for a uniform distribution for different times.

    Parameters
    ----------
    t : ndarray
        times
    I : float
        cut_point
    I : float
        maximum
    I_int_min : float
        minimum initial integration point
    I_star : float
        coefficient
    exponent : float
        coefficient
    c : float
        coefficient

    Returns
    -------
    list
        current
    """    
    return [scipy.integrate.quad(lambda x: current_point(a_t, x, I_max, I_star, exponent, c), I_int_min, I)[0] for a_t in t]


def current_generic(t, rho, I_max, I_int_min, I_star, exponent, c):
    return [
        scipy.integrate.quad(
            lambda x: 
                current_point(a_t, x, I_max, I_star, exponent, c) * rho(x),
            I_int_min, I_max
        )[0] for a_t in t
    ]


def compute_current_with_peak(I_0, I_max, I_star, exponent, t_sampling=1000, I_sampling=50000, I_min=0.0):
    c = standard_c(I_min, I_max, I_star, exponent)
    I_linspace, dI = np.linspace(I_min, I_max, I_sampling, retstep=True)
    sigma = dI * 5
    def rho_0(I):
        return np.exp(-0.5 * (I - I_0)**2/sigma**2) / (sigma*np.sqrt(2*np.pi))

    ana_current_peak_time = current_peak_time(I_0, I_max, I_star, exponent, c)
    ana_current_peak_value = current_peak_value(I_0, I_max, I_star, exponent, c)

    dt = ana_current_peak_time/t_sampling
    engine = cn.cn_generic(I_min, I_max, rho_0(
        I_linspace), dt, lambda x: D(x, I_star, exponent, c, halved=True), normalize=False)
    times, current = engine.current(t_sampling*2, 1)

    num_current_peak = np.max(current)
    num_current_time = times[np.argmax(current)]

    return times, current, ana_current_peak_time, ana_current_peak_value, num_current_time, num_current_peak


def compute_generic_current_with_peak(I_0, I_max, rho, I_star, exponent, t_sampling=1000, t_multiplier=2, I_sampling=50000, I_min=0.0):
    c = standard_c(I_min, I_max, I_star, exponent)
    I_linspace, dI = np.linspace(I_min, I_max, I_sampling, retstep=True)
    ana_current_peak_time = current_peak_time(I_0, I_max, I_star, exponent, c)
    
    dt = ana_current_peak_time/t_sampling
    engine = cn.cn_generic(
        I_min, I_max,
        rho(I_linspace),
        dt,
        lambda x: D(x, I_star, exponent, c, halved=True),
        normalize=False
    )
    times, current = engine.current(t_sampling * t_multiplier, 1)

    num_current_peak = np.max(current)
    num_current_time = times[np.argmax(current)]

    return c, times, current, ana_current_peak_time, num_current_time, num_current_peak


def locate_generic_maximum(I_0, I_max, rho, I_star, exponent, starting_point=None, I_min_int=0.1):
    c = standard_c(0.0, I_max, I_star, exponent)
    if starting_point is None:
        starting_point = current_peak_time(I_0, I_max, I_star, exponent, c)
    result = scipy.optimize.fmin(
        lambda x: - current_generic(x, rho, I_max, I_min_int, I_star, exponent, c)[0],
        starting_point,
        disp=False
    )
    return result


def analytical_recover(I_0, I_max, I_star, exponent, ratio=0.1, I_min=0.0):
    c = standard_c(I_min, I_max, I_star, exponent)
    t0 = current_peak_time(I_0, I_max, I_star, exponent, c)
    m0 = current_peak_value(I_0, I_max, I_star, exponent, c)
    result = scipy.optimize.fmin(
        lambda x: np.absolute(current_point(
            x, I_0, I_max, I_star, exponent, c) - m0 * ratio),
        t0*0.5,
        disp=False
    )
    return result[0]


def single_fit_routine(x, peak_time, peak_value, I_0, I_max, I_star, exponent, I_min=0):
    c = standard_c(I_min, I_max, I_star, exponent)
    t = current_peak_time(I_0, I_max, x[0], x[1], c)
    v = current_peak_value(I_0, I_max, x[0], x[1], c)
    e1 = np.absolute(peak_time - t)/peak_time
    e2 = np.absolute(peak_value - v)/peak_value
    return np.sqrt(e1**2 + e2**2)
