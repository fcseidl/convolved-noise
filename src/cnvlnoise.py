import numpy as np
from numpy.fft import rfftn, irfftn

from typing import Callable, Iterable


def distfromcenter(shape: Iterable[int]) -> np.ndarray:
    """
    Create an array of the requested shape where each index holds its
    distance from the center index.
    """
    grid = np.ogrid[*(slice(0, s) for s in shape), ]
    d2 = sum((g - s / 2 + 0.5) ** 2 for g, s in zip(grid, shape))
    return np.sqrt(d2)


def bisection(
        f: Callable[[float], float],
        a: float,
        b: float,
        atol: float = 1e-6,
        fa=None,
        fb=None
) -> float:
    """
    Use the bisection method to estimate a root of f on the interval [a, b]. The root must exist and be unique.

    :param f: function for rootfinding
    :param a: lower bound
    :param b: upper bound
    :param atol: absolute tolerance
    :param fa: optionally f(a)
    :param fb: optionally f(b)
    :return: the approximate root, or None if no root is found
    """
    m = (a + b) / 2
    fa = fa if fa is not None else f(a)
    fb = fb if fb is not None else f(b)
    fm = f(m)
    # base case: return mid- or endpoint closest to root
    if b - a < atol:
        return [a, m, b][np.argmin(np.abs([fa, fm, fb]))]
    # recursive case: search for root on subinterval
    if fa * fm < 0:
        return bisection(f, a=a, b=m, atol=atol, fa=fa, fb=fm)
    if fm * fb <= 0:
        return bisection(f, a=m, b=b, atol=atol, fa=fm, fb=fb)


# parameters for eff range search
_ftol = 1e-4
_eff_range_max = 1.


def estimate_effective_range(radial_func: np.ufunc) -> float:
    """Return an estimate for the effective range of a radial function, or throw an exception."""
    divisor = radial_func(0.)
    result = bisection(lambda x: radial_func(x) / divisor - _ftol, 0, _eff_range_max) if divisor != 0 else None
    if result is None:
        raise ValueError("Unable to infer effective range of radial function. This is only guaranteed for "
                         + f"monotonically decreasing functions satisfying f({_eff_range_max}) < {_ftol}.")
    return result


def convolve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convolve two arrays of the same dimension using FFTs."""
    # determine large enough shape
    s = np.maximum(x.shape, y.shape)
    ax = np.arange(s.shape[0])
    fx = rfftn(x, s=s, axes=ax)
    fy = rfftn(y, s=s, axes=ax)
    return irfftn(fx * fy)


def noise(
        shape: int | Iterable[int],
        radial_func: np.ufunc = lambda x: np.exp(-50. * x * x),
        eff_range: float | None = None,
        channel_cov: float | np.ndarray = 1.,
        periodic: bool | Iterable[bool] = False,
        seed: int = None
) -> np.ndarray:
    """
    Sample a stationary, isotropic mean-zero Gaussian process over a box
    in n-dimensional space. The user specifies a radial function f such
    that if N is the process, k(x, y) = f(||x - y||), and * represents
    the (discrete) convolution operator, then Cov(N(x), N(y)) is
    proportional to (k * k)(x, y) for all sample locations x and y.

    :param shape: Shape of grid in which to sample noise. The grid
                    resolution is set so that the length along the first
                    axis is 1. For instance, shape=np.ones(d) gives a
                    unit hypercube in d dimensions.
    :param radial_func: Radial function used to define noise autocorrelation.
                        Must be broadcastable. If not specified, a gaussian
                        kernel is used.
    :param eff_range: Effective range of radial function. Longer-range
                        correlations are reduced or absent in the noise
                        simulated. If not provided, the effective range
                        is inferred automatically assuming a monotonically
                        decreasing radial function.
    :param channel_cov: Covariance matrix of channels in the noise sample. Provide
                            a positive scalar for single-channel noise. Otherwise,
                            provide a symmetric positive definite matrix with a row
                            for each channel. The channels appear along the last
                            axis of the output, if there are multiple channels.
    :param periodic: Whether to wrap noise around each axis, e.g. False for
                        non-repeating noise, (True, False) for 2d-noise which
                        is periodic along the first axis.
    :param seed: Random seed for replicability.
    :return: Array of specified shape with channels along the last axis. For instance,
                shape == (1, 2, 3) and channel_cov is a scalar, then the result has
                shape (1, 2, 3). If alternatively, channel_cov is a 4x4 matrix, then
                the result has shape (1, 2, 3, 4).
    """
    if seed is not None:
        np.random.seed(seed)
    shape = np.atleast_1d(shape).astype(int)
    if type(periodic) == bool:
        periodic = [periodic for _ in shape]

    # infer effective range if needed
    if eff_range is None:
        eff_range = estimate_effective_range(radial_func)

    # construct filter to convolve with
    d = distfromcenter([2 * eff_range * shape[0] + 1 for _ in shape])
    d /= shape[0]
    filt = radial_func(d)
    filt[d > eff_range] = 0
    filt /= np.linalg.norm(filt)

    # determine shape of white noise to sample
    pad_shape = shape.copy()
    pad_shape[np.equal(periodic, False)] += filt.shape[0] - 1
    channel_cov = np.atleast_2d(channel_cov)
    n_channels = channel_cov.shape[0]
    result = np.empty(tuple(shape) + (n_channels,))

    # sample and smooth noise channels
    for c in range(n_channels):
        white = np.random.randn(*pad_shape)
        smooth = convolve(white, filt)
        result[..., c] = smooth[*(slice(0, sj) for sj in shape), ]

    # induce channel joint distribution
    cholesky_factor = np.linalg.cholesky(channel_cov)
    result = result @ cholesky_factor.T
    if n_channels == 1:
        result = result[..., 0]     # no need for channel axis with one channel
    return result


if __name__ == "__main__":
    pass
