import numpy as np
from numpy.fft import rfftn, irfftn

from typing import Callable


def distfromcenter(shape):
    """
    Create an array of the requested shape where each index holds its
    distance from the center index.
    """
    grid = np.ogrid[*(slice(0, s) for s in shape),]
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


def estimate_effective_range(kernel: np.ufunc) -> float:
    """Return an estimate for the effective range of a kernel, or throw an exception."""
    result = bisection(lambda x: kernel(x) - _ftol, 0, _eff_range_max)
    if result is None:
        raise ValueError("Unable to infer effective range. This is only guaranteed for "
                         + f"monotonically decreasing kernels satisfying k({_eff_range_max}) < {_ftol}.")
    return result


def convolve(x, y):
    """Convolve matrices using FFTs."""
    # determine large enough shape
    s = np.maximum(x.shape, y.shape)
    ax = np.arange(s.shape[0])
    fx = rfftn(x, s=s, axes=ax)
    fy = rfftn(y, s=s, axes=ax)
    return irfftn(fx * fy)


def noise(
        shape: float | tuple,
        kernel: np.ufunc,
        eff_range: float | None = None,
        channel_cov: float | np.ndarray = 1.,
        periodic: bool | tuple = False,
        seed: int = None
) -> np.ndarray:
    """
    Sample a mean-zero Gaussian process over a box in n-dimensional space.
    If N is the process, k is a kernel function, and * represents the
    (discrete) convolution operator, then Cov(N(x), N(y)) is proportional
    to (k * k)(||x - y||) for all sample locations x and y.

    :param shape: Shape of grid in which to sample noise. The grid resolution
                    is set so that the length along the first axis is 1. For
                     instance, shape=np.ones(d) gives a unit hypercube in d
                     dimensions.
    :param kernel: Convolutional kernel used to define noise autocorrelation. Must be broadcastable.
    :param eff_range: Effective range of kernel. Longer-range correlations are reduced or
                        absent in the noise simulated. If not provided, the effective range
                        inferred automatically assuming a monotonically decreasing kernel.
    :param channel_cov: Covariance matrix of channels in the noise sample. Provide a
                            positive scalar for single-channel noise. Otherwise, provide
                            a symmetric positive definite matrix with a row for each
                            channel. The channels appear along the last axis of the output.
    :param periodic: Whether to wrap noise around each axis, e.g. False for
                        non-repeating noise, (True, False) for 2d-noise which
                        is periodic along the first axis.
    :param seed: Random seed for replicability.
    :return: Array of shape (shape + channel_cov.shape[0]) containing a realization
                of the Gaussian process.
    """
    if seed is not None:
        np.random.seed(seed)
    shape = np.atleast_1d(shape).astype(int)
    if type(periodic) == bool:
        periodic = [periodic for _ in shape]

    # infer effective range if needed
    if eff_range is None:
        eff_range = estimate_effective_range(kernel)

    # construct kernel to convolve with
    d = distfromcenter([2 * eff_range * shape[0] + 1 for _ in shape])
    d /= shape[0]
    filt = kernel(d)
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
        result[..., c] = smooth[*(slice(0, sj) for sj in shape),]

    # induce channel correlations
    cholesky_factor = np.linalg.cholesky(channel_cov)
    return result @ cholesky_factor.T


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    # based on SO: https://stackoverflow.com/a/457805/14149906
    def erf(x):
        # save the sign of x
        sign = np.sign(x)
        x = np.abs(x)

        # constants
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        return sign * y  # erf(-x) = -erf(x)


    crg, crb, cgb = 0.0, 0.0, 0.9
    rgb = noise(
        shape=(800, 1000),
        kernel=lambda x: 0.2 - x,  # np.ones_like(x),eff_range=0.2,  # todo: make this optional
        periodic=False,  # [False, False],
        channel_cov=np.array([
            [1, crg, crb],
            [crg, 1, cgb],
            [crb, cgb, 1]
        ])
    )

    rgb = 0.5 * (1 + erf(rgb))  # inverse cdf (approximated)

    rgb = np.roll(rgb, (100, 100), axis=(0, 1))

    plt.imshow(rgb)
    plt.axis('off')
    plt.show()
