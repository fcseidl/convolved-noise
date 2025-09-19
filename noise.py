import numpy as np
from numpy.fft import rfftn, irfftn

from typing import Callable


def distfromcenter(shape):
    """
    Create an array of the requested shape where each index holds its
    distance from the center index.
    """
    grid = np.ogrid[*(slice(0, s) for s in shape), ]
    d2 = sum((g - s / 2 + 0.5) ** 2 for g, s in zip(grid, shape))
    return np.sqrt(d2)


def convolve(x, y):
    """Convolve matrices using FFTs."""
    # determine large enough shape
    s = np.maximum(x.shape, y.shape)
    ax = np.arange(s.shape[0])
    fx = rfftn(x, s=s, axes=ax)
    fy = rfftn(y, s=s, axes=ax)
    return irfftn(fx * fy)


def cone_filter(radius, resolution, dimension) -> np.ndarray:
    """
    Convolutional filter for cone kernel, scaled to unit norm.

    :param radius: Radius of cone.
    :param resolution: Number of grid cells per unit.
    :param dimension: Dimension of filter.
    """
    hw = int(radius * resolution)
    w = 2 * hw + 1
    shape = tuple(w for _ in range(dimension))
    d = distfromcenter(shape)
    filt = (d < radius)
    return filt / np.linalg.norm(filt)


def rbf_filter(sigma, nsig, resolution, dimension) -> np.ndarray:
    """
    Convolutional filter for RBF kernel, scaled to unit norm.

    :param sigma: Sigma paremeter of radial basis function. (Not squared.)
    :param nsig: Half-width of filter in units of sigma.
    :param resolution: Number of grid cells per unit.
    :param dimension: Dimension of filter.
    """
    hw = int(sigma * nsig * resolution)
    w = 2 * hw + 1
    shape = tuple(w for _ in range(dimension))
    d2 = distfromcenter(shape) ** 2
    filt = np.exp(-d2 / (sigma * sigma))  # convolution of Gaussian is Gaussian
    return filt / np.linalg.norm(filt)


def noise(
        shape: float | tuple,
        kernel: np.ufunc,
        eff_range: float,
        channel_cov: float | np.ndarray = 1.,
        periodic: bool | tuple = False,
        seed: int = None
) -> np.ndarray:
    """
    Sample a mean-zero Gaussian process over a box in n-dimensional space, satisfying
        Cov(N(x), N(y)) = (k * k)(||x - y||),
    where N is the process, k is a kernel function, and * represents the
    (discrete) convolution operator.

    :param shape: Shape of grid in which to sample noise. The grid resolution
                    is set so that the length along the first axis is 1. For
                     instance, shape=np.ones(d) gives a unit hypercube in d
                     dimensions.
    :param kernel: Convolutional kernel used to define noise covariance. Must be broadcastable.
    :param eff_range: Effective range of kernel. Longer-range correlations are reduced or
                        absent in the noise simulated.
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
    cholesky_factor = np.linalg.cholesky(channel_cov)

    result = np.empty(tuple(shape) + (n_channels, ))

    for c in range(n_channels):
        white = np.random.randn(*pad_shape)
        smooth = convolve(white, filt)
        result[..., c] = smooth[*(slice(0, sj) for sj in shape), ]

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

    crg, crb, cgb = 0.0, 0.99, 0.0
    rgb = noise(
        shape=(800, 1000),
        kernel=lambda x: np.ones_like(x),
        eff_range=0.2,
        periodic=False,     # [False, False],
        channel_cov=np.array([
                        [1, crg, crb],
                        [crg, 1, cgb],
                        [crb, cgb, 1]
                    ])
    )

    rgb = 0.5 * (1 + erf(rgb))      # inverse cdf (approximated)

    rgb = np.roll(rgb, (100, 100), axis=(0, 1))

    plt.imshow(rgb)
    plt.axis('off')
    plt.show()
