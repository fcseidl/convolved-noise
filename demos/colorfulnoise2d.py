from noise import noise
import matplotlib.pyplot as plt
import numpy as np


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


fig, axs = plt.subplots(2, 2)
shape = (480, 640)
seed = 2

crg, crb, cgb = 0.0, 0.0, 0.0
n00 = noise(shape,
            radial_func=lambda x: np.exp(-100 * x * x),
            channel_cov=np.array([
                [1, crg, crb],
                [crg, 1, cgb],
                [crb, cgb, 1]
            ]),
            seed=seed)
n00 = 0.5 * (1 + erf(n00))
axs[0, 0].imshow(n00)
axs[0, 0].set_title("Uncorrelated Colors")
axs[0, 0].axis('off')

crg, crb, cgb = 0.99, 0.0, 0.0
n01 = noise(shape,
            radial_func=lambda x: np.exp(-100 * x * x),
            channel_cov=np.array([
                [1, crg, crb],
                [crg, 1, cgb],
                [crb, cgb, 1]
            ]),
            seed=seed)
n01 = 0.5 * (1 + erf(n01))
axs[0, 1].imshow(n01)
axs[0, 1].set_title("corr(R, G) = 0.99")
axs[0, 1].axis('off')

crg, crb, cgb = 0.0, 0.99, 0.0
n10 = noise(shape,
            radial_func=lambda x: np.exp(-100 * x * x),
            channel_cov=np.array([
                [1, crg, crb],
                [crg, 1, cgb],
                [crb, cgb, 1]
            ]),
            seed=seed)
n10 = 0.5 * (1 + erf(n10))
axs[1, 0].imshow(n10)
axs[1, 0].set_title("corr(R, B) = 0.99")
axs[1, 0].axis('off')

crg, crb, cgb = 0.95, 0.95, 0.95
n11 = noise(shape,
            radial_func=lambda x: np.exp(-100 * x * x),
            channel_cov=np.array([
                [1, crg, crb],
                [crg, 1, cgb],
                [crb, cgb, 1]
            ]),
            seed=seed)
n11 = 0.5 * (1 + erf(n11))
axs[1, 1].imshow(n11)
axs[1, 1].set_title("All correlations 0.95")
axs[1, 1].axis('off')

plt.show()
