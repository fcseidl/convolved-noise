from noise import noise
import matplotlib.pyplot as plt
import numpy as np


fig, axs = plt.subplots(2, 2)
nx = 1000
seed = 2

n00 = noise(nx, lambda x: np.ones_like(x), eff_range=0.25, seed=seed)
axs[0, 0].plot(n00, c='black')
axs[0, 0].set_title("f(r) = 1, range = 0.25")
axs[0, 0].axis('off')

n01 = noise(nx, lambda x: 0.25 - x, seed=seed)
axs[0, 1].plot(n01, c='black')
axs[0, 1].set_title("f(r) = max(0.25 - r, 0)")
axs[0, 1].axis('off')

n10 = noise(nx, lambda x: np.cos(10 * x), eff_range=np.pi / 10, seed=seed)
axs[1, 0].plot(n10, c='black')
axs[1, 0].set_title("f(r) = cos(10r), range = pi/10")
axs[1, 0].axis('off')

n11 = noise(nx, lambda x: np.exp(-100 * x * x), seed=seed)
axs[1, 1].plot(n11, c='black')
axs[1, 1].set_title("f(r) = e^{-100r^2}")
axs[1, 1].axis('off')

plt.show()
