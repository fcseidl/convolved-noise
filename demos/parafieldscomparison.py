import numpy as np

from src.cnvlnoise import noise

from time import time
import parafields as pf
import matplotlib.pyplot as plt


nx = 1000
n = 1

scale = 0.19

a = pf.generate_field(cells=[nx, nx], extensions=(1., 1.), covariance="gaussian", corrLength=scale).evaluate()
b = noise([nx, nx]) #, lambda x: np.exp(-2 * (x / scale) ** 2))

fig, axs = plt.subplots(1, 2)
axs[0].imshow(a, cmap='gray'); axs[0].set_title("parafields"); axs[0].axis('off')
axs[1].imshow(b, cmap='gray'); axs[1].set_title("convolution"); axs[1].axis('off')
plt.show()


print(f"Sampling {n} random fields with a resolution of {nx}...")

total = 0.
for _ in range(n):
    t0 = time()
    pf.generate_field(cells=[nx, nx], extensions=(1., 1.), covariance="gaussian", corrLength=scale)
    total += time() - t0
print(f"parafields averaged {total / n} seconds per field.")

total = 0.
for _ in range(n):
    t0 = time()
    noise([nx, nx], lambda x: np.exp(-2 * (x / scale) ** 2))
    total += time() - t0
print(f"convolutional-noise averaged {total / n} seconds per field.")
