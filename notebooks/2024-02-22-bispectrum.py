import sys
root = '/home/emastr/github/phd/projects/vahid_project/'
sys.path.append('..')
sys.path.append(f'{root}src/')
sys.path.append(f'{root}util/')
sys.path.append(f'{root}data/')
from multiprocessing import Pool
from alignment_vmap import *
import numpy as np
from numpy.fft import fft, ifft

import jax
import time
from jax import vmap, grad, jit, random, lax
from jax import numpy as jnp
from jax.numpy.fft import fft as jfft, ifft as jifft
from util.logger import EventTracker
import matplotlib.pyplot as plt
from alignment_vmap import em_method, fix_point_iter, align_average, autocorr_fft, bispectrum_inversion
from alignment_vmap import align_average_and_project, align, circulant, invariants_from_data
from itertools import product



def get_signal(L):
    ## GENERATE SIGNAL
    t = jnp.linspace(0, 2*np.pi, L+1)[:-1]
    #x = jnp.sin(2*t)+1.0
    x = (t < np.pi*1.0).astype(float)
    #x = np.random.randn(L)
    return t, x


def get_samples(key, x, noise_std, N):
    L = len(x)
    shiftkey, noisekey = jax.random.split(key, 2)
    shift = jax.random.randint(shiftkey, (N,), 0, L)
    noise = jax.random.normal(noisekey, (N, L)) * noise_std
    y = vmap(lambda s, z: jnp.roll(x + z, s), in_axes=(0, 0))(shift, noise)
    return y, noise, shift

def get_mse(x, y):
    return jnp.mean((x - y) ** 2)

def get_snr(x, noise_std):
    return get_mse(x, jnp.mean(x)) / noise_std ** 2

def get_rel_mse(x, y):
    return get_mse(x, y) / get_mse(x, 0.0)

def get_mean_centered_rel_mse(x, y):
    return get_mse(x, y) / get_mse(x, jnp.mean(x))


N = 1000
L = 41
std = 0.01
t, x = get_signal(L)
Y, noise, shift = get_samples(random.PRNGKey(0), x, std, N)
y_mean, yc_auto_fft, B = invariants_from_data(Y, std)
print("Invariants computed")
x1, x2, z = bispectrum_inversion(y_mean, yc_auto_fft, B, tol=1e-6, maxiter=(15, 200))
x1 = align(jfft(x), jfft(x1), x1)
x2 = align(jfft(x), jfft(x2), x2)

#if get_mean_centered_rel_mse(x, x1) > get_mean_centered_rel_mse(x, x2):
#    x1 = x2

fig = plt.figure()

plt.subplot(121)
plt.plot(t, x)
plt.plot(t, x1)

plt.subplot(122)
plt.plot(z.real)
plt.plot(fft(x).real / jnp.abs(fft(x)))

print(get_mean_centered_rel_mse(x, x1) ** 0.5)
#plt.xlim([2,4])
#plt.plot(t, x-x1)
fig.savefig('/home/emastr/phd/projects/vahid_project/data/invariants.png')