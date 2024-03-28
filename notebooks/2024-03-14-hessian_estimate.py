
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
from alignment_vmap import em_method, fix_point_iter, align_average, autocorr_fft
from alignment_vmap import align_average_and_project, align, invariants_from_data, bispectrum_inversion, approx_newton_declarative
from itertools import product

def get_signal(L):
    ## GENERATE SIGNAL
    t = jnp.linspace(0, 2*np.pi, L+1)[:-1]
    #x = jnp.sin(t)
    x = (t < np.pi).astype(float)
    #x = np.random.randn(d)
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

def get_aligned_rel_mse(x, y):
    y_al = align(jfft(x), jfft(y), y)
    return get_rel_mse(x, y_al) ** 0.5

# Main parameters
L = 31
N = int(1e5)
t, x = get_signal(L)
xfft = jfft(x)
key = random.PRNGKey(0)

# Generate samples
noise_std = 0.1
y, noise, shift = get_samples(key, x, noise_std, N)
yfft = jfft(y, axis=1)
y_mean = jnp.mean(yfft[:, 0])  # Mean of means
y_auto_fft = jnp.clip(jnp.mean(autocorr_fft(yfft), axis=0) - noise_std**2 * L, 0, None) # mean of autocorrelation


key, subkey = random.split(key)
#z = random.normal(subkey, (L,))
z = jnp.roll(x, len(x)//2)
zfft = project_moments(jfft(z), y_auto_fft, y_mean)
z = jifft(zfft).real


h_list = np.logspace(-4, -1, 20)
loss_h = lambda vfft: loss_fft(zfft + vfft, yfft)
col = ['b', 'r']
for i in range(2):
    key, subkey = random.split(key)
    vfft = jfft(random.normal(subkey, (L,)))
    vfft = vfft - xfft * jnp.einsum("i,i->", xfft.conj(), vfft) / \
                     jnp.einsum("i,i->", xfft.conj(), xfft)

    vfft = vfft / jnp.linalg.norm(vfft)
    v = jifft(vfft).real
    changes = {}
    estimates = [[], []]
    for h in h_list:
        #print(f"h={h}")
        
        # Estimate hessian
        #hes = jnp.mean(vmap(hessian_declarative, (None, 0))(zfft, yfft), axis=0)
        #change_1 = float((vfft.conj().T @ hes @ vfft).real)
        hes = jnp.mean(vmap(hessian_declarative_real, (None, 0, 0))(z, yfft, y), axis=0)
        change_1 = float(v.T @ hes @ v) * 2
        change_2 = float((loss_h(h*vfft)-2*loss_h(0*vfft)+loss_h(-h*vfft))/(h**2))
        #change_2 = float(loss_h_real(h*v)-2*loss_h_real(0)+loss_h_real(-h*v))/h**2
        changes[h]=f"{change_1:.2e},{change_2:.2e}"
        estimates[0].append(change_1)
        estimates[1].append(change_2)
    plt.semilogx(h_list, estimates[0], col[i] +  '-*', label="analytic hessian")
    plt.semilogx(h_list, estimates[1], col[i] + '-o', label="finite differences")
    plt.semilogx(h_list, h_list*0 + jnp.linalg.norm(vfft)**2, 'r-x', label="norm squared")
    print(changes)

plt.legend()
plt.show()
    
def fd_hessian(f, x, h=1e-3):
    n = len(x)
    hes = jnp.zeros((n, n))
    I = h * jnp.eye(n)
    for i in range(n):
        for j in range(i, n):
            hes = hes.at[i, j].set((f(x + I[i] + I[j]) - f(x + I[i]) - f(x + I[j]) + f(x) + f(x - I[i] - I[j]) - f(x - I[i]) - f(x - I[j]) + f(x)) / (2 * h**2))
            hes = hes.at[j, i].set(hes[i, j])
    return hes

hes_fd = fd_hessian(lambda zfft: loss_fft(zfft, yfft), zfft)
plt.subplot(1, 2, 1)
plt.imshow(np.log(np.abs(hes.real)))
plt.colorbar()
plt.subplot(1, 2, 2)
#plt.imshow(np.log(np.abs(hes.imag)))
plt.imshow(np.log(np.abs(hes_fd)))
plt.colorbar()
plt.show()
#fig = plt.figure(figsize=(12, 8))
#fig.savefig(f"{root}data/hessian_estimate.png", bbox_inches='tight')




# phase sync
# bispectrum in
# fix oracle bug (try true momets)
# sigma^6? (sigma,N)
# size=5
# numerical hessian estimate

#1) minisympo
#2) 29th april deadline posters 
#3) october conference

#1) genoa summer school 
# applied garmonic analysis ml
