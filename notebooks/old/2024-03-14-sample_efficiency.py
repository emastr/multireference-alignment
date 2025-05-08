
import sys
root = '/home/emastr/github/phd/projects/vahid_project/'
sys.path.append('..')
sys.path.append(f'{root}src/')
sys.path.append(f'{root}util/')
sys.path.append(f'{root}data/')
from multiprocessing import Pool
from matplotlib.colors import LogNorm
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
from alignment_vmap import align_average_and_project, align, invariants_from_data, bispectrum_inversion
from itertools import product

MAXITER = 1000

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

def em_run(x0, y, noise_std):
    return em_method(jfft(x0), y, noise_std, tol=1e-6, batch_niter=3000, full_niter=MAXITER)

def fpi_run(x0, y, noise_std):
    N, L = y.shape
    yfft = jfft(y, axis=1)
    y_auto_fft = jnp.clip(jnp.mean(autocorr_fft(yfft), axis=0) - noise_std**2 * L, 0, None) # mean of autocorrelation
    y_mean = jnp.mean(yfft[:, 0])  # Mean of means

    return  ifft(fix_point_iter(jfft(x0), yfft, y_auto_fft, y_mean, alpha=1.0, tol=1e-6, maxiter=MAXITER)).real

def bsi_run(x0, y, noise_std):
    N, L = y.shape
    yfft = jfft(y, axis=1)
    y_mean, yc_auto_fft, B = invariants_from_data(y, noise_std)
    x, x2, _ = bispectrum_inversion(y_mean, yc_auto_fft, B, tol=1e-6, maxiter=(200, MAXITER))
    if loss_fft(jfft(x), yfft) > loss_fft(jfft(x2), yfft):
        x = x2
    return x


def algn_run(x0, y, noise_std):
    return ifft(align_average(jfft(x0),  fft(y, axis=1))).real


def average_std(M):
    def decorator(func):
        def wrapper(key, *args, **kwargs):
            keys = random.split(key, M)
            vals = jnp.array([func(keys[m], *args, **kwargs) for m in range(M)])
            return {"mean": jnp.mean(vals, axis=0), "std": jnp.std(vals, axis=0)}
        return wrapper
    return decorator

@average_std(5)
def run_methods(key, x, N, noise_std, methods):
    y, noise, shift = get_samples(key, x, noise_std, N)

    xfft = fft(x)
    x_auto_fft = jnp.abs(xfft)**2.

    
    # Add oracle - the other methods do not have access to the noise
    oracle = lambda x0, y, noise_std: x + jnp.mean(noise, axis=0)
    
    
    def oracle_fpi(x0, y, noise_std):
        yfft = jfft(y, axis=1)
        y_mean = jnp.mean(yfft[:, 0])  # Mean of means
        return  ifft(fix_point_iter(fft(x0), yfft, x_auto_fft, y_mean, alpha=1.0, tol=1e-20, maxiter=MAXITER)).real

    methods = methods + (oracle, )
    
    # Run methods
    def run(method):
        t = time.time()
        val = get_aligned_rel_mse(x, method(y[0, :].copy(), y, noise_std))
        t = time.time() - t
        return val, t
    return jnp.array([run(method) for method in methods])




compute = False
if compute:
    # Main parameters
    L = 41
    N = int(1e3)
    t, x = get_signal(L)

    # Compute sample efficiency: error vs (number of samples, noise_std)
    num_samples = np.logspace(1, 4, 10, dtype=int)
    noise_stds = np.logspace(-3, 1, 10, dtype=float)
    methods = (fpi_run, bsi_run, em_run)
    names = ["fpi", "bsi", "em", "oracle", "N", "STD"]

    N_mat, STD_mat = np.meshgrid(num_samples, noise_stds, indexing='ij')
    E_mat = np.zeros(N_mat.shape + (len(methods)+3,), dtype=float)
    E_mat[:,:,-1] = N_mat
    E_mat[:,:,-2] = STD_mat

    for i in range(N_mat.shape[0]):
        for j in range(N_mat.shape[1]):
            N = N_mat[i, j]
            noise_std = STD_mat[i, j]
            res = run_methods(random.PRNGKey(0), x, N, noise_std, methods)
            for k in range(len(methods)+1):
                E_mat[i, j, k] = res["mean"][k][0]
            print(f"result {i*N_mat.shape[1] + j + 1}/{N_mat.size} done")
    # Save resulp

    np.save(f"{root}data/efficiency_table", E_mat)

else:
    E_mat = np.load(f"{root}data/efficiency_table.npy")
    N_mat = E_mat[:, :, -2]
    STD_mat = E_mat[:, :, -1]
    
    ### HACK, REMOVE LATER
    num_samples = np.logspace(1, 4, 10, dtype=int)
    noise_stds = np.logspace(-3, 1, 10, dtype=float)
    methods = (fpi_run, bsi_run, em_run)
    names = ["Fix Point", "Bispectrum", "EM", "Oracle"]
    N_mat, STD_mat = np.meshgrid(num_samples, noise_stds, indexing='ij')
    ############
    
    
    fig = plt.figure(figsize=(15, 5))
    #plt.pcolormesh(N_mat, STD_mat, E_mat[:, :, 2], norm=LogNorm())
    for i in range(len(names)):
        plt.subplot(1, len(names), i+1)
        plt.title(f"Sample efficiency {names[i]}")
        
        # contour begin
        levels = [float(r) for r in [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]]
        strings = ['$10^{' + f'{i}' + '}$' for i in [-4, -3, -2, -1, 0, 1]]
        fmt = {l: s for s, l in zip(strings, levels)}
        CS = plt.contour(STD_mat, N_mat, E_mat[:, :, i], levels=levels, norm=LogNorm())
        for j, curve in enumerate(CS.allsegs):
            curve = curve[0]
            print(curve)
            if len(curve) > 0:
                x = curve[len(curve)//2][0] #(curve[0][0]*curve[-1][0])**0.5
                y  = curve[len(curve)//2][1]#(curve[0][1]*curve[-1][1])**0.5
                plt.text(x, y, strings[j], backgroundcolor="white")
        plt.grid(True, which='both', linewidth=0.5)
        #lbls = plt.gca().clabel(CS, inline=True, fontsize=10, fmt=fmt, inline_spacing=0.01)
        #for lbl in lbls:
            #lbl.set_rotation(0)
        ### Contour end
        plt.yscale("log")
        plt.xscale("log")
        plt.ylabel("Number of samples N")
        plt.xlabel("Noise level $\sigma$")
        plt.ylim([N_mat.min(), N_mat.max()])
        plt.xlim([STD_mat.min(), STD_mat.max()])
    #plt.colorbar()
    plt.tight_layout()
    fig.savefig(f"{root}data/efficiency2.png")
    
    
    
    
    fig2 = plt.figure(2)#figsize=(15, 5))
    ax2 = plt.gca()
    fig = plt.figure(3)
    ax = plt.gca()
    colors = ["coral", "seagreen", "steelblue", "indianred"]
    styles = ["-s", "-o", "-D", "-^"]
    #plt.pcolormesh(N_mat, STD_mat, E_mat[:, :, 2], norm=LogNorm())
    
    plt.title(f"Sample efficiency at different error levels $\epsilon$")
    for i in range(len(names)):
        
        # contour begin
        #levels = [float(r) for r in [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]]
        levels = [float(r) for r in [1e-4, 1e-3, 1e-2]]
        strings = ['$\epsilon = 10^{' + f'{i}' + '}$' for i in [-4, -3, -2, -1, 0, 1]]
        fmt = {l: s for s, l in zip(strings, levels)}
        CS = ax2.contour(STD_mat, N_mat, E_mat[:, :, i], levels=levels, norm=LogNorm(), colors=colors[i])
        #plt.plot([], [], color=colors[i], label=names[i])

        for j, curve in enumerate(CS.allsegs):
            curve = curve[0]
            x = [curve[i][0] for i in range(len(curve))]
            y = [curve[i][1] for i in range(len(curve))]
            if j != 0:
                ax.plot(x, y, styles[i], color=colors[i], markerfacecolor='white')
            else:
                ax.plot(x, y, styles[i], color=colors[i], label=names[i], markerfacecolor='white')
    
    for j, curve in enumerate(CS.allsegs):
        curve = curve[0]
        print(curve)
        if len(curve) > 0:
            x = 0.7*curve[len(curve)//2][0] #(curve[0][0]*curve[-1][0])**0.5
            y  = curve[len(curve)//2][1]#(curve[0][1]*curve[-1][1])**0.5
            plt.text(x, y, strings[j], backgroundcolor="white")
            
    ss = STD_mat[0, :]
    plt.plot(ss, [(2000*s)**2 for s in ss], '--', color='black', label='$O(\sigma^2)$')
    plt.plot(ss, [(100*s)**3 for s in ss], '--', color='black', label='$O(\sigma^3)$')
    plt.legend(loc="lower right")
    plt.grid(True, which='both', linewidth=0.5)
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Number of samples N")
    plt.xlabel("Noise level $\sigma$")
    plt.ylim([N_mat.min(), N_mat.max()])
    #plt.xlim([STD_mat.min(), STD_mat.max()])
    plt.xlim([STD_mat.min(), 1.])
    #plt.colorbar()
    plt.tight_layout()
    fig.savefig(f"{root}data/efficiency3.png")
