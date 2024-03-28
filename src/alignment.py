import sys

if __name__ == '__main__':
    sys.path.append('/home/emastr/phd/projects/vahid_project/src/')

import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, random, lax, pmap
from jax.numpy.fft import fft, ifft

def autocorr(x):
    return ifft(autocorr_fft(fft(x)))


def trunc(xfft, k):
    xfft.at[1+k:-k].set(0.)
    return xfft


def autocorr_fft(xfft):
    return jnp.abs(xfft) ** 2


def find_shifts(xfft, yfft):
    return jnp.argmax(ifft(xfft * yfft.conj(), axis=1).real, axis=1)


def align(xfft, yfft, y):
    shift_rec = find_shifts(xfft, yfft)
    y_shift = jnp.vstack([jnp.roll(y_i, s_i) for y_i, s_i in zip(y, shift_rec)])
    return y_shift


def project_moments(xfft, acf_fft, mean):
    xfft = xfft.at[0].set(mean)
    xfft = xfft.at[1:].set(xfft[1:] * acf_fft[1:]**0.5 / jnp.abs(xfft[1:]))
    return xfft

def align_average_and_project(xfft, yfft, y, acf_fft, mean):
    y_shift = align(xfft, yfft, y)
    x = jnp.mean(y_shift, axis=0)
    xfft = project_moments(fft(x), acf_fft, mean)
    return xfft

def loss_fourier(xfft, yfft, y):
    y_shift = align(xfft, yfft, y)
    return xfft.shape[0]  **2 * jnp.mean(jnp.abs(y_shift - ifft(xfft))**2) / 2

def loss_real(x, yfft, y):
    y_shift = align(fft(x), yfft, y)
    return x.shape[0] * jnp.mean(jnp.abs(y_shift - x)**2) / 2


def shift_cont(yfft, shifts):
    """Shift signals according to shifts. Shifts should be complex numbers on the unit circle."""
    assert yfft.shape[0] == len(shifts), "Must have same length as signal"
    d = yfft.shape[1]
    powers = np.arange(0, d, 1)
    shifts = shifts[:, None] ** powers[None, :]
    return yfft * shifts


def loss_cont(xfft, yfft, shifts):
    yfft_shifted = shift_cont(yfft, shifts)
    return jnp.mean(jnp.abs(xfft - yfft_shifted)**2).real

grad_loss = grad(loss_fourier)
grad_loss_real = grad(loss_real)
grad_loss_cont = grad(loss_cont, argnums=[0,2])



############## METHODS ###################################


def fix_point_iter(x0, yfft, y, acf_fft, mean, tol, maxiter, alpha=0.5, callback=None):
    res = tol + 1.
    i = 0
    xfft = fft(x0)
    while (res > tol) and i < maxiter:
        xfft_new = (1 - alpha) * xfft + alpha * align_average_and_project(xfft, yfft, y, acf_fft, mean)
        res = jnp.mean(jnp.abs(xfft - xfft_new))
        xfft = xfft_new
        i += 1
        if callback is not None:
            callback(xfft, res, i)
    return ifft(xfft).real


def projected_gradient_descent_fourier(x0, yfft, y, acf_fft, mean, step, tol, maxiter, callback=None, logger=None):
    res = tol + 1
    i = 0
    
    xfft = fft(x0)
    while (res > tol) and (i < maxiter):
        xfft_new = project_moments(xfft - step * jnp.conjugate(grad_loss(xfft, yfft, y)), acf_fft, mean)
        res = jnp.mean(jnp.abs(xfft - xfft_new))
        i += 1
        xfft = xfft_new
        if callback is not None:
            callback(xfft, res, i)
    return ifft(xfft).real



def projected_gradient_descent_real(x0, yfft, y, acf_fft, mean, step, tol, maxiter, callback=None):
    res = tol + 1
    i = 0
    
    x = x0
    while (res > tol) and (i < maxiter):
        x_new = ifft(project_moments(fft(x - step * jnp.conjugate(grad_loss_real(x, yfft, y))), acf_fft, mean))
        res = jnp.mean(jnp.abs(x - x_new))
        i += 1
        x = x_new

        if callback is not None:
            callback(fft(x), res, i)

    return x.real


def projected_newton_descent(x0, yfft, y, acf_fft, mean, step, tol, maxiter, callback=None):
    res = tol + 1
    i = 0
    
    xfft = fft(x0)
    while (res > tol) and (i < maxiter):
        xfft_new = project_moments(xfft - step * jnp.conjugate(grad_loss(xfft, yfft, y)) / (acf_fft**0.5 + 0.1) * np.linalg.norm(acf_fft)**0.5, acf_fft, mean)
        res = jnp.mean(jnp.abs(xfft - xfft_new))
        i += 1
        xfft = xfft_new

        if callback is not None:
            callback(xfft, res, i)

    return ifft(xfft).real


def cont_projected_gradient_descent(x0, yfft, acf_fft, mean, step, tol, maxiter, callback=None):
    res = tol + 1.
    i = 0
    d = len(x0)

    xfft = fft(x0)
    shifts = find_shifts(xfft, yfft)
    shifts = jnp.exp(-2j * np.pi * shifts / d)


    #### NYQUIST FREQUENCY SET 0? 
    #### NEGATIVE SHIFTS INSTEAD OF POSITIVE?
    
    while (res > tol) and (i < maxiter):
        grad_x, grad_shift = grad_loss_cont(xfft, yfft, shifts)
        xfft_new = project_moments(xfft - step * jnp.conjugate(grad_x), acf_fft, mean)
        shifts = shifts - 10*step * jnp.conjugate(grad_shift)
        shifts = shifts / jnp.abs(shifts)

        res = jnp.mean(jnp.abs(xfft_new - xfft))
        xfft = xfft_new
        i += 1

        if callback is not None:
            callback(xfft, res, i)

    return ifft(xfft).real, shifts
