import jax
from jax import grad, jit, vmap, random
from jax.numpy.fft import fft, ifft
import jax.numpy as jnp

def add_noise(key, x, sigma):
    L = x.shape[0]
    key, subkey = jax.random.split(key)
    s1, s2 = random.randint(subkey, (2,), 0, L)
    x = jnp.roll(jnp.roll(x, s1, axis=0), s2, axis=1)
    noise = pure_noise(key, L, sigma)
    return x + noise, noise, s1, s2

def pure_noise(key, L, sigma):
    key, subkey = jax.random.split(key)
    noise = random.normal(subkey, (L, L)) * sigma
    return noise

def fft_2d(x):
    return fft(fft(x, axis=0), axis=1)

def fft_2d_center(x):
    (L1, L2) = x.shape
    x = fft_2d(x)
    x = jnp.roll(x, (L1//2, L2//2), axis=(0, 1))
    return x

def ifft_2d(x):
    return ifft(ifft(x, axis=0), axis=1)

def align(x, y):
    conv = ifft_2d(fft_2d(x) * jnp.conj(fft_2d(y))).real
    imax = jnp.argmax(conv)
    s1, s2 = jnp.unravel_index(imax, conv.shape)
    return jnp.roll(y, (s1, s2), axis=(0, 1))

def align_average(x, y):
    align_y = jit(vmap(lambda y: align(x, y)))
    return jnp.mean(align_y(y), axis=0)

def translate_img_fft(x, xshift, yshift):
    xfft = fft_2d(x)
    #kx = jnp.fft.fftfreq(x.shape[0])
    #ky = jnp.fft.fftfreq(x.shape[1])
    Lx = x.shape[0]
    Ly = x.shape[1]
    kx = jnp.arange(0, x.shape[0], 1)
    ky = jnp.arange(0, x.shape[1], 1)
    shift = jnp.exp(1j*2*jnp.pi*(kx[:, None]*xshift/Lx + ky[None, :]*yshift/Ly))
    xfft_shifted = xfft * shift# * jnp.exp(-(kx**2+ky**2)/1000)
    x_shifted = jnp.fft.ifftn(xfft_shifted).real 
    return x_shifted