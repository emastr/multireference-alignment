import sys
sys.path.append('..')
sys.path.append('/home/emastr/phd/projects/vahid_project/src/')
sys.path.append('/home/emastr/phd/projects/vahid_project/util/')
sys.path.append('/home/emastr/phd/projects/vahid_project/data/')


from multiprocessing import Pool, cpu_count

from matplotlib import pyplot as plt
from src.alignment_vmap import *
import jax
from jax import vmap, grad, jit, random, lax
import jax.numpy as jnp
from jax.numpy.fft import fft as jfft, ifft as jifft



def get_signal(L):
    t = jnp.linspace(0, 2*np.pi, L+1)[:-1]
    x = (t < np.pi).astype(float)
    return t, x

def get_samples(key, x, noise_std, N):
    L = len(x)
    shiftkey, noisekey = jax.random.split(key, 2)
    shift = jax.random.randint(shiftkey, (N,), 0, L)
    noise = jax.random.normal(noisekey, (N, L)) * noise_std
    y = vmap(lambda s, z: jnp.roll(x + z, s), in_axes=(0, 0))(shift, noise)
    return y, noise, shift


def create_signal(angle1, angle2, mean, xfft_abs):
    e1 = jnp.exp(1j * angle1)
    e2 = jnp.exp(1j * angle2)
    return jnp.array([mean,
                      e1 * xfft_abs[1],
                      e2 * xfft_abs[2],
                      e2.conj() * xfft_abs[3],
                      e1.conj() * xfft_abs[4]])
    
def hessian(z, yfft, y):
    y_align = vmap(align, (None, 0, 0))(z, yfft, y)
    return jnp.einsum('mi,mj -> ij', y_align, y_align) / y_align.shape[0]


def hessian_project(z, yfft, y):
    hes_ = hessian(z, yfft, y)
    hes_ = jfft(jfft(hes_, axis=0), axis=1)
    P = jnp.eye(len(z)) - jnp.outer(z, z.conj())/jnp.dot(z.conj(), z)
    return P @ hes_ @ P

N = 1000
n_pix = 90
L = 5
stdev = .1

t, x = get_signal(L)
y, noise, shift = get_samples(random.PRNGKey(4), x, stdev, N)
yfft = jfft(y, axis=1)
ymean, yauto_fft, _ = invariants_from_data(y, stdev)
yabs_fft = yauto_fft**0.5

def loss_fft_angle(angle1, angle2):
    z = create_signal(angle1, angle2, ymean, yabs_fft)
    return loss_fft(z, yfft)

def hessian_angle(angle1, angle2):
    def sign(z):
        return z / jnp.abs(z)
    z = create_signal(angle1, angle2, ymean, yabs_fft)
    hes_ = hessian(z, yfft, y)
    dza = jnp.array([[1j * sign(z[1]), 0, 0, 0, -1j * sign(z[4])],
                     [0, 1j * sign(z[2]), -1j * sign(z[3]), 0, 0]])
    dza = jfft(dza, axis=1).real
    return dza @ hes_ @ dza.T
    
    

def gauss_curv_angle(angle1, angle2):
    hes_ = hessian_angle(angle1, angle2)
    return jnp.linalg.det(hessian_angle(angle1, angle2))
    #return hes_[0,0] + hes_[1,1]



loss_ = jit(vmap(vmap(loss_fft_angle, (0, 0)), (0,0)))
gauss_ = jit(vmap(vmap(gauss_curv_angle, (0, 0)), (0,0)))


T1, T2 = np.meshgrid(np.linspace(0, 2*np.pi, n_pix), np.linspace(0, 2*np.pi, n_pix))

L12 = loss_(T1, T2)
C12 = gauss_(T1, T2)

fig = plt.figure(figsize=(10,5))
#plt.plot(t, y[0, :])
plt.subplot(1, 2, 1)
plt.pcolormesh(T1, T2, L12)
plt.axis("equal")
#plt.colorbar()
plt.subplot(1, 2, 2)
plt.pcolormesh(T1, T2, C12)
plt.axis("equal")
plt.colorbar()

plt.tight_layout()
#plt.plot(x)
fig.savefig('/home/emastr/phd/projects/vahid_project/data/loss_landscape.png')


