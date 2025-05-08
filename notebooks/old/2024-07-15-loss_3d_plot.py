import jax
from jax.numpy.fft import fft, ifft
import jax.numpy as jnp
import matplotlib.pyplot as plt  
from jax import grad, vmap
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

import sys
root = '/home/emastr/github/phd/projects/vahid_project/'
sys.path.append('..')
sys.path.append(f'{root}src/')
sys.path.append(f'{root}util/')
sys.path.append(f'{root}data/')
from multiprocessing import Pool
from matplotlib.colors import LogNorm
from alignment_vmap import *
from alignment_vmap import em_method, fix_point_iter, align_average, autocorr_fft, hessian_declarative, hessian_declarative_real
from alignment_vmap import align_average_and_project, align, invariants_from_data, bispectrum_inversion

def get_samples(key, x, noise_std, N):
    L = len(x)
    shiftkey, noisekey = jax.random.split(key, 2)
    shift = jax.random.randint(shiftkey, (N,), 0, L)
    noise = jax.random.normal(noisekey, (N, L)) * noise_std
    y = vmap(lambda s, z: jnp.roll(x + z, s), in_axes=(0, 0))(shift, noise)
    return y, noise, shift

# Use finite difference hessian for stepping.

import math
outer = lambda x, y: jnp.einsum('i,j->ij', x.conj(), y)
inner = lambda x, y: jnp.einsum('i,i->', x.conj(), y)
diff = lambda x: (x - jnp.roll(x, len(x)//10))/float(len(x)//10)*float(len(x))

Lhalf = 3
L = 2*Lhalf + 1
t = jnp.linspace(0, 2*jnp.pi, L+1)[:-1]
f = lambda t: t == 0. #> jnp.pi
#f = lambda t: t> jnp.pi
x = f(t)
xfft = fft(x)

M = 50
angles = jnp.angle(xfft[1:Lhalf+1])
angles = angles + jnp.pi * jnp.array([1., 1., 0.])
i, j = 0, 1

loss_wrap = lambda t1, t2, t3: loss_angle(angles + jnp.array([t1,t2,t3]), xfft_abs, xfft0)
t1 = jnp.linspace(-jnp.pi, jnp.pi, M) + angles[0]
t2 = jnp.linspace(-jnp.pi, jnp.pi, M) + angles[1]
t3 = jnp.linspace(-jnp.pi, jnp.pi, M) + angles[2]

T1, T2, T3 = jnp.meshgrid(t1, t2, t3, indexing='ij')
U = jnp.load('U.npy')

angles_xyz = np.zeros((Lhalf, 2**Lhalf))
angles_idx = np.zeros(2**Lhalf)
for i in range(2**Lhalf):
    shifts = [int(x) for x in bin(i)[2:]]
    shifts = [0]*(Lhalf-len(shifts)) + shifts
    angles_xyz[:, i] = (jnp.pi * jnp.array(shifts))%(2*jnp.pi)
    angles_idx[i] = np.sum(shifts)

df = pd.DataFrame(angles_xyz.T, columns=['t1', 't2', 't3'])
df['idx'] = angles_idx
    

isominmax = [(0.85, 1.3), (0.0, 0.6), (1., 1.3), (0., 1.3), (0.65, 0.7)]
i = -2

#U = np.where(((T1-np.pi)**2 + (T2-np.pi)**2 + (T3-0)**2)**0.5 <= 2., U, np.nan)

print(angles_idx)
fig = px.scatter_3d(df, x="t1", y="t2", z="t3", color="idx")
fig.add_trace(go.Volume(
    x=T1.flatten(), y=T2.flatten(), z=T3.flatten(),
    value=U.flatten(),
    isomin=isominmax[i][0],
    isomax=isominmax[i][1],
    opacity=0.2,
    slices_y=dict(show=True, locations=[jnp.pi]),
    surface_count=1,
    caps= dict(x_show=False, y_show=False, z_show=False)
    ))

fig.update_layout(scene_xaxis_showticklabels=False,
                  scene_yaxis_showticklabels=False,
                  scene_zaxis_showticklabels=False)


fig.show()


