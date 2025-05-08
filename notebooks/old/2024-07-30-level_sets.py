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
from alignment_vmap import em_method, fix_point_iter, align_average, autocorr_fft, hessian_declarative, hessian_declarative_real
from alignment_vmap import align_average_and_project, align, invariants_from_data, bispectrum_inversion, manifold_iter
from itertools import product

import matplotlib as mpl

def get_signal(L):
    t = jnp.linspace(0, 2*np.pi, L+1)[:-1]
    #x = (t < np.pi).astype(float)
    x = 0 * t
    x = x.at[0].set(1.)
    x = x - 1/5
    #x = jnp.exp(jnp.sin(t))
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


def laplace(L):
    return jnp.roll(L, 1, axis=0) + jnp.roll(L, -1, axis=0) + jnp.roll(L, 1, axis=1) + jnp.roll(L, -1, axis=1) - 4*L


def get_angles(signal):
    fft_sig = jfft(signal)
    return jnp.angle(fft_sig[4]), jnp.angle(fft_sig[3])


def get_signal(L):
    t = jnp.linspace(0, 2*np.pi, L+1)[:-1]
    #x = (t < np.pi).astype(float)
    x = 0 * t
    x = x.at[0].set(1.)
    x = x - 1/5
    #x = jnp.exp(jnp.sin(t))
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
    
    
def rotated_xmark(ax, x, y, w, angle, col1, col2):
    corners = tuple((p*w/2, q*w/2) for q,p in zip([-1,1,1,-1],[-1,-1,1,1]))
    cosa = np.cos(angle)
    sina = np.sin(angle)
    corners = tuple((x + xi*cosa - yi*sina, y + xi*sina + yi*cosa) for xi, yi in corners)
    for i in [0, 2, 1, 3]:
        x1, y1 = corners[i]
        x2, y2 = corners[(i+1)%4]
        xlist = np.array([x, x1, x2])
        ylist = np.array([y, y1, y2])
        #print(f"({x1:.2}), ({y1:.2}), ({x2:.2}), ({y2:.2})")
        ax.fill(xlist, ylist, color=col1 if i % 2 == 0 else col2, edgecolor="none")


def get_angles(signal):
    fft_sig = jfft(signal)
    return jnp.angle(fft_sig[4]), jnp.angle(fft_sig[3])


def loss_fft_angle(angle1, angle2):
    z = create_signal(angle1, angle2, ymean, yabs_fft)
    yffts = jnp.split(yfft, 10, axis=0)
    loss_ = 0
    for yfft_ in yffts:
        loss_ += loss_fft(z, yfft_)/10
    return loss_
        

@jit
def loss_fft(xfft, yfft):
    yfft_align = vmap(align_fft, (None, 0))(xfft, yfft)
    return jnp.mean(jnp.abs(yfft_align - xfft)**2)

N = int(5e5)
n_pix = 200
L = 5


t, x = get_signal(L)
xfft = jfft(x)

t1x , t2x = get_angles(x)
t1x = t1x % (2*np.pi)
t2x = t2x % (2*np.pi)

for stdev in (0.1, 0.5, 1.0, 2.0):
    y, noise, shift = get_samples(random.PRNGKey(4), x, stdev, N)

    yfft = jfft(y, axis=1)
    ymean, yauto_fft, _ = invariants_from_data(y, stdev)
    yabs_fft = yauto_fft**0.5

    loss_ = jit(vmap(vmap(loss_fft_angle, (0, 0)), (0,0)))
    loss_2 = jit(vmap(loss_fft_angle, (0, 0)))



    T1, T2 = np.meshgrid(np.linspace(0, 2*np.pi, n_pix), np.linspace(0, 2*np.pi, n_pix))
    #L12 = loss_(T1, T2)
    L12 = np.zeros_like(T1)
    for i in range(n_pix):
        L12[i, :] = loss_2(T1[i], T2[i])
        #for j in range(n_pix):
            #L12[i, j] = loss_fft_angle(T1[i, j], T2[i, j])
        print(i, end='\r')


            
    #help(plt.legend)


    fig = plt.figure(figsize=(15,5))
    fig.add_subplot(111, aspect=1/jnp.abs(xfft[1]/xfft[2]))
    plt.contour(T1, T2, L12, colors="black", alpha=0.9, levels=11)
    #plt.colorbar()

    for k in range(8):
        t1xi = (t1x + k * 2 * np.pi / 5) % (2*np.pi)
        t2xi = (t2x + k * 2 * np.pi * 2 / 5) % (2*np.pi)
        plt.scatter(t1xi, t2xi, color="black", s=15, marker="+", label="true" if k==0 else "", zorder = 100)



    colors_ab = ["lightblue", "brown", "green", "red"]
    markers_ab = ["v", "b+", "r+", "^"]
    size_ab = [40, 50, 50, 40]
    for i, a in enumerate([0, np.pi]):
        for j, b in enumerate([0, np.pi]):
            for k in range(8):
                f =1/5
                ak = (a + k * 2 * jnp.pi * f) % (2*np.pi)
                bk = (b + k * 2 * jnp.pi * 2 * f) % (2*np.pi)
                m = markers_ab[i+2*j]
                if m == "r+":
                    rotated_xmark(plt.gca(), ak, bk, 0.2, 60/180*np.pi, "red", "lightblue")
                elif m == "b+":
                    rotated_xmark(plt.gca(), ak, bk, 0.2, 60/180*np.pi, "lightblue", "red")
                    if k==0:
                        plt.scatter(np.pi/2, np.pi/2, color="black", s=15, marker="", label="saddle", zorder=-5)
                else:            #get_angles(signal)
                    if k== 0:
                        if i == 0 and j == 0:
                            label = "minimum"
                        elif i == 1 and j == 0:
                            label = "saddle"
                        elif i == 0 and j == 1:
                            label = "saddle"
                        elif i == 1 and j == 1:
                            label = "maximum"
                        else:
                            label = ""
                    else:
                        label = ""
                        
                    plt.scatter(ak, bk, color=colors_ab[i+2*j], s=size_ab[i + 2*j], marker=markers_ab[i+2*j], label=label)

    #plt.figure()
    #legend = plt.gca()

    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=False, shadow=False, ncol=5)
    leg.get_frame().set_color("none")
    #plt.axis("equal")
    plt.xlim([0, 2*np.pi])
    plt.ylim([0, 2*np.pi])
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(color="black", linewidth=0.2)
    labels = ["$0$", "$\\frac{1}{2}\pi$", "$\pi$", "$\\frac{3}{2}\pi$", "$2\pi$"]
    #plt.scatter(t1_y, t2_y, color="black", s=1)
    plt.xticks([r*np.pi/2 for r in range(5)], labels)
    plt.yticks([r*np.pi/2 for r in range(5)], labels)
    plt.tight_layout()

    # HACK: Write over saddle
    print(leg)
    legend = plt.gca().inset_axes([0., -0.2, 1., 0.1])
    legend.set_xlim(0, 1)
    legend.set_ylim(0, 0.1)
    legend.axis("off")
    for i, label in enumerate(["signal", "limiting minimum", "limiting saddle", "limiting maximum"]):
        if label=="saddle":
            #pass
            rotated_xmark(legend, (i+0.2)/4, 0.05, 0.03, 0, "lightblue", "red")
        #legend.text(i/4, 0.05, label)
        



    fig.savefig(f"/home/emastr/github/multireference-alignment/figures/contour_{stdev}.pdf")
    #fig.savefig(f"/home/emastr/Downloads/heatmap.png")

