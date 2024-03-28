import sys

if __name__ == '__main__':
    sys.path.append('/home/emastr/phd/projects/vahid_project/src/')

import numpy as np
import jax.numpy as jnp
from jax import grad, jit, random, lax, pmap
from jax import vmap
from jax.numpy.fft import fft, ifft
from jax.scipy.linalg import toeplitz
import time
from jax.lax import while_loop

## Fourier tools
def resize(xfft, n):
    n_old = len(xfft)
    k = (min(n, n_old)-1) // 2 
    xfft_new = jnp.zeros(n, dtype=xfft.dtype)
    xfft_new = xfft_new.at[:1+k].set(xfft[:1+k])
    xfft_new = xfft_new.at[-k:].set(xfft[-k:])
    return xfft_new

## Autocorrelation

@jit
def autocorr(x):
    return ifft(autocorr_fft(fft(x)))


def autocorr_fft(xfft):
    return jnp.abs(xfft) ** 2


def compl_sign(x):
    return x / jnp.abs(x)
## Alignment

@jit
def find_shift(xfft, yfft):
    return jnp.argmax(ifft(xfft * yfft.conj()).real)

@jit
def roll_fft(yfft, shift):
    """Shift signals according to shifts. Shifts should be complex numbers on the unit circle."""
    d = len(yfft)
    powers = jnp.arange(0, d, 1)
    return yfft * jnp.exp(-2 * jnp.pi * 1j * shift/d) ** powers

@jit
def align(xfft, yfft, y):
    shift = find_shift(xfft, yfft)
    return jnp.roll(y, shift)

@jit
def align_fft(xfft, yfft):
    shift = find_shift(xfft, yfft)
    return roll_fft(yfft, shift)

## Moments
@jit
def project_moments(xfft, acf_fft, mean):
    xfft = xfft.at[0].set(mean)
    xfft = xfft.at[1:].set(jnp.where(xfft[1:] != 0, \
        xfft[1:]*acf_fft[1:]**0.5 / jnp.abs(xfft[1:]), 0))
    return xfft

@jit
def align_average(xfft, yfft):
    return jnp.mean(vmap(align_fft, (None, 0))(xfft, yfft), axis=0)
    #return jnp.mean(pmap(align_fft, in_axes=(None, 0))(xfft, yfft), axis=0)

@jit
def align_average_and_project(xfft, yfft, acf_fft, mean):
    xfft = align_average(xfft, yfft)
    xfft = project_moments(xfft, acf_fft, mean)
    return xfft


def relative_error(x, y):
    return jnp.linalg.norm(x - y) / jnp.linalg.norm(x)


@jit
def loss_fft(xfft, yfft):
    yfft_align = vmap(align_fft, (None, 0))(xfft, yfft)
    return jnp.mean(jnp.abs(yfft_align - xfft)**2)

############## METHODS #####################################

def fix_point_iter(x0fft, yfft, acf_fft, mean, tol, maxiter, alpha=0.5, callback=None):
    res = tol + 1.
    i = 0
    xfft = x0fft
    while (res > tol) and i < maxiter:
        xfft_new = (1 - alpha) * xfft + alpha * align_average_and_project(xfft, yfft, acf_fft, mean)
        res = relative_error(ifft(xfft), ifft(xfft_new))
        xfft = xfft_new
        i += 1
        if callback is not None:
            callback(xfft, res, i)
    return xfft

def manifold_iter(x0fft, yfft, acf_fft, mean, tol, maxiter, alpha=0.1, callback=None):
    res = tol + 1.
    i = 0
    xfft = project_moments(x0fft, acf_fft, mean)
    while (res > tol) and i < maxiter:
        grad_fft = xfft - align_average(xfft, yfft)
        grad_fft = grad_fft - inner(grad_fft, xfft) * xfft / inner(xfft, xfft)
        direction = jnp.imag(jnp.conjugate(xfft)*grad_fft) / (tol + jnp.abs(xfft))# / jnp.abs(grad_fft)
        phase_shift = jnp.exp(-1j * alpha * direction)
        #print(jnp.linalg.norm(phases_grad_fft))
        
        xfft_new = xfft * phase_shift #project_moments(xfft * phases_grad_fft, acf_fft, mean)
        res = relative_error(ifft(xfft), ifft(xfft_new))
        xfft = xfft_new
        i += 1
        if callback is not None:
            callback(xfft, res, i)
    return xfft


def stochastic_fix_point_iter(x0fft, yfft, acf_fft, mean, tol, maxiter, batch_size=2, alpha=0.5, callback=None):
    res = tol + 1.
    i = 0
    xfft = x0fft
    while (res > tol) and i < maxiter:
        idx = random.choice(random.PRNGKey(i), len(yfft), shape=(batch_size,))
        xfft_new = (1 - alpha) * xfft + alpha * align_average_and_project(xfft, yfft[idx], acf_fft, mean)
        res = relative_error(ifft(xfft), ifft(xfft_new))
        xfft = xfft_new
        i += 1
        if callback is not None:
            callback(xfft, res, i)
    return xfft


def competing_fix_point_iter(x0fft, yfft, acf_fft, mean, tol, maxiter, num_inits=3, decay=0.9, alpha=0.5, callback=None):
    res = tol + 1.
    i = 0
    std = 1. # Random initialization std
    xfft = x0fft
    while (res > tol) and i < num_inits:
        j = 0
        while (res > tol) and j < maxiter:
            xfft_new = (1 - alpha) * xfft + alpha * vmap(align_average_and_project, (0, None, None, None))(xfft, yfft, acf_fft, mean)
            res = relative_error(ifft(xfft), ifft(xfft_new))
            xfft = xfft_new
            j += 1
            if callback is not None:
                callback(xfft[0], res, i)
        loss_x = vmap(loss_fft, (0, None))(xfft, yfft)
        idx_best = jnp.argmin(loss_x)
        xfft = xfft[idx_best] + std * random.normal(random.PRNGKey(i), xfft.shape)
        std = std * decay
        i += 1
    # Final optimization
    loss_x = vmap(loss_fft, (0, None))(xfft, yfft)
    idx_best = jnp.argmin(loss_x)
    xfft = xfft[idx_best]
    return xfft




def em_method(x0fft, X, sigma, tol=1e-10, batch_niter=3000, full_niter=10000, callback=None):
    X = X.T
    d, N = X.shape
    #print(d, N, x0fft.shape)
    assert x0fft.shape[0] == d, 'Initial guess x must have length N.'

    # In practice, we iterate with the DFT of the signal x
    xfft = x0fft

    # Precomputations on the observations
    Xfft = fft(X, axis=0)
    sqnormX = jnp.sum(jnp.abs(X)**2, axis=0)[None, :]

    # If the number of observations is large, get started with iterations
    # over only a sample of the observations
    t  = time.time()
    if N >= 3000:
        batch_size = 1000
        for _ in range(batch_niter):
            sample = random.randint(random.PRNGKey(2), (batch_size,), 0, N)
            xfft_new = em_iteration(xfft, Xfft[:, sample], sqnormX[:, sample], sigma)
            xfft = xfft_new
    #print(f'Time for batch iterations: {time.time() - t}\n')

    # In any case, finish with full passes on the data
    for iter in range(full_niter):
        xfft_new = em_iteration(xfft, Xfft, sqnormX, sigma)
        if relative_error(ifft(xfft), ifft(xfft_new)) < tol:
            break
        xfft = xfft_new
        if callback is not None:
            callback(xfft, jnp.linalg.norm(xfft-xfft_new), iter)

    x = ifft(xfft).real
    return x

@jit
def em_iteration(fftx, fftX, sqnormX, sigma):
    # (d - dimension of signal, N - samples)
    C = ifft(fftx.conj()[:, None] * fftX, axis=0)
    T = (2 * C - sqnormX) / (2 * sigma**2)
    T = T - jnp.max(T, axis=0, keepdims=True)
    W = jnp.exp(T)
    W = W / jnp.sum(W, axis=0, keepdims=True)
    fftx_new = jnp.mean(fft(W, axis=0).conj() * fftX, axis=1)
    return fftx_new

## Phase synchronization

def flip_modes(z):
    return jnp.roll(jnp.flip(z), 1)    

@jit
def circulant(v):
    v_flip = flip_modes(v)
    return toeplitz(v_flip, v)


@jit
def invariants_from_data(X, sigma):
    (N, L) = X.shape
    xmean = jnp.mean(jnp.mean(X))
    Xc = X - xmean
    Xc_fft = fft(Xc, axis=1)
    Xc_auto_fft = jnp.clip(jnp.mean(jnp.abs(Xc_fft)**2., axis=0) - sigma**2 * L, 0, None)
    
    B_est = jnp.zeros((L, L))
    
    def B_row(xm_fft):
        Bm = jnp.multiply((xm_fft[:, None] * jnp.conjugate(xm_fft[None, :])), circulant(xm_fft))
        return Bm
    
    B_est = jnp.mean(vmap(B_row, 0)(Xc_fft), axis=0)
    #for n in range(N):
     #   B_est = B_row(Xc_fft[n, :]) + B_est
    
    #B_est = B_est/N
    return xmean, Xc_auto_fft, B_est

def sign(y):
        return y / jnp.abs(y)

@jit
def symmetrize(u):
    u_rev = flip_modes(u)
    return (jnp.conjugate(u_rev) + u)/2

#@jit
def power_iters(M, x0, tol, maxiter=100):
    val = (x0, tol + 1., 0)
    
    def bdy_func(val):
        x, res, it = val
        z = M @ x
        z = sign(z)
        z = z * jnp.sign(z[0])
        res = jnp.linalg.norm(z - x) / len(x)**0.5
        # define z
        x = z
        it = it + 1
        return (x, res, it)
    
    def cond_func(val):
        _, res, it = val
        return (res > tol) & (it < maxiter)
    
    while cond_func(val):
        val = bdy_func(val)
        #print(f"iter {val[2]}, res {val[1]:.2e}", end="\r")
   # print("\n Done")
    return val[0]# while_loop(cond_func, bdy_func, val)


def iterative_phase_synch(B, z0=None, tol=1e-6, maxiter = None, callback=None):
    """Maxitger =(outer, inner) or inner, in which case outer = 15"""
    if isinstance(maxiter, int):
        inner_maxiter = maxiter
        outer_maxiter = 15
    elif isinstance(maxiter, tuple):
        if (len(maxiter) == 2):
            outer_maxiter, inner_maxiter = maxiter
        else:
            raise ValueError('maxiter should be an integer or a tuple of two integers')
    else:
        inner_maxiter = 1000
        outer_maxiter = 15
        
    
    if z0 is None: 
        z0 = sign(B[0,0])
    elif z0 == 0:
        z0 = 1
    else:
        z0 = sign(z0)
        
    L = B.shape[0]
    B = (B + jnp.conj(B).T) / 2
    
    Mfun = lambda z: jnp.multiply(B, jnp.conj(circulant(z)))
    it = 0
    #np.random.seed(3)
    z = np.random.randn(L) + 1j*np.random.randn(L)
    z[0] = z0
    z = sign(symmetrize(z))
     
    res = tol+1.
    it = 0
    while (it < outer_maxiter) and (res > tol):
        M = Mfun(z)
        znew = power_iters(M, z, tol=tol, maxiter=inner_maxiter)
        znew = znew*sign(z0) / znew[0]
        znew = sign(symmetrize(znew))
        res = jnp.linalg.norm(z - znew) / L ** 0.5
        z = znew
        it=it+1
        if callback is not None:
            callback(z, res, it)
    return z


def bispectrum_inversion(y_mean, y_auto_fft, B, **kwargs):
    L = len(y_auto_fft)
    z = iterative_phase_synch(B, y_mean, **kwargs)
    
    xfft = (y_auto_fft ** 0.5) * z
    xfft = xfft.at[0].set(y_mean * L)
    
    # Hotfix
    xfft2 = (y_auto_fft ** 0.5) * (-z)
    xfft2 = xfft2.at[0].set(y_mean * L)

    x = ifft(xfft).real
    x2 = ifft(xfft2).real
    return x, x2, z

# Run with same initial guess as matlab, check intermediate results
# run with same data 
# Use fix-point to estimate phases
# Use EM to estimate amplitudes (Chi2? what distribution?)
# Check how good fix-point if amplitudes are correct


############## MULTI-SCALE ###################################

def hessian(xfft, y, yfft):
    y_align = vmap(align, (None, 0))(xfft, yfft, y)
    hes_ = jnp.einsum('mi,mj -> ij', y_align, y_align) / y_align.shape[0]
    return hes_ # TODO: check if this is correct

def outer(x, y):
    return jnp.einsum('i,j->ij', x.conj(), y)

def inner(x, y):
    return jnp.einsum('i,i->', x.conj(), y)

def diff_fft(xfft, trunc):
    ks = 1j * jnp.fft.fftfreq(len(xfft), 2*jnp.pi/len(xfft)) * 2 * jnp.pi
    ks = ks * (jnp.abs(ks) < trunc)
    return ks * xfft

def multiscale_optim(method, steps, x0, aux, **kwargs):
    """Multiscale optimization. Function `method` is applied `steps` times, 
    each time with double the resolution. The next output is used as input for the next step"""
    
    N = len(x0)
    v_resize = vmap(resize, (0, None))
    for i in range(steps):
        N_i = (N - 1)//2**(steps - 1 - i) + 1
        x0 = resize(x0, N_i)
        aux_i = [v_resize(a, N_i) if len(a.shape)>1 else resize(a, N_i) for a in aux]
        x0 = method(x0, aux_i, **kwargs)
    return x0


def multiscale_residual_optim(method, steps, x0, aux, **kwargs):
    """Multiscale optimization. Function `method` is applied `steps` times, 
    each time with double the resolution. The next output is used as input for the next step"""
    
    N = len(x0)
    v_resize = vmap(resize, (0, None))
    for i in range(steps):
        N_i = (N - 1)//2**(steps - 1 - i) + 1
        x0 = resize(x0, N_i)
        aux_i = [v_resize(a, N_i) if len(a.shape)>1 else resize(a, N_i) for a in aux]
        x0 = method(x0, aux_i, **kwargs)
    return x0

############# SECOND ORDER METHODS ############################

@jit
def hessian_declarative(xfft, yfft, kmax=5):
    yfft_shift = align_fft(xfft, yfft)
    dxfft = diff_fft(xfft, kmax)    
    dyfft_shift = diff_fft(yfft_shift, kmax)
    #dyfft = diff_fft(yfft, kmax)
    return (jnp.eye(len(xfft)) - outer(dyfft_shift, dyfft_shift) / inner(dyfft_shift, dxfft))#/len(xfft)
    #return jnp.eye(len(xfft))/len(xfft)

def hessian_declarative_real(x, yfft, y):
    y_align = align(fft(x), yfft, y)
    dy_align = jnp.roll(y_align, 1) - jnp.roll(y_align, -1)
    dx = jnp.roll(x, 1) - jnp.roll(x, -1)
    return jnp.eye(len(x)) -  outer(dy_align, dy_align)/ inner(dy_align, dx) 

def hessian_vmap(xfft, yfft, y):
    y_align = vmap(align, (None, 0))(xfft, yfft, y)
    def hessian(v):
        yv = jnp.dot(y_align, v)
        

def approx_newton_declarative(x0fft, yfft, acf_fft, mean, tol, maxiter, callback=None, **kwargs):
    res = tol + 1.
    j = 0
    xfft = x0fft
    while (res > tol) and j < maxiter:
        grad_fft = xfft - align_average_and_project(xfft, yfft, acf_fft, mean)#jnp.mean(vmap(align_fft, (None, 0))(xfft, yfft), axis=0)
        hess_fft = jnp.mean(vmap(hessian_declarative, (None, 0))(xfft, yfft), axis=0) * len(xfft)
        grad_fft = grad_fft - inner(grad_fft, xfft) * xfft / inner(xfft, xfft)
        #hess_fft = jnp.conjugate(hess_fft)
        #print(hess_fft)
        #step_fft = jnp.linalg.solve(hess_fft, grad_fft.conj())
        #step_fft = grad_fft
        
        # KKT system
        hess_kkt_fft = jnp.vstack([jnp.hstack([hess_fft, xfft[:, None]]), jnp.hstack([xfft.conj()[None, :], jnp.array([[0]])])])
        grad_kkt_fft = jnp.vstack([grad_fft[:, None], jnp.array([[0]])])
        step_fft = jnp.linalg.solve(hess_kkt_fft, grad_kkt_fft)[:-1, 0]
        step_fft *= line_search(xfft, step_fft, yfft, acf_fft, mean)
        ratio = inner(step_fft, xfft) / jnp.linalg.norm(xfft) / jnp.linalg.norm(step_fft)
        assert jnp.abs(ratio) < 1e-6, f"Must be in tangent plane, {ratio:.2e}"
        print(f"cos: {inner(grad_fft, step_fft):.2e}", end="\r")
        #print(hess_kkt_fft.shape, grad_kkt_fft.shape, step_fft.shape)
        res = jnp.mean(jnp.abs(step_fft))
        xfft = xfft - step_fft#min(1., 1/jnp.linalg.norm(step_fft))*step_fft
        xfft = project_moments(xfft, acf_fft, mean)
        j += 1
        if callback is not None:
            callback(xfft, res, j)
    # Final optimization
    #print(ifft(xfft).real)
    #print(jnp.linalg.norm(xfft), jnp.linalg.norm(project_moments(xfft, acf_fft, mean)))
    return project_moments(xfft, acf_fft, mean)

def line_search(xfft, step_fft, yfft, acf_fft, mean):
    step_size = 1.
    loss_decrease = True
    xfft_new = project_moments(xfft - step_size*step_fft, acf_fft, mean)
    loss_x = loss_fft(xfft_new, yfft)
    while loss_decrease and step_size > 1e-12:
        step_size = step_size / 2.
        #print(f"decreased: {step_size*2}->{step_size}")
        xfft_new = project_moments(xfft - step_size*step_fft, acf_fft, mean)
        loss_x_temp = loss_fft(xfft_new, yfft)
        loss_decrease = loss_x_temp < loss_x
        loss_x = loss_x_temp
    return step_size

def approx_newton_declarative_real(x0fft, yfft, y, acf_fft, mean, tol, maxiter, num_inits=3, step_size=0.1, callback=None):
    res = tol + 1.
    j = 0
    xfft = x0fft
    while (res > tol) and j < maxiter:
        grad_fft = xfft - jnp.mean(vmap(align_fft, (None, 0))(xfft, yfft), axis=0)
        hess= jnp.mean(vmap(hessian_declarative_real, (None, 0, 0))(ifft(xfft).real, yfft, y), axis=0)
        step = jnp.linalg.solve(hess, ifft(grad_fft).real)
        step_fft = fft(step)
        # KKT system
        res = jnp.mean(jnp.abs(step_fft))
        step_size = line_search(xfft, step_fft, yfft, acf_fft, mean)
        xfft = xfft - step_size*step_fft#min(1., 1/jnp.linalg.norm(step_fft))*step_fft
        xfft = project_moments(xfft, acf_fft, mean)
        j += 1
        if callback is not None:
            callback(xfft, res, j)
    # Final optimization
    return xfft


def approx_newton_iter(x0fft, yfft, acf_fft, mean, tol, maxiter, num_inits=3, decay=0.9, alpha=0.5, callback=None):
    res = tol + 1.
    i = 0
    std = 0.1 # Random initialization std
    xfft = x0fft
    while (res > tol) and i < num_inits:
        j = 0
        while (res > tol) and j < maxiter:
            grad = fft(xfft - vmap(align_average_and_project, (0, None, None, None))(xfft, yfft, acf_fft, mean))
            hess = jnp.mean(vmap(hessian, (0, None))(xfft, yfft), axis=0)
            
            xfft_new = jnp.linalg.solve(hess, grad)
            res = jnp.mean(jnp.abs(xfft - xfft_new))
            xfft = xfft_new
            j += 1
            if callback is not None:
                callback(xfft[0], res, i)
        loss_x = vmap(loss_fft, (0, None))(xfft, yfft)
        idx_best = jnp.argmin(loss_x)
        xfft = xfft[idx_best] + std * random.normal(random.PRNGKey(i), xfft.shape)
        std = std * decay
        i += 1
    # Final optimization
    loss_x = vmap(loss_fft, (0, None))(xfft, yfft)
    idx_best = jnp.argmin(loss_x)
    xfft = xfft[idx_best]
    return xfft