import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.optimize import minimize
import time as tm
import pandas as pd
from scipy.stats import norm
%matplotlib widget

# Seed based on the clock
np.random.seed(int(sum(100 * np.array(list(tm.localtime())))))

# Chapeau function:
def chapeau(dt=0.1):

    # Generate the time values based on the time step
    t_x = np.arange(0, 1 + dt, dt)  # Time values for the input signal x(t), injected between 0 and 1
    t_s = np.arange(0, 8 + dt, dt)  # Time values for the transfer function s(t), defined between 0 and 8

    # Define the input signal x(t) (injected salt tracer between 0 and 1)
    def input_signal(t):
        return np.where((t >= 0) & (t <= 1), t * (1 - t) * np.exp(1 - t) * (np.exp(8) - 41), 0)

    # Define the transfer function s(t) (Chapeau function from 0 to 8)
    def transfer_function(t):
        return np.where((t >= 0) & (t < 4), t, np.where((t >= 4) & (t <= 8), 8 - t, 0))

    # Calculate the input signal and transfer function values
    x_t = input_signal(t_x)
    s_t = transfer_function(t_s)

    # Perform the convolution of x(t) and s(t) using numpy's convolve function with mode='full'
    y_t = np.convolve(x_t, s_t, mode='full') * dt

    # Create the time vector for the convolution result (it should span from 0 to t_x[-1] + t_s[-1])
    t_conv = np.arange(0, len(y_t) * dt, dt)

    # Plot the input signal x(t)
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(t_x, x_t, label='Input Signal x(t)', color='blue')
    plt.title('Input Signal x(t)')
    plt.xlim(0, 9)  # Set x-axis limits to 0 to 9 for consistency
    plt.grid(True)
    plt.legend()

    # Plot the transfer function s(t)
    plt.subplot(3, 1, 2)
    plt.plot(t_s, s_t, label='Transfer Function s(t)', color='orange')
    plt.title('Transfer Function s(t)')
    plt.xlim(0, 9)  # Set x-axis limits to 0 to 9 for consistency
    plt.grid(True)
    plt.legend()

    # Plot the convolution result
    plt.subplot(3, 1, 3)
    plt.plot(t_conv, y_t, label='Convolution of x(t) and s(t)', color='green')
    plt.title('Convolution Result (x(t) * s(t))')
    plt.xlim(0, 9)  # Set x-axis limits to 0 to 9 for consistency
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # make x(t) same length as t_conv
    temp = np.zeros(len(t_conv))
    temp[:len(x_t)] = x_t
    x_t = temp
    
    return t_conv, x_t, s_t, y_t


def neu_and_mars(dt=0.1):
    # Neuman and de Marsily (176) example:
    # Generate the time values based on the time step
    t_x = np.arange(0, 1 + dt, dt)  # Time values for the input signal x(t), injected between 0 and 1
    t_s = np.arange(0, 8 + dt, dt)  # Time values for the transfer function s(t), defined between 0 and 8

    # Define the input signal x(t) (injected salt tracer between 0 and 1)
    def input_signal(t):
        return np.where((t >= 0) & (t <= 1), t * (1 - t) * np.exp(1 - t) * (np.exp(8) - 41), 0)

    # Define the transfer function s(t) (Chapeau function from 0 to 8)
    def transfer_function(t):
        return t * (np.exp(8 - t) - 1) / (np.exp(8) - 41)
    
    def y(t):
        if 0 <= t < 1:
            return ((np.exp(9-t))*t**3*(2-t)/12)+((np.exp(1-t))*(t**2+3*t+4))-(np.exp(4-t))
        elif 1 <= t < 8:
            return ((np.exp(9-t))*(2*t-1)/12)+(11-3*t)-(np.exp(4-t))
        elif 8 <= t <= 9:
            return ((np.exp(9-t))/12)*(12*t**2-130*t+335-(t-8)**2*(152-14*t-t**2))+(11-3*t)
    
    
    # Calculate the input signal and transfer function values
    x_t = input_signal(t_x)
    s_t = transfer_function(t_s)

    # Perform the convolution of x(t) and s(t) using numpy's convolve function with mode='full'
    y_t = np.convolve(x_t, s_t, mode='full') * dt

    # Create the time vector for the convolution result (it should span from 0 to t_x[-1] + t_s[-1])
    t_conv = np.arange(0, len(y_t) * dt, dt)

    # Plot the input signal x(t)
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(t_x, x_t, label='Input Signal x(t)', color='blue')
    plt.title('Input Signal x(t)')
    plt.xlim(0, 9)  # Set x-axis limits to 0 to 9 for consistency
    plt.grid(True)
    plt.legend()

    # Plot the transfer function s(t)
    plt.subplot(3, 1, 2)
    plt.plot(t_s, s_t, label='Transfer Function s(t)', color='orange')
    plt.title('Transfer Function s(t)')
    plt.xlim(0, 9)  # Set x-axis limits to 0 to 9 for consistency
    plt.grid(True)
    plt.legend()

    # Plot the convolution result
    plt.subplot(3, 1, 3)
    plt.plot(t_conv, y_t, label='Convolution of x(t) and s(t)', color='green')
    plt.title('Convolution Result (x(t) * s(t))')
    plt.xlim(0, 9)  # Set x-axis limits to 0 to 9 for consistency
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # make x(t) same length as t_conv
    temp = np.zeros(len(t_conv))
    temp[:len(x_t)] = x_t
    x_t = temp
    
    return t_conv, x_t, s_t, y_t


def bimodal(dt=0.1):
    
    
    t_x = np.linspace(0, 3,500)

    # Parameters for x(t)
    mean_val = 1.5
    sd_val = 0.5
    scale_factor = 1000

    # Compute x(t) as a scaled normal distribution N(1.5, 0.5)
    x_t = norm.pdf(t_x, mean_val, sd_val) * scale_factor

    # Parameters for s(t)
    mean_val_1 = 1
    sd_val_1 = 0.2
    mean_val_2 = 2
    sd_val_2 = 0.2

    # Compute s(t) as the sum of two normal distributions, with the second one divided by 2
    s_t = norm.pdf(t_x, mean_val_1, sd_val_1) + (norm.pdf(t_x, mean_val_2, sd_val_2) / 2)

    # Time values for the convolution from 0 to 6
    t_values_conv = np.linspace(0, 6, 500)
    x_t_extended = np.pad(x_t, (0, len(t_values_conv) - len(t_x)), 'constant')
    s_t_extended = np.pad(s_t, (0, len(t_values_conv) - len(t_x)), 'constant')
    
    y_t_conv_full_corrected = np.convolve(x_t_extended, s_t_extended, mode='full') * (t_values_conv[1] - t_values_conv[0])

    # Normalize the result to ensure the peak value is approximately 800
    scaling_factor = 800 / np.max(y_t_conv_full_corrected)
    y_t_conv_full_corrected *= scaling_factor

    # Generate time values for the full convolution (since 'full' extends the length)
    t_values_conv_full = np.linspace(0, 6, len(y_t_conv_full_corrected))  # Adjust length to match the full convolution

    # Plot the combined graphs with subplots
    fig, ax1 = plt.subplots(2, 1, figsize=(10, 10))

    # First subplot: x(t) and s(t) on the same subplot, with s(t) on a secondary axis
    ax1[0].plot(t_x, x_t, label=r'$x(t)$', color='b')
    ax2 = ax1[0].twinx()
    ax2.plot(t_x, s_t, label=r'$s(t)$', color='g')

    # Customize the first subplot
    ax1[0].set_title('Input Functions $x(t)$ and $s(t)$')
    ax1[0].set_xlabel('t')
    ax1[0].set_ylabel('x(t)', color='b')
    ax2.set_ylabel('s(t)', color='g')
    ax1[0].grid(True)

    # Create legends for both axes
    ax1[0].legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Second subplot: convolution y(t) = x(t) * s(t)
    ax1[1].plot(t_values_conv_full, y_t_conv_full_corrected, label=r'$y(t) = x(t) * s(t)$', color='r')
    ax1[1].set_title('Convolution $y(t) = x(t) * s(t)$ with Corrected Peak')
    ax1[1].set_xlabel('t')
    ax1[1].set_ylabel('y(t)')
    ax1[1].grid(True)
    ax1[1].legend()

    plt.tight_layout()
    plt.show()
    
    t_full = np.arange(0, 6.1, 0.1)

    # Compute x(t) and s(t) for time t <= 3, and set values to zero for t > 3
    x_t_full = np.where(t_full <= 3, norm.pdf(t_full, mean_val, sd_val) * scale_factor, 0)
    s_t_full = np.where(t_full <= 3, norm.pdf(t_full, mean_val_1, sd_val_1) + (norm.pdf(t_full, mean_val_2, sd_val_2) / 2), 0)

    # Compute the convolution y(t) based on the extended arrays
    y_t_full = np.convolve(x_t_full, s_t_full, mode='full') * (t_full[1] - t_full[0])

    # Trim the convolution result to match the time range of 0 to 6
    y_t_full = y_t_full[:len(t_full)]
    
    assert len(t_full) == len(x_t_full) == len(s_t_full) == len(y_t_full), 'Lengths of time, input, and output signals must be equal.'
    
    return t_full, x_t_full, s_t_full, y_t_full
    


def load_gambill_data(ws=os.path.join('medQ'),prefix='medQ_R2'):
    data = pd.read_csv(os.path.join(ws, prefix + '.csv'))
    time = data['time'].values
    in_signal = data['in'].values
    out_signal = data['out'].values
    
    return time, in_signal, out_signal


def py_deconv_dave(time, in_signal, out_signal):
    # Numerical details
    theta = 5 # max cov at 0 lag
    corr_time = 24 
    sigma = 0.231 # correlated noise
    sigma_max = 0.4 # max val of correlated noise
    n_h = 64  # Length of transfer-function vector
    nreal = 10  # Number of realizations

    # Input and output signals
    x = in_signal
    y = out_signal
    t = time
    dt = time[1] - time[0]
    corr_time = min(n_h * dt, corr_time)
    n_corr_time = int(np.ceil(corr_time / dt))

    # Construction of Jacobian (convolution matrix)
    c = dt * x
    r = np.zeros(n_h)
    J = la.toeplitz(c, r)

    theta_old = 0

    while abs(theta_old - theta) / theta > 0.01:
        print(f'Current theta: {theta}, convergence: {abs(theta_old - theta) / theta}')
        theta_old = theta 

        # Generalized covariance matrix construction
        co = np.arange(n_h, 0, -1) * dt * theta  # Olaf's method
        c[:n_corr_time] = np.arange(n_corr_time, 0, -1) * dt * theta  # DAB's approx linear cov function
        print(baba)
        
        Q = la.toeplitz(c)
        C = la.cholesky(Q)
        iQ = la.inv(Q)

        # Vector of indices
        ii = np.arange(n_h)

        # Best estimate
        hL = []
        nL = 0
        iter = 0
        while iter < 40:
            print(f'Iteration {iter}')
            iter += 1
            JRJ = J.T @ J / sigma**2
            u = np.ones(n_h)
            umat = np.block([[JRJ + iQ, JRJ @ u[:, None]], [u.T @ JRJ, u.T @ JRJ @ u[:, None]]])
            urhs = np.concatenate([J.T @ y / sigma**2, [u.T @ J.T @ y / sigma**2]])

            # Matrix related to Lagrange multipliers
            Lmat = np.zeros((n_h + 1, nL))
            Lrhs = np.zeros(nL)

            for j in range(nL):
                Lmat[int(hL[j]), j] = 1
                Lmat[n_h, j] = 1
                Lrhs[j] = 0

            mat = np.block([[umat, Lmat], [Lmat.T, np.zeros((nL, nL))]])
            rhs = np.concatenate([urhs, Lrhs])

            a = np.diag(mat).copy()
            a[n_h + 1:] = 1
            a_inv = np.diag(a ** -1)  # Create the diagonal matrix with 1/a
            imat = np.linalg.inv(a_inv @ mat) @ a_inv
            sol = imat @ rhs
            h_be = sol[:n_h] + sol[n_h]
            
            if np.array(hL).size > 0:  # Only assign if hL is not empty
                hL = np.array(hL, dtype=int) 
                h_be[hL] = 0
            
            nu = sol[n_h + 1:]

            sim = J @ h_be
            sigma = np.sqrt(((y - sim).T @ (y - sim)) / (len(y) - n_h + nL - 1))
            sigma = min(sigma, sigma_max)
            print(f'iteration {iter}, sigma: {sigma}, number of Lagrange mults: {nL}')
            
            hL_old = hL
            hL_add = ii[h_be < 0]
            hL = np.array(hL)
            hL_rem = hL[np.where(nu > 0)]
            hL = np.setdiff1d(hL, hL_rem)
            hL = np.union1d(hL, hL_add)
            nL = len(hL)

            if not np.setdiff1d(hL_old, hL).size and not np.setdiff1d(hL, hL_old).size:
                print('breaking...')
                break

    # Initialize sum of h and sum of h squared
    h_all = np.zeros((n_h, nreal))

    # Loop over all realizations
    for ireal in range(nreal):
        # Unconditional realization
        h_uc = C.T @ np.random.randn(n_h, 1)
        h_uc = h_uc[:, 0]
        # Measurement error
        me = sigma * np.random.randn(len(y))

        hL = []
        nL = 0
        iter = 0
        while iter < 40:
            iter += 1
            # Construction of unconstrained matrix
            JRJ = J.T @ J / sigma**2
            u = np.ones(n_h)
            umat = np.block([[JRJ + iQ, JRJ @ u[:, None]], [u.T @ JRJ, u.T @ JRJ @ u[:, None]]])
            urhs = np.concatenate([J.T @ (y + me) / sigma**2 - JRJ @ h_uc, [u.T @ J.T @ (y + me) / sigma**2 - u.T @ JRJ @ h_uc]])

            # Matrix related to Lagrange multipliers
            Lmat = np.zeros((n_h + 1, nL))
            Lrhs = np.zeros(nL)

            for j in range(nL):
                hL = np.array(hL, dtype=int) 
                Lmat[int(hL[j]), j] = 1
                Lmat[n_h, j] = 1
                Lrhs[j] = -h_uc[hL[j]]

            mat = np.block([[umat, Lmat], [Lmat.T, np.zeros((nL, nL))]])
            rhs = np.concatenate([urhs, Lrhs])

            a = np.diag(mat).copy()
            a[n_h + 1:] = 1
            #mat_inv = la.inv(np.diag(1 / a) @ mat @ np.diag(1 / a))
            a_inv = np.diag(a ** -1)  # Create the diagonal matrix with 1/a
            try:
                imat = np.linalg.inv(a_inv @ mat) @ a_inv  # Try regular inverse
                print(np.shape(imat))
            except np.linalg.LinAlgError:
                print("Matrix is singular, switching to pseudo-inverse.")
                imat = np.linalg.pinv(a_inv @ mat) @ a_inv  # Use pseudo-inverse if regular inverse fails
                print(np.shape(imat))
            sol = imat @ rhs
            h = sol[:n_h] + sol[n_h] + h_uc
            h[hL] = 0
            nu = sol[n_h + 1:]

            sim = J @ h
            print(f'iteration {iter}, number of Lagrange mults: {nL}')

            hL_old = hL
            hL_add = ii[h < 0]
            hL = np.array(hL)
            hL_rem = hL[np.where(nu > 0)]
            hL = np.setdiff1d(hL, hL_rem)
            hL = np.union1d(hL, hL_add)
            nL = len(hL)
        
            # Plot realization
            t_h = dt * np.arange(n_h)
            h_all[:, ireal] = h
            plt.subplot(3, 1, 1)
            plt.plot(t, y + me, '-r', t, sim, '-k')
            plt.xlabel('t [hr]')
            plt.legend(['meas.', 'sim.'])
            plt.title(f'Output after iteration {iter}')

            plt.subplot(3, 1, 2)
            plt.plot(np.arange(n_h) * dt, h, 'k')
            plt.title(f'Transfer function after iteration {iter}')
            plt.xlabel(r'$\tau$ [hr]')
            plt.draw()

            plt.subplot(3, 1, 3)
            plt.plot(t_h, np.mean(h_all[:, :ireal + 1], axis=1), 'r', linewidth=1.5)
            plt.xlabel(r'$\tau$ [hr]')
            percentiles = np.percentile(h_all[:, :ireal + 1], [10, 90], axis=1)
            plt.plot(t_h, percentiles[0, :], 'b', label="10th Percentile")  # 10th percentile
            plt.plot(t_h, percentiles[1, :], 'b', label="90th Percentile")  # 90th percentile
    
            plt.legend(['mean', '10%', '90%', 'min', 'max'])
            
            outdir = os.path.join('test_figs')
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            plt.savefig(os.path.join(outdir, f'iter_{iter}.png'))
            plt.close()
            if not np.setdiff1d(hL_old, hL).size and not np.setdiff1d(hL, hL_old).size:
                print('breaking...')
                break

    # Save the results
    #np.savez(prefix + '_transfer_func_condreal.npz', t_h=t_h, h=h_be, h_all=h_all, theta=theta, sigma=sigma)

    # Stats (trimming for fitting ADE)
    trim = min(n_h, 200)
    h_trim = h[:trim]
    time_trim = dt * np.arange(len(h_trim))

    # Compute m_0, m_1, and variance
    m_0 = dt * np.sum(h_trim)
    m_1 = (dt / m_0) * np.sum(time_trim * h_trim)
    var = (dt / m_0) * np.sum(((time_trim - m_1) ** 2) * h_trim)

    # Compute RMSE
    RMSE = np.sqrt(np.mean((J @ h - y) ** 2))

    dx = 24  # m downstream
    v = dx / m_1  # m/hr
    Disp = var * v**3 / dx  # m^2/hr

    tplot = np.linspace(0, 4, 500)
    InvG = (dx / np.sqrt(2 * np.pi * Disp * tplot**3)) * np.exp((dx - v * tplot) ** 2 / (-2 * Disp * tplot))
    InvG[0] = 0

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_h, np.mean(h_all[:, :ireal + 1], axis=1), 'k', linewidth=1.5)
    ax.set_xlabel(r'$\tau$ [hr]')
    ax.set_ylabel('g [1/hr]')
    ax.plot(t_h, np.percentile(h_all[:, :ireal + 1], 10, axis=1), '--r')
    ax.plot(t_h, np.percentile(h_all[:, :ireal + 1], 90, axis=1), '--r', label='10th and 90th percentiles')
    ax.plot(tplot, InvG, 'b--', label='early times of advect\dispers\nwith best-fit velocity and dispersion coeff')
    ax.legend()
    txt = f"Kernel stats:\nmass: {m_0}\nmean (hr): {m_1}\nVar (hr^2): {var}"
    plt.text(0.01, 1.05, txt, fontsize=12, transform=ax.transAxes,
         bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

    txt = f"ADE parameters:\nv (m/hr): {v}\nD (m^2/hr): {Disp}"
    plt.text(0.51, 1.05, txt, fontsize=12, transform=ax.transAxes,
         bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))



if __name__ == '__main__':
    print('Running deconvolution...')
    
    # run Gambill data:
    time, in_signal, out_signal = load_gambill_data(ws=os.path.join('medQ'),prefix='medQ_R2')
    py_deconv_dave(time, in_signal, out_signal)
    
    # run chapeau function:
    time, in_signal, transfer_fx ,out_signal = chapeau()
    assert len(time) == len(in_signal) == len(out_signal), 'Lengths of time, input, and output signals must be equal.'
    py_deconv_dave(time, in_signal, out_signal)
    
    # bimodal example:
    time, in_signal, transfer_fx ,out_signal = bimodal(dt=0.1)
    assert len(time) == len(in_signal) == len(out_signal), 'Lengths of time, input, and output signals must be equal.'
    py_deconv_dave(time, in_signal, out_signal)
    