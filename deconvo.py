import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib widget 
# uncommment to enable interactive plots (ie., zoom, pan, etc.)
from IPython.display import display, clear_output
import scipy.linalg as la
from scipy.optimize import minimize, fmin
from scipy.stats import norm
import time as tm
import multiprocessing
from multiprocessing import Pool, cpu_count

# Seed based on the clock
np.random.seed(int(sum(100 * np.array(list(tm.localtime())))))

def run_single_realization(args):
    """
    Compute a single realization of the transfer function h.
    This function runs in parallel processes.
    """
    (C, iQ, J, y, sigma, sigma_max, n_h, dt, h_uc, ii) = args

    # Initialization of constraints
    hL = []
    nL = 0
    iter_count = 0
    max_iter = 50

    # Add measurement error for this realization:
    me = sigma * np.random.randn(len(y))

    while iter_count < max_iter:
        iter_count += 1
        JRJ = J.T @ J / sigma**2
        u = np.ones(n_h)
        umat = np.block([[JRJ + iQ, JRJ @ u[:, None]], 
                         [u.T @ JRJ, u.T @ JRJ @ u[:, None]]])
        
        urhs = np.concatenate([J.T @ (y + me) / sigma**2 - JRJ @ h_uc, 
                               [u.T @ J.T @ (y + me) / sigma**2 - u.T @ JRJ @ h_uc]])
        
        # Matrix related to the Lagrange multipliers
        Lmat = np.zeros((n_h + 1, nL))
        Lrhs = np.zeros(nL)
        for j in range(nL):
            Lmat[hL[j], j] = 1
            Lmat[n_h, j] = 1
            Lrhs[j] = -h_uc[hL[j]]

        mat = np.block([[umat, Lmat], 
                        [Lmat.T, np.zeros((nL, nL))]])
        rhs = np.concatenate([urhs, Lrhs])

        a = np.diag(mat).copy()
        a[(n_h + 1):] = 1
        imat = la.inv(np.diag(1 / a) @ mat) @ np.diag(1 / a)
        sol = imat @ rhs
        h = sol[:n_h] + sol[n_h] + h_uc
        h = h.flatten()
        h[hL] = 0
        nu = sol[n_h + 1:].flatten()

        sim = J @ h

        hLold = hL.copy()
        # Set of entries that need Lagrange multiplier
        hLadd = ii[h < 0]
        # Remove entries that don't need a Lagrange multiplier anymore
        hLrem = [hL[j] for j in range(len(nu)) if nu[j] > 0]
        hL = [hv for hv in hL if hv not in hLrem]
        hL = sorted(set(hL) | set(hLadd))
        nL = len(hL)

        # If no change in hL, we have converged
        if set(hLold) == set(hL):
            return h  # Return the computed h for this realization

    # If not converged within max_iter, return what we have
    return h


def chapeau(dt=0.1,ex_nm='chapeau',add_error=True):
    if add_error:
        outdir = os.path.join('test_datasets', ex_nm+'_withNoise')
    else:
        outdir = os.path.join('test_datasets', ex_nm)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
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

    # Lengths of the input signal and transfer function
    n_x = len(x_t)
    n_s = len(s_t)
    n_y = n_x + n_s - 1  # Length of the convolution result

    # Zero-pad x_t and s_t to match the convolution length
    x_padded = np.pad(x_t, (0, n_s - 1), 'constant')
    s_padded = np.pad(s_t, (0, n_x - 1), 'constant')

    # Construct the convolution matrix J using the padded x_t
    c = dt * x_padded  # First column of the Toeplitz matrix
    r = np.zeros(n_y)  # First row of the Toeplitz matrix (zeros)
    J = la.toeplitz(c, r)

    # Convolve:
    y_t = J @ s_padded
    # Time values for the convolution result y(t)
    t_y = np.arange(0, n_y * dt, dt)
    
    # If adding error:
    if add_error:
        # Scale factors for the noise, can be adjusted
        error_scale_x = 0.025  # noise std is 5% of local value for input
        error_scale_y = 0.025  # noise std is 5% of local value for output

        # Add noise to x_padded
        x_noise = np.random.normal(0, np.abs(x_padded) * error_scale_x)
        x_padded = x_padded + x_noise
        x_padded = np.where(x_padded < 0, 0, x_padded)

        # Add noise to y_t
        y_noise = np.random.normal(0, np.abs(y_t) * error_scale_y)
        y_t = y_t + y_noise
        y_t = np.where(y_t < 0, 0, y_t)

    # Generate figures:
    fig, [ax1, ax3] = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(t_y, x_padded, 'b',label='x(t)') # input signal
    ax2 = ax1.twinx()
    ax2.plot(t_s, s_t, '-.g',label='h(t)') # transfer function
    ax3.plot(t_y, y_t, 'k',label='C(t)') # convolution result
    
    # formatting:
    ax1.set_ylabel('I(t)')
    ax2.set_ylabel('h(t)')
    ax3.set_ylabel('C(t)')
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    # combined legend:
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    if add_error:
        ax1.set_title('Input Signal I(t) with Noise and Known Transfer Function h(t) ')
        ax3.set_title('Convolution Result C(t) = I(t) * h(t) with Noise')
    else:
        ax1.set_title('Input Signal h(t) and Transfer Function s(t)')
        ax3.set_title('Convolution Result C(t) = I(t) * h(t)')
    
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    # inside ticks:
    ax1.tick_params(axis='both', direction='in')
    ax2.tick_params(axis='both', direction='in')
    ax3.tick_params(axis='both', direction='in')
    
    ax1.set_xlim(0, 9)
    ax3.set_xlim(0, 9)
    
    ax3.set_xlabel('Time')
    plt.tight_layout()
    
    # make out dataframe:
    df = pd.DataFrame({'time': t_y, 'in': x_padded, 'out': y_t, 'known_transfer_fx': s_padded})
    df.to_csv(os.path.join(outdir, f'{ex_nm}_data.csv'), index=False)
    
    
    if add_error:
        exdir = os.path.join('examples', ex_nm+'_withNoise')
    else:
        exdir = os.path.join('examples', ex_nm)
    if not os.path.exists(exdir):
        os.makedirs(exdir)
        os.makedirs(os.path.join(exdir, 'data'))
    df.to_csv(os.path.join(exdir, 'data', f'{ex_nm}_data.csv'), index=False)
    plt.savefig(os.path.join(exdir, 'data', f'{ex_nm}_fig.png'))
    
    return t_y, x_padded, y_t


def gamma(dt=0.1, ex_nm='gamma',add_error=False):
    if add_error:
        outdir = os.path.join('test_datasets', ex_nm+'_withNoise')
    else:
        outdir = os.path.join('test_datasets', ex_nm)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Generate the time values based on the time step
    t_x = np.arange(0, 1 + dt, dt)  # Time values for the input signal x(t), injected between 0 and 1
    t_s = np.arange(0, 15 + dt, dt)  # Time values for the transfer function s(t), defined between 0 and 15
    
    # Define the input signal x(t) (injected salt tracer between 0 and 1)
    def input_signal(t):
        return np.where((t >= 0) & (t <= 1), t * (1 - t) * np.exp(1 - t) * (np.exp(8) - 41), 0)
    
    # Define the transfer function s(t) (Gamma function)
    def transfer_function(t):
        # Parameters for the gamma function
        k = 4 # Shape parameter
        theta = 0.25  # Scale parameter
        coef = (theta ** k) * np.math.gamma(k)
        return np.where(t >= 0, (t ** (k - 1)) * np.exp(-t / theta) / coef, 0)
    
    # Calculate the input signal and transfer function values
    x_t = input_signal(t_x)
    s_t = transfer_function(t_s)
    
    # Lengths of the input signal and transfer function
    n_x = len(x_t)
    n_s = len(s_t)
    n_y = n_x + n_s - 1  # Length of the convolution result
    
    # Zero-pad x_t and s_t to match the convolution length
    x_padded = np.pad(x_t, (0, n_s - 1), 'constant')
    s_padded = np.pad(s_t, (0, n_x - 1), 'constant')
    
    # Construct the convolution matrix J using the padded x_t
    c = dt * x_padded  # First column of the Toeplitz matrix
    r = np.zeros(n_y)  # First row of the Toeplitz matrix (zeros)
    J = la.toeplitz(c, r)
    
    # Convolve:
    y_t = J @ s_padded
    # Time values for the convolution result y(t)
    t_y = np.arange(0, n_y * dt, dt)
    y_t_no_noise = y_t.copy()
    
    # If adding error:
    if add_error:
        # Scale factors for the noise, can be adjusted
        error_scale_x = 0.05  # noise std is 5% of local value for input
        error_scale_y = 0.05  # noise std is 5% of local value for output

        # Add noise to x_padded
        x_noise = np.random.normal(0, np.abs(x_padded) * error_scale_x)
        x_padded = x_padded + x_noise
        x_padded = np.where(x_padded < 0, 0, x_padded)

        # Add noise to y_t
        y_noise = np.random.normal(0, np.abs(y_t) * error_scale_y)
        y_t = y_t + y_noise
        y_t = np.where(y_t < 0, 0, y_t)
    
    # Generate figures:
    fig, [ax1, ax3] = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(t_x, x_t, 'b', label='I(t)')
    if add_error:
        ax1.plot(t_y, x_padded, '.b', label='I(t) with noise', alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(t_s, s_t, '-.g', label='h(t)')
    
    ax3.plot(t_y, y_t_no_noise, 'k', label='C(t)')
    if add_error:
        ax3.plot(t_y, y_t, '.k', label='C(t) with noise', alpha=0.5)
    
    # Formatting:
    ax1.set_ylabel('I(t)')
    ax2.set_ylabel('h(t)')
    ax3.set_ylabel('C(t)')
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    # Combined legend:
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    ax3.legend(loc='upper right')
    ax1.set_title('Input Signal I(t) and Transfer Function h(t)')
    ax3.set_title('Convolution Result C(t) = I(t) * h(t)')
    
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    # Inside ticks:
    ax1.tick_params(axis='both', direction='in')
    ax2.tick_params(axis='both', direction='in')
    ax3.tick_params(axis='both', direction='in')
    
    ax1.set_xlim(0, t_y[-1])
    ax3.set_xlim(0, t_y[-1])
    
    ax3.set_xlabel('Time')
    plt.tight_layout()
    
    # Make output dataframe:
    df = pd.DataFrame({'time': t_y, 'in': x_padded, 'out': y_t, 'known_transfer_fx': s_padded})
    df.to_csv(os.path.join(outdir, f'{ex_nm}_data.csv'), index=False)
    
    if add_error:
        exdir = os.path.join('examples', ex_nm+'_withNoise')
    else:
        exdir = os.path.join('examples', ex_nm)
    if not os.path.exists(exdir):
        os.makedirs(exdir)
        os.makedirs(os.path.join(exdir, 'data'))
    df.to_csv(os.path.join(exdir, 'data', f'{ex_nm}_data.csv'), index=False)
    plt.savefig(os.path.join(exdir, 'data', f'{ex_nm}_fig.png'))
    
    return t_y, x_padded, y_t


def neu_and_mars(dt=0.1,ex_nm='neu_and_mars',add_error=False):
    if add_error:
        outdir = os.path.join('test_datasets', ex_nm+'_withNoise')
    else:
        outdir = os.path.join('test_datasets', ex_nm)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
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
    #y_t = np.convolve(x_t, s_t, mode='full') * dt

    # Lengths of the input signal and transfer function
    n_x = len(x_t)
    n_s = len(s_t)
    n_y = n_x + n_s - 1  # Length of the convolution result
    
    # Zero-pad x_t and s_t to match the convolution length
    x_padded = np.pad(x_t, (0, n_s - 1), 'constant')
    s_padded = np.pad(s_t, (0, n_x - 1), 'constant')
    
    # Construct the convolution matrix J using the padded x_t
    c = dt * x_padded  # First column of the Toeplitz matrix
    r = np.zeros(n_y)  # First row of the Toeplitz matrix (zeros)
    J = la.toeplitz(c, r)
    
    # Convolve:
    y_t = J @ s_padded
    # Time values for the convolution result y(t)
    t_y = np.arange(0, n_y * dt, dt)
    y_t_no_noise = y_t.copy()
    
    # If adding error:
    if add_error:
        # Scale factors for the noise, can be adjusted
        error_scale_x = 0.05  # noise std is 5% of local value for input
        error_scale_y = 0.05  # noise std is 5% of local value for output

        # Add noise to x_padded
        x_noise = np.random.normal(0, np.abs(x_padded) * error_scale_x)
        x_padded = x_padded + x_noise
        x_padded = np.where(x_padded < 0, 0, x_padded)

        # Add noise to y_t
        y_noise = np.random.normal(0, np.abs(y_t) * error_scale_y)
        y_t = y_t + y_noise
        y_t = np.where(y_t < 0, 0, y_t)
    
    # Generate figures:
    fig, [ax1, ax3] = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(t_x, x_t, 'b', label='I(t)')
    if add_error:
        ax1.plot(t_y, x_padded, '.b', label='I(t) with noise', alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(t_s, s_t, '-.g', label='h(t)')
    
    ax3.plot(t_y, y_t_no_noise, 'k', label='C(t)')
    if add_error:
        ax3.plot(t_y, y_t, '.k', label='C(t) with noise', alpha=0.5)
    
    # Formatting:
    ax1.set_ylabel('I(t)')
    ax2.set_ylabel('h(t)')
    ax3.set_ylabel('C(t)')
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    # Combined legend:
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    ax3.legend(loc='upper right')
    ax1.set_title('Input Signal I(t) and Transfer Function h(t)')
    ax3.set_title('Convolution Result C(t) = I(t) * h(t)')
    
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    # Inside ticks:
    ax1.tick_params(axis='both', direction='in')
    ax2.tick_params(axis='both', direction='in')
    ax3.tick_params(axis='both', direction='in')
    
    ax1.set_xlim(0, t_y[-1])
    ax3.set_xlim(0, t_y[-1])
    
    ax3.set_xlabel('Time')
    plt.tight_layout()
    
    df = pd.DataFrame({'time': t_y, 'in': x_padded, 'out': y_t, 'known_transfer_fx': s_padded})
    df.to_csv(os.path.join(outdir, f'{ex_nm}_data.csv'), index=False)
    
    if add_error:
        exdir = os.path.join('examples', ex_nm+'_withNoise')
    else:
        exdir = os.path.join('examples', ex_nm)
    if not os.path.exists(exdir):
        os.makedirs(exdir)
        os.makedirs(os.path.join(exdir, 'data'))
    df.to_csv(os.path.join(exdir, 'data', f'{ex_nm}_data.csv'), index=False)
    plt.savefig(os.path.join(exdir, 'data', f'{ex_nm}_fig.png'))

    return t_y, x_padded, y_t


def bimodal(dt=0.006, ex_nm='bimodal', add_error=False):
    if add_error:
        outdir = os.path.join('test_datasets', ex_nm + '_withNoise')
    else:
        outdir = os.path.join('test_datasets', ex_nm)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    # Time vector for x(t) and s(t)
    # Using 500 points from 0 to 3
    t_x = np.linspace(0, 3, 150)
    t_s = t_x.copy()  # s(t) defined over the same interval

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

    # Compute s(t) as the sum of two normal distributions
    s_t = norm.pdf(t_s, mean_val_1, sd_val_1) + (norm.pdf(t_s, mean_val_2, sd_val_2) / 2)

    # Time increment
    dt = t_x[1] - t_x[0]

    # Lengths of the signals
    N_x = len(x_t)
    N_s = len(s_t)
    N_y = N_x + N_s - 1

    # Construct the convolution matrix J
    J = np.zeros((N_y, N_s))
    for i in range(N_s):
        J[i:i + N_x, i] = x_t

    # Perform the convolution using matrix multiplication
    y_t = J @ s_t * dt

    # Time vector for y_t
    t_y = np.arange(len(y_t)) * dt
    y_t_no_noise = y_t.copy()

    # Pad x(t) and s(t) to match the length of the convolution result (for saving/plotting)
    x_t_padded = np.pad(x_t, (0, N_y - N_x), 'constant')
    s_t_padded = np.pad(s_t, (0, N_y - N_s), 'constant')

    if add_error:
        # Scale factors for the noise
        error_scale_x = 0.05  # 5% of local value for input
        error_scale_y = 0.05  # 5% of local value for output

        # Add noise to the input before padding
        x_noise = np.random.normal(0, np.abs(x_t) * error_scale_x)
        x_t_noisy = x_t + x_noise
        x_t_noisy = np.where(x_t_noisy < 0, 0, x_t_noisy)

        # Reconstruct J with noisy input
        J_noisy = np.zeros((N_y, N_s))
        for i in range(N_s):
            J_noisy[i:i+N_x, i] = x_t_noisy

        # Convolution with noisy input
        y_t_noisy = J_noisy @ s_t * dt

        # Add noise to the output
        y_noise = np.random.normal(0, np.abs(y_t_noisy) * error_scale_y)
        y_t_noisy = y_t_noisy + y_noise
        y_t_noisy = np.where(y_t_noisy < 0, 0, y_t_noisy)

        # Update x_t_padded and y_t with noisy values
        x_t_padded = np.pad(x_t_noisy, (0, N_y - N_x), 'constant')
        y_t = y_t_noisy

    # Generate figures:
    fig, [ax1, ax3] = plt.subplots(2, 1, figsize=(10, 6))

    # Plot original input
    ax1.plot(t_x, x_t, 'b', label='I(t)')

    # Plot noisy input if add_error
    if add_error:
        ax1.plot(t_y, x_t_padded, '.b', label='I(t) with noise', alpha=0.5)

    # Plot s(t)
    ax2 = ax1.twinx()
    ax2.plot(t_s, s_t, '-.g', label='h(t)')

    # Plot original output
    ax3.plot(t_y, y_t_no_noise, 'k', label='C(t)')

    # Plot noisy output if add_error
    if add_error:
        ax3.plot(t_y, y_t, '.k', label='C(t) with noise', alpha=0.5)

    # Formatting:
    ax1.set_ylabel('I(t)')
    ax2.set_ylabel('h(t)')
    ax3.set_ylabel('C(t)')

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # Combined legend:
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    ax3.legend(loc='upper right')

    ax1.set_title('Input Signal I(t) and Transfer Function h(t)')
    ax3.set_title('Convolution Result C(t) = I(t) * h(t)')

    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    # Inside ticks:
    ax1.tick_params(axis='both', direction='in')
    ax2.tick_params(axis='both', direction='in')
    ax3.tick_params(axis='both', direction='in')

    ax1.set_xlim(0, t_y[-1])
    ax3.set_xlim(0, t_y[-1])

    ax3.set_xlabel('Time')
    plt.tight_layout()

    # Save results to CSV
    df = pd.DataFrame({'time': t_y, 'in': x_t_padded, 'out': y_t, 'known_transfer_fx': s_t_padded})
    df.to_csv(os.path.join(outdir, f'{ex_nm}_data.csv'), index=False)

    if add_error:
        exdir = os.path.join('examples', ex_nm+'_withNoise')
    else:
        exdir = os.path.join('examples', ex_nm)

    if not os.path.exists(exdir):
        os.makedirs(exdir)
        os.makedirs(os.path.join(exdir, 'data'))

    df.to_csv(os.path.join(exdir, 'data', f'{ex_nm}_data.csv'), index=False)
    plt.savefig(os.path.join(exdir, 'data', f'{ex_nm}_fig.png'))

    return t_y, x_t_padded, y_t


def load_gooseff_data(ws=os.path.join('test_datasets','gooseff'),prefix='1'):
    data_in = pd.read_csv(os.path.join(ws, 'in'+ prefix + '.txt'), header=None)
    data_in.columns = ['in']
    time = pd.read_csv(os.path.join(ws, 'time.txt'),header=None)
    time.columns = ['time']
    
    data_out = pd.read_csv(os.path.join(ws, 'out' + prefix + '.txt'), header=None)
    data_out.columns = ['out']
    
    in_signal = data_in['in'].values
    out_signal = data_out['out'].values
    
    outdir = os.path.join('examples',f'goosef_STS_zn{prefix}','data')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    data = pd.DataFrame({'time': time.values.flatten(), 'in': in_signal, 'out': out_signal})
    data.to_csv(os.path.join(outdir, f'{prefix}.csv'), index=False)
    time = data['time'].values
    
    return time, in_signal, out_signal
    

def load_gambill_data(ws=os.path.join('test_datasets','gambill','data'),prefix='medQ_R2'):
    data = pd.read_csv(os.path.join(ws, prefix + '.csv'))
    time = data['time'].values
    in_signal = data['in'].values
    out_signal = data['out'].values
    
    outdir = os.path.join('examples',f'gambill_{prefix}','data')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    data.to_csv(os.path.join(outdir, f'{prefix}.csv'), index=False)
    
    return time, in_signal, out_signal


def deconv(num_dets,time, in_signal, out_signal,fit_ade=False,ex_nm='',method='',add_error=False):
    # elapsed time:
    start_time = tm.time()
    
    # Numerical details
    theta = num_dets['theta'] # A first guess applied linearly to corr_time
    corr_time = num_dets['corr_time'] # Only applies if less than the total h(t) length
    sigma = num_dets['sigma'] # std dev of noise
    sigma_max = num_dets['sigma_max'] # Places a forced maximum on iterated sigma
    nreal = num_dets['nreal'] # number of realizations
    n_h = num_dets['n_h'] # Length of transfer-function vector (dt remains the same)

    if add_error:
        rdir = os.path.join('examples',ex_nm + '_withNoise','results')
        fdir = os.path.join(rdir,'figs')
    else:
        rdir = os.path.join('examples',ex_nm,'results')
        fdir = os.path.join(rdir,'figs')        
    if not os.path.exists(rdir):
        os.makedirs(rdir)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
        
    np.random.seed()   # Initialize random seed

    x = in_signal.copy()
    y = out_signal.copy()
    t = time.copy()

    # Time increment
    dt = t[1] - t[0]
    if n_h is None:
        n_h = int(np.ceil(corr_time / dt))
    
    corr_time = min(n_h * dt, corr_time)  # Adjust corr_time
    n_corr_time = int(np.ceil(corr_time / dt))
    
    # Initial covariance (first guess)
    cov = np.zeros(n_h)
    cov[:n_corr_time] = (theta / n_corr_time) * np.arange(n_corr_time, 0, -1)

    # Construction of Jacobian (convolution matrix)
    co = dt * x
    r = dt * np.zeros(n_h)
    J = la.toeplitz(co, r)
    
    theta_old = 0
    while abs(theta_old - theta) / theta > 0.01:
        
        # Covariance matrix construction methods:
        if method == 'olaf':
            co = np.arange(n_h, 0, -1) * dt * theta  # Olaf's method
        elif method == 'linprox':
            co = np.zeros(n_h) # linear approx of cov function
            co[:n_corr_time] = np.arange(n_corr_time, 0, -1) * dt * theta
        elif method == 'actcov':
            co = cov.copy()  # actual cov function
        else:
            assert False, 'Please provide a valid method for covariance matrix construction.\n' \
                            '\t\tValid methods are: "olaf", "linprox", "actcov"'
        
        # Construction of generalized covariance matrix
        c = co.copy()
        Q = la.toeplitz(c)
        C = la.cholesky(Q)
        iQ = la.inv(Q)

        # Vector of indices
        ii = np.arange(0, n_h)

        # Best estimate
        hL = []
        nL = 0
        iter = 0
       
        while iter < 50:
            iter += 1
            # Construction of unconstrained matrix
            JRJ = J.T @ J / sigma**2
            u = np.ones((n_h, 1))
            umat = np.block([[JRJ + iQ, JRJ @ u], [u.T @ JRJ, u.T @ JRJ @ u]])
            urhs = np.concatenate([J.T @ y / sigma**2, u.T @ J.T @ y / sigma**2])
           
            # Matrix related to the Lagrange multipliers
            Lmat = np.zeros((n_h + 1, nL))
            Lrhs = np.zeros(nL)
            for j in range(nL):
                Lmat[int(hL[j]), j] = 1
                Lmat[n_h, j] = 1
                Lrhs[j] = 0

            mat = np.block([[umat, Lmat], [Lmat.T, np.zeros((nL, nL))]])
            rhs = np.concatenate([urhs, Lrhs])
        
            a = np.diag(mat).copy()
            #a = np.where(a != 0, a, 1)
            a[(n_h + 1):] = 1
            imat = la.inv(np.diag(1 / a) @ mat) @ np.diag(1 / a)
            sol = imat @ rhs
            h_be = sol[:n_h] + sol[n_h]
            h_be = h_be.flatten()
            h_be[hL] = 0
            nu = sol[n_h + 1 : n_h + 1 + nL].flatten()

            sim = J @ h_be
            #sigma = np.sqrt(np.sum((y - sim) ** 2) / (len(y) - n_h + nL - 1))
            sigma = np.sqrt(((y - sim).T @ (y - sim)) / (len(y) - n_h + nL - 1))
            sigma = min(sigma, sigma_max)
            print(f"    Iteration {iter}: sigma = {sigma:.3g}, number of Lagrange multipliers {nL}")
            hLold = hL.copy()
            
            # Set of entries that need Lagrange multiplier
            hLadd = ii[h_be < 0]
            # Remove entries that don't need a Lagrange multiplier anymore
            hLrem = [hL[i] for i in range(len(nu)) if nu[i] > 0]
            hL = [h for h in hL if h not in hLrem]
            hL = sorted(set(hL) | set(hLadd))
            nL = len(hL)
            if not np.setdiff1d(hLold, hL).size and not np.setdiff1d(hL, hLold).size:
                break
     
        # Initialize sum of h and sum of h squared
        h_all = np.zeros((n_h, nreal))
       
        # Loop over all realizations
        ireal = 0
        rmse_list = []
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        while ireal<nreal:
            # Unconditional realization
            h_uc = C.T @ np.random.randn(n_h)
            # Measurement error
            me = sigma * np.random.randn(len(y))
            # Initialization of constraints
            hL = []
            nL = 0
            iter = 0
            break_while = False
            while iter < 50:
                if break_while:
                    break
                iter += 1
                # Construction of unconstrained matrix
                JRJ = J.T @ J / sigma**2
                u = np.ones(n_h)
                umat = np.block([[JRJ + iQ, JRJ @ u[:, None]], [u.T @ JRJ, u.T @ JRJ @ u[:, None]]])
                urhs = np.concatenate([J.T @ (y + me) / sigma**2 - JRJ @ h_uc, [u.T @ J.T @ (y + me) / sigma**2 - u.T @ JRJ @ h_uc]])
                # Matrix related to the Lagrange multipliers
                Lmat = np.zeros((n_h + 1, nL))
                Lrhs = np.zeros(nL)
                for j in range(nL):
                    Lmat[hL[j], j] = 1
                    Lmat[n_h, j] = 1
                    Lrhs[j] = -h_uc[hL[j]]

                mat = np.block([[umat, Lmat], [Lmat.T, np.zeros((nL, nL))]])
                rhs = np.concatenate([urhs, Lrhs])

                a = np.diag(mat).copy()
                a[(n_h + 1):] = 1
                imat = la.inv(np.diag(1 / a) @ mat) @ np.diag(1 / a)
                sol = imat @ rhs
                h = sol[:n_h] + sol[n_h] + h_uc
                h = h.flatten()
                h[hL] = 0
                nu = sol[n_h + 1:].flatten()
                sim = J @ h
                
                hLold = hL.copy()
                # Set of entries that need Lagrange multiplier
                hLadd = ii[h < 0]
                # Remove entries that don't need a Lagrange multiplier anymore
                hLrem = [hL[i] for i in range(len(nu)) if nu[i] > 0]
                hL = [h for h in hL if h not in hLrem]
                hL = sorted(set(hL) | set(hLadd))
                nL = len(hL)
                
                t_h = dt * np.arange(n_h)
                # Clear each axis before plotting
                for ax in axes:
                    ax.cla()
                
                axes[0].plot(t, y + me, '-r',label='measured')
                axes[0].plot(t, sim, '--k',label='simulated')
                axes[0].set_ylabel('Concentration, C(t)')
                axes[0].set_title(f'{method.title()} - iteration {ireal+1}, subiteration {iter}')
                axes[0].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
                axes[0].tick_params(axis='both', direction='in',top=True,right=True)
                if iter == 0:
                    axes[0].legend() 
                
                axes[1].plot(np.arange(n_h) * dt, h, 'k',label='transfer fx, h(t)',linewidth=.5,alpha=.5)
                axes[1].set_title(f'Transfer function after subiteration {iter}')
                axes[1].tick_params(axis='both', direction='in',top=True,right=True)
                axes[1].set_ylabel('Transfer function\nh(t)')
                
                axes[2].set_xlim(axes[1].get_xlim())
                axes[2].set_ylim(axes[1].get_ylim())
                axes[2].set_ylabel('Current Best-Estimate \n Transfer function h(t)')
                
                # Compute RMSE
                current_mean_h = np.mean(h_all[:, :ireal+1], axis=1)
                current_sim = J @ current_mean_h
                current_rmse = np.sqrt(np.mean((current_sim - y) ** 2))
                rmse_list.append(current_rmse)    
                
                if set(hLold) == set(hL):
                    # Save realization if converged
                    h_all[:, ireal] = h
                    ireal += 1
                    
                    axes[2].plot(t_h, np.mean(h_all[:, :ireal + 1], axis=1), 'r', linewidth=1.5)
                    #plt.xlabel(r'$\tau$ [hr]')
                    percentiles = np.percentile(h_all[:, :ireal + 1], [10, 90], axis=1)
                    axes[2].plot(t_h, percentiles[0, :], '--b', label="10th Percentile")  # 10th percentile
                    axes[2].plot(t_h, percentiles[1, :], '--b', label="90th Percentile")  # 90th percentile
                    axes[2].legend(['mean', '10%', '90%', 'min', 'max'])
                    #axes[2].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
                    axes[2].tick_params(axis='both', direction='in',top=True,right=True)
                    
                    break_while = True # need this for proper plotting
                elapsed_time = tm.time() - start_time
                if elapsed_time > 15*60:
                    break_while = True
                    print(f'Elapsed time: {elapsed_time:.2f} seconds')
                display(fig)
                clear_output(wait=True)
          
        
        # Plot RMSE over realizations
        # plt.figure()
        # plt.plot(range(1, len(rmse_list) + 1), rmse_list, '-o')
        # plt.xlabel('Number of Realizations')
        # plt.ylabel('RMSE')
        # plt.title('RMSE vs. Number of Realizations')
        # plt.tight_layout()
        # plt.savefig(os.path.join(sdir, f'rmse_over_realizations_{ireal}.png'))
        # plt.close()
        
        
        # Define sumprob function
        def sumprob(lntheta):
            theta = np.exp(lntheta)
            lnpsum = 0
            for i in range(nreal):
                h_i = h_all[:, i]
                nnz = np.sum(h_i > 0)
                ng = len(h_i)
                nz = ng - nnz
                lnp_i = -nnz / 2 * np.log(4 * np.pi * theta) - (ng - 1) / 2 * np.log(1)
                for jj in range(ng - 1):
                    lnp_i -= (h_i[jj + 1] - h_i[jj]) ** 2 / (4 * theta)
                lnpsum -= lnp_i
            return lnpsum
            
        theta_old = theta
        res = fmin(lambda lntheta: sumprob(lntheta), np.log(theta), disp=False)
        theta = np.exp(res[0])
        h = h_be
        
        if method != 'olaf':
            h_mean = np.mean(h_all[:, :ireal], axis=1)
            mean_vector = np.mean(h_all[:, :ireal], axis=1)
            var_exp = np.var(mean_vector)
            zeros_vector = np.zeros_like(h) 
            mean_vector = mean_vector.flatten()
            zeros_vector = zeros_vector.flatten()
            h_mean = np.concatenate((mean_vector, zeros_vector), axis=0)
            
            h_fft = np.fft.fft(h_mean)
            cov = (1 / len(h)) * np.fft.ifft(h_fft * np.conj(h_fft))
            cov = np.real(cov[:len(h)])
            cov = np.maximum(0, cov)
            int_cov = dt * np.sum(cov)
            cov = (int_cov / (dt * np.sum(cov))) * cov
            theta = cov[0]
            corr_time = min(n_h * dt, 2 * int_cov / theta)
            theta = max(theta, 2 * int_cov / corr_time)
            n_corr_time = int(np.ceil(corr_time / dt))
        
            fig_cov, ax_cov = plt.subplots(figsize=(8, 6))
            ax_cov.plot(dt * np.arange(len(h)), cov, 'o', label='C(s)=<g(t+s)g(t)>')
            ax_cov.plot(dt * np.arange(n_corr_time),
                    (theta / (n_corr_time - 1)) * np.arange(n_corr_time - 1, -1, -1), '+-', label='Linear approx.')
            ax_cov.legend()
            ax_cov.set_xlabel('Time lag s (hr)')
            # ax.set_ylabel('C(s) (1/hr^2)') make C(s) italic and times new roman:
            ax_cov.set_ylabel(r'$\mathit{C(s)}$ (1/hr$^2$)')
            ax_cov.set_title(f'Final estimated autocovariance function - Method: {method.title()}')
            # inside ticks:
            ax_cov.tick_params(axis='both', direction='in',top=True,right=True)
        else:
            h_mean = np.mean(h_all[:, :ireal], axis=1)
            mean_vector = np.mean(h_all[:, :ireal], axis=1)
            var_exp = np.var(mean_vector)
            zeros_vector = np.zeros_like(h) 
            mean_vector = mean_vector.flatten()
            zeros_vector = zeros_vector.flatten()
            h_mean = np.concatenate((mean_vector, zeros_vector), axis=0)
            
            fig_cov, ax_cov = plt.subplots(figsize=(8, 6))
            olaf = np.arange(n_h, 0, -1) * dt * theta  # Olaf's method
            ax_cov.plot(dt * np.arange(len(h)), olaf, '+-', label='Olaf method')
            ax_cov.legend()
            ax_cov.set_xlabel('Time lag s (hr)')
            # ax.set_ylabel('C(s) (1/hr^2)') make C(s) italic and times new roman:
            ax_cov.set_ylabel(r'$\mathit{C(s)}$ (1/hr$^2$)')
            ax_cov.set_title(f'Final estimated autocovariance function - Method: {method.title()}')
            # inside ticks:
            ax_cov.tick_params(axis='both', direction='in',top=True,right=True)     
            
        plt.tight_layout()
        plt.savefig(os.path.join(fdir, f'{ex_nm}_autocovariance.png'))
        print(f'Current theta: {theta}, convergence if : {abs(theta_old - theta) / theta} < 0.01')
        
    # Some stats
    trim_time = min(dt * n_h, 30)
    n_trim = min(n_h, 1 + int(np.ceil(trim_time / dt)))
    h_trim = h_mean[:n_trim]
    time_trim = dt * np.arange(len(h_trim))
    time_trim[0] = 1e-10
    m_0 = dt * np.sum(h_trim)
    m_1 = (dt / m_0) * np.sum(time_trim * h_trim)
    var_exp = (dt / m_0) * np.sum(((time_trim - m_1) ** 2) * h_trim)
    RMSE = np.sqrt(np.mean((J @ h - y) ** 2))

    # Plot the transfer function
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_h, np.mean(h_all[:, :ireal + 1], axis=1), 'k', linewidth=1.5)
    ax.plot(t_h, np.percentile(h_all[:, :ireal], 10, axis=1), '--r')
    ax.plot(t_h, np.percentile(h_all[:, :ireal], 90, axis=1), '--r', label='10th and 90th percentiles')
    ax.legend()
    ax.set_title('Simulated Transfer Function')
    ax.set_xlabel(r'$\tau$ [hr]')
    ax.set_ylabel('h [1/hr]')
    ax.tick_params(axis='both', direction='in',top=True,right=True)
    txt = f"Kernel stats:\nmass: {np.round(m_0, 5)}\nmean: {np.round(m_1, 5)}\nvariance: {np.round(var_exp, 5)}\nRMSE: {np.round(RMSE, 5)}"
    ax.set_xlim([0, np.max(t_h)])
    ax.set_ylim([0, np.max(np.mean(h_all[:, :ireal + 1]+0.5, axis=1))])
    plt.text(0.01, -.35, txt, fontsize=12, transform=ax.transAxes,
         bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

    if fit_ade:
        # Some reach details
        dx = 35  # m downstream of the upstream (only needed for ADE)
            # Generate an Inverse Gaussian with same parameters
        time_inv = np.linspace(0, (n_trim - 1) * dt, 1000)
        time_inv[0] = 1e-10
        h_inv = np.interp(time_inv, time_trim, h_trim)
        v = dx / m_1  # m/hr
        Disp = var_exp * v ** 3 / dx / 2  # m^2/hr

        # Best-fit using optimization
        def obj_func(params):
            V, D = params
            fun = (dx / np.sqrt(4 * np.pi * D * time_inv ** 3)) * np.exp(-(dx - V * time_inv) ** 2 / (4 * D * time_inv))
            return np.sum((fun - h_inv) ** 2)

        res = minimize(obj_func, [v, Disp], bounds=[(0, None), (0, None)])
        v_opt, Disp_opt = res.x

        tplot = np.linspace(0, 1, 500)
        dt_plot = tplot[1] - tplot[0]
        InvG = (dx / np.sqrt(4 * np.pi * Disp_opt * tplot ** 3)) * np.exp(-(dx - v_opt * tplot) ** 2 / (4 * Disp_opt * tplot))
        InvG[0] = 0
        txt = f"ADE parameters:\nv (m/hr): {np.round(v, 5)}\nDisp (m^2/hr): {np.round(Disp, 5)}"
        plt.text(0.51, -.35, txt, fontsize=12, transform=ax.transAxes,
            bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(fdir, f'{ex_nm}_simulated_transfer_fx.png'))
    
    # save results to csv:
    pct10 = np.percentile(h_all[:, :ireal], 10, axis=1)
    pct90 = np.percentile(h_all[:, :ireal], 90, axis=1)
    
    df = pd.DataFrame({'time': t_h, 'transfer_fx': np.mean(h_all[:, :ireal + 1], axis=1), '10th_pct': pct10, '90th_pct': pct90})
    df_stats = pd.DataFrame({'mass': m_0, 'mean': m_1, 'variance': var_exp, 'RMSE': RMSE}, index=[0])
    linprox = (theta / (n_corr_time - 1)) * np.arange(n_corr_time - 1, -1, -1)
    #df_cov = pd.DataFrame({'time': dt * np.arange(len(h)), 'cov': cov, 'linprox': linprox})
    
    df.to_csv(os.path.join(rdir, f'{ex_nm}_transfer_fx.csv'), index=False)
    df_stats.to_csv(os.path.join(rdir, f'{ex_nm}_stats.csv'), index=False)
    #df_cov.to_csv(os.path.join(rdir, f'{ex_nm}_cov.csv'), index=False)
    
    return df, df_stats#, df_cov


def deconv_parallel(num_dets, time, in_signal, out_signal, fit_ade=False, ex_nm='', method='', add_error=False):
    # elapsed time:
    start_time = tm.time()

    # Numerical details
    theta = num_dets['theta'] # A first guess applied linearly to corr_time
    corr_time = num_dets['corr_time'] # Only applies if less than the total h(t) length
    sigma = num_dets['sigma'] # std dev of noise
    sigma_max = num_dets['sigma_max'] # Places a forced maximum on iterated sigma
    nreal = num_dets['nreal'] # number of realizations
    n_h = num_dets['n_h'] # Length of transfer-function vector (dt remains the same)

    if add_error:
        rdir = os.path.join('examples',ex_nm + '_withNoise','results')
        fdir = os.path.join(rdir,'figs')
    else:
        rdir = os.path.join('examples',ex_nm,'results')
        fdir = os.path.join(rdir,'figs')        
    if not os.path.exists(rdir):
        os.makedirs(rdir)
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    np.random.seed()   # Initialize random seed

    x = in_signal.copy()
    y = out_signal.copy()
    t = time.copy()

    # Time increment
    dt = t[1] - t[0]
    if n_h is None:
        n_h = int(np.ceil(corr_time / dt))
    
    corr_time = min(n_h * dt, corr_time)  # Adjust corr_time
    n_corr_time = int(np.ceil(corr_time / dt))
    
    # Initial covariance (first guess)
    cov = np.zeros(n_h)
    cov[:n_corr_time] = (theta / n_corr_time) * np.arange(n_corr_time, 0, -1)

    # Construction of Jacobian (convolution matrix)
    co = dt * x
    r = dt * np.zeros(n_h)
    J = la.toeplitz(co, r)

    theta_old = 0
    while abs(theta_old - theta) / theta > 0.01:
        
        # Covariance matrix construction methods:
        if method == 'olaf':
            co = np.arange(n_h, 0, -1) * dt * theta  # Olaf's method
        elif method == 'linprox':
            co = np.zeros(n_h) # linear approx of cov function
            co[:n_corr_time] = np.arange(n_corr_time, 0, -1) * dt * theta
        elif method == 'actcov':
            co = cov.copy()  # actual cov function
        else:
            assert False, 'Please provide a valid method for covariance matrix construction.\n' \
                            '\t\tValid methods are: "olaf", "linprox", "actcov"'
        
        # Construction of generalized covariance matrix
        c = co.copy()
        Q = la.toeplitz(c)
        C = la.cholesky(Q)
        iQ = la.inv(Q)

        # Vector of indices
        ii = np.arange(0, n_h)

        # Best estimate (without realizations)
        # Solve unconstrained system and adjust sigma until stable:
        iter = 0
        hL = []
        nL = 0
        while iter < 70:
            iter += 1
            JRJ = J.T @ J / sigma**2
            u = np.ones((n_h, 1))
            umat = np.block([[JRJ + iQ, JRJ @ u], [u.T @ JRJ, u.T @ JRJ @ u]])
            urhs = np.concatenate([J.T @ y / sigma**2, u.T @ J.T @ y / sigma**2])

            Lmat = np.zeros((n_h + 1, nL))
            Lrhs = np.zeros(nL)
            for j in range(nL):
                Lmat[hL[j], j] = 1
                Lmat[n_h, j] = 1
                Lrhs[j] = 0

            mat = np.block([[umat, Lmat], [Lmat.T, np.zeros((nL, nL))]])
            rhs = np.concatenate([urhs, Lrhs])

            a = np.diag(mat).copy()
            a[(n_h + 1):] = 1
            imat = la.inv(np.diag(1 / a) @ mat) @ np.diag(1 / a)
            sol = imat @ rhs
            h_be = sol[:n_h] + sol[n_h]
            h_be = h_be.flatten()
            h_be[hL] = 0
            nu = sol[n_h + 1 : n_h + 1 + nL].flatten()

            sim = J @ h_be
            sigma = np.sqrt(((y - sim).T @ (y - sim)) / (len(y) - n_h + nL - 1))
            sigma = min(sigma, sigma_max)
            hLold = hL.copy()
            # Set of entries that need Lagrange multiplier
            hLadd = ii[h_be < 0]
            # Remove entries that don't need a Lagrange multiplier anymore
            hLrem = [hL[i] for i in range(len(nu)) if nu[i] > 0]
            hL = [h for h in hL if h not in hLrem]
            hL = sorted(set(hL) | set(hLadd))
            nL = len(hL)
            if not np.setdiff1d(hLold, hL).size and not np.setdiff1d(hL, hLold).size:
                break

        # Parallel computation of realizations
        h_all = np.zeros((n_h, nreal))

        # Prepare arguments for parallel runs
        realizations_args = []
        for _ in range(nreal):
            h_uc = C.T @ np.random.randn(n_h)  # unconditional realization
            # We'll add measurement error inside run_single_realization
            args = (C, iQ, J, y, sigma, sigma_max, n_h, dt, h_uc, ii)
            realizations_args.append(args)

        # Run in parallel
        with Pool(cpu_num) as p:
            results = p.map(run_single_realization, realizations_args)

        for i, h_res in enumerate(results):
            h_all[:, i] = h_res

        # Estimate theta:
        def sumprob(lntheta):
            theta_test = np.exp(lntheta)
            lnpsum = 0
            for i in range(nreal):
                h_i = h_all[:, i]
                nnz = np.sum(h_i > 0)
                ng = len(h_i)
                # from code: lnp_i etc.
                lnp_i = -nnz / 2 * np.log(4 * np.pi * theta_test) - (ng - 1) / 2 * np.log(1)
                for jj in range(ng - 1):
                    lnp_i -= (h_i[jj + 1] - h_i[jj]) ** 2 / (4 * theta_test)
                lnpsum -= lnp_i
            return lnpsum

        theta_old = theta
        res = fmin(lambda lntheta: sumprob(lntheta), np.log(theta), disp=False)
        theta = np.exp(res[0])
        h = h_be

        if method != 'olaf':
            h_mean = np.mean(h_all, axis=1)
            mean_vector = h_mean.flatten()
            zeros_vector = np.zeros_like(h) 
            h_mean_concat = np.concatenate((mean_vector, zeros_vector), axis=0)
            
            h_fft = np.fft.fft(h_mean_concat)
            cov = (1 / len(h)) * np.fft.ifft(h_fft * np.conj(h_fft))
            cov = np.real(cov[:len(h)])
            cov = np.maximum(0, cov)
            int_cov = dt * np.sum(cov)
            cov = (int_cov / (dt * np.sum(cov))) * cov
            theta = cov[0]
            corr_time = min(n_h * dt, 2 * int_cov / theta)
            theta = max(theta, 2 * int_cov / corr_time)
            n_corr_time = int(np.ceil(corr_time / dt))
        
            fig_cov, ax_cov = plt.subplots(figsize=(8, 6))
            ax_cov.plot(dt * np.arange(len(h)), cov, 'o', label='C(s)=<g(t+s)g(t)>')
            ax_cov.plot(dt * np.arange(n_corr_time),
                        (theta / (n_corr_time - 1)) * np.arange(n_corr_time - 1, -1, -1), '+-', label='Linear approx.')
            ax_cov.legend()
            ax_cov.set_xlabel('Time lag s (hr)')
            ax_cov.set_ylabel(r'$\mathit{C(s)}$ (1/hr$^2$)')
            ax_cov.set_title(f'Final estimated autocovariance function - Method: {method.title()}')
            ax_cov.tick_params(axis='both', direction='in',top=True,right=True)
        else:
            h_mean = np.mean(h_all, axis=1)
            mean_vector = h_mean.flatten()
            zeros_vector = np.zeros_like(h) 
            h_mean_concat = np.concatenate((mean_vector, zeros_vector), axis=0)
            
            fig_cov, ax_cov = plt.subplots(figsize=(8, 6))
            olaf = np.arange(n_h, 0, -1) * dt * theta  # Olaf's method
            ax_cov.plot(dt * np.arange(len(h)), olaf, '+-', label='Olaf method')
            ax_cov.legend()
            ax_cov.set_xlabel('Time lag s (hr)')
            ax_cov.set_ylabel(r'$\mathit{C(s)}$ (1/hr$^2$)')
            ax_cov.set_title(f'Final estimated autocovariance function - Method: {method.title()}')
            ax_cov.tick_params(axis='both', direction='in',top=True,right=True)     
        
        plt.tight_layout()
        plt.savefig(os.path.join(fdir, f'{ex_nm}_autocovariance.png'))
        print(f'Current theta: {theta}, convergence if : {abs(theta_old - theta) / theta} < 0.01')

    # Some stats
    trim_time = min(dt * n_h, 30)
    n_trim = min(n_h, 1 + int(np.ceil(trim_time / dt)))
    h_trim = h_mean[:n_trim]
    time_trim = dt * np.arange(len(h_trim))
    time_trim[0] = 1e-10
    m_0 = dt * np.sum(h_trim)
    m_1 = (dt / m_0) * np.sum(time_trim * h_trim)
    var_exp = (dt / m_0) * np.sum(((time_trim - m_1) ** 2) * h_trim)
    RMSE = np.sqrt(np.mean((J @ h - y) ** 2))

    # Plot the transfer function
    t_h = dt * np.arange(n_h)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_h, np.mean(h_all, axis=1), 'k', linewidth=1.5)
    ax.plot(t_h, np.percentile(h_all, 10, axis=1), '--r')
    ax.plot(t_h, np.percentile(h_all, 90, axis=1), '--r', label='10th and 90th percentiles')
    ax.legend()
    ax.set_title('Simulated Transfer Function')
    ax.set_xlabel(r'$\tau$ [hr]')
    ax.set_ylabel('h [1/hr]')
    ax.tick_params(axis='both', direction='in',top=True,right=True)
    txt = f"Kernel stats:\nmass: {np.round(m_0, 5)}\nmean: {np.round(m_1, 5)}\nvariance: {np.round(var_exp, 5)}\nRMSE: {np.round(RMSE, 5)}"
    ax.set_xlim([0, np.max(t_h)])
    ax.set_ylim([0, np.max(np.mean(h_all, axis=1)+0.5)])
    plt.text(0.01, -.35, txt, fontsize=12, transform=ax.transAxes,
         bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

    if fit_ade:
        # Some reach details
        dx = 35  # m downstream (only needed for ADE)
        time_inv = np.linspace(0, (n_trim - 1) * dt, 1000)
        time_inv[0] = 1e-10
        h_inv = np.interp(time_inv, time_trim, h_trim)
        v = dx / m_1  # m/hr
        Disp = var_exp * v ** 3 / dx / 2  # m^2/hr

        def obj_func(params):
            V, D = params
            fun = (dx / np.sqrt(4 * np.pi * D * time_inv ** 3)) * np.exp(-(dx - V * time_inv) ** 2 / (4 * D * time_inv))
            return np.sum((fun - h_inv) ** 2)

        res = minimize(obj_func, [v, Disp], bounds=[(0, None), (0, None)])
        v_opt, Disp_opt = res.x

        tplot = np.linspace(0, 1, 500)
        dt_plot = tplot[1] - tplot[0]
        InvG = (dx / np.sqrt(4 * np.pi * Disp_opt * tplot ** 3)) * np.exp(-(dx - v_opt * tplot) ** 2 / (4 * Disp_opt * tplot))
        InvG[0] = 0
        txt = f"ADE parameters:\nv (m/hr): {np.round(v, 5)}\nDisp (m^2/hr): {np.round(Disp, 5)}"
        plt.text(0.51, -.35, txt, fontsize=12, transform=ax.transAxes,
            bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(fdir, f'{ex_nm}_simulated_transfer_fx.png'))

    # save results to csv:
    pct10 = np.percentile(h_all, 10, axis=1)
    pct90 = np.percentile(h_all, 90, axis=1)

    df = pd.DataFrame({'time': t_h, 'transfer_fx': np.mean(h_all, axis=1), '10th_pct': pct10, '90th_pct': pct90})
    df_stats = pd.DataFrame({'mass': [m_0], 'mean': [m_1], 'variance': [var_exp], 'RMSE': [RMSE]})
    
    df.to_csv(os.path.join(rdir, f'{ex_nm}_transfer_fx.csv'), index=False)
    df_stats.to_csv(os.path.join(rdir, f'{ex_nm}_stats.csv'), index=False)

    return df, df_stats


def rmse_latex_table(exp_dict,fn_out='rmse_methods_table.tex'):
    # build dataframe of all RMSE values:
    data_stor = pd.DataFrame(data=None, columns=['Example Type','RMSE','Method'])
    for key in exp_dict.keys():
        rmse = exp_dict[key]['stats']['RMSE'].values[0]
        method = key.split('_')[-1]
        ex_nm = key.split('_')[0]
        temp = pd.DataFrame(data=[[ex_nm,rmse,method]],columns=['Example Type','RMSE','Method'])
        data_stor = pd.concat([data_stor,temp],axis=0)
        
    data_stor['Example Type'] = data_stor['Example Type'].str.title()
    data_stor.loc[data_stor['Example Type'] == 'Neu','Example Type'] = 'Neumann and Marsili (1976)'
    data_stor.loc[data_stor['Method'] == 'actcov','Method'] = 'Actual Covariance RMSE'
    data_stor.loc[data_stor['Method'] == 'linprox','Method'] = 'Linear Approximation RMSE'
    data_stor.loc[data_stor['Method'] == 'olaf','Method'] = 'Cirpka et al. 2007 RMSE'
    
    # reshape data_stor:
    data_stor = data_stor.pivot(index='Example Type',columns='Method',values='RMSE')
    
    data_stor_reset = data_stor.reset_index()
    columns_order = ['Example Type'] + sorted([col for col in data_stor_reset.columns if col != 'Example Type'])
    data_stor_reset = data_stor_reset[columns_order]

    # Round RMSE values to a consistent number of decimal places, e.g., 3 decimals
    for col in data_stor_reset.columns[1:]:
        data_stor_reset[col] = data_stor_reset[col].astype(float).round(3)

    # Generate the LaTeX table
    latex_table = data_stor_reset.to_latex(index=False, 
                                        column_format='l' + 'r' * (len(data_stor_reset.columns) - 1),
                                        caption='RMSE values for different methods',
                                        label='tab:rmse_methods',
                                        escape=False)

    # Print or save the LaTeX table
    latex_table = latex_table.replace('\\begin{table}', '\\begin{table}[ht]\n\\captionsetup{justification=justified,singlelinecheck=false}')

    print(latex_table)

    outdir = os.path.join('examples')
    # Optionally, write the LaTeX table to a .tex file
    with open(os.path.join(outdir,fn_out), 'w') as f:
        f.write(latex_table)
    

if __name__ == '__main__':
    print('Running deconvolution...')
    
    cpu_num = None # number of cpus to use in parallel processing, set to what you have available, otherwise it will use all available cpus
    if cpu_num is None:
        cpu_num = cpu_count()
    
    add_error = False
    # -----------------------------------------------------------------------------------------------
    # RUN EXPERIMENTAL EXAMPLES:
    #--------------------------------------------------------------
    exp_dict = {} # save results into dict
    
    # run chapeau function:
    print('Running chapeau example...')
    time, in_signal,out_signal = chapeau(add_error=add_error)
    num_dets = {'theta': 2, 'corr_time': 9, 'sigma': 0.1, 'sigma_max': 0.15, 'n_h': None, 'nreal': 24}
    assert len(time) == len(in_signal) == len(out_signal), 'Lengths of time, input, and output signals must be equal.'
    
    print('Method: Actual Covariance')
    if num_dets['nreal'] < 20:
        tf, stats = deconv(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='chapeau',method='actcov',add_error=add_error)
    else:
        # run in parallel, for nreal >= 20, single core is faster than parallel for under 20 realizations because time it takes to
        # set up parallel processing is longer than the time it takes to run the deconvolution for 20 realizations.
        tf, stats = deconv_parallel(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='chapeau',method='actcov',add_error=add_error)    
    exp_dict['chapeau_actcov'] = {'tf': tf, 'stats': stats}
    print('Method: Cirpka et al. 2007')
    if num_dets['nreal'] < 20:
        tf, stats = deconv(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='chapeau',method='olaf')
    else:
        tf, stats = deconv_parallel(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='chapeau',method='olaf')
    exp_dict['chapeau_olaf'] = {'tf': tf, 'stats': stats}
    print('Method: Linear Approximation')
    if num_dets['nreal'] < 20:
        tf, stats = deconv(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='chapeau',method='linprox')
    else:
        tf, stats = deconv_parallel(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='chapeau',method='linprox')
    exp_dict['chapeau_linprox'] = {'tf': tf, 'stats': stats}
    
    
    # run gamma function:
    print('Running gamma example...')
    time, in_signal,out_signal = gamma(add_error=add_error)
    num_dets = {'theta': 0.1, 'corr_time': 6, 'sigma': 0.1, 'sigma_max': 0.15, 'n_h': None, 'nreal': 24}
    assert len(time) == len(in_signal) == len(out_signal), 'Lengths of time, input, and output signals must be equal.'
    print('Method: Actual Covariance')
    if num_dets['nreal'] < 20:
        tf, stats = deconv(num_dets,time, in_signal, out_signal,fit_ade=False,ex_nm='gamma',method='actcov',add_error=add_error)
    else:
        tf, stats = deconv_parallel(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='gamma',method='actcov',add_error=add_error)
    exp_dict['gamma_actcov'] = {'tf': tf, 'stats': stats}
    print('Method: Cirpka et al. 2007')
    if num_dets['nreal'] < 20:
        tf, stats = deconv(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='gamma',method='olaf') # fails to converge
    else:
        tf, stats = deconv_parallel(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='gamma',method='olaf')
    exp_dict['gamma_olaf'] = {'tf': tf, 'stats': stats}
    print('Method: Linear Approximation')
    if num_dets['nreal'] < 20:
        tf, stats = deconv(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='gamma',method='linprox')
    else:
        tf, stats = deconv_parallel(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='gamma',method='linprox')
    exp_dict['gamma_linprox'] = {'tf': tf, 'stats': stats}
    
    # run neu and mars:
    print('Running neu and mars example...')
    time, in_signal,out_signal = neu_and_mars(add_error=add_error)
    num_dets = {'theta': 2, 'corr_time': 8, 'sigma':10, 'sigma_max': 20, 'n_h': None, 'nreal': 24}
    assert len(time) == len(in_signal) == len(out_signal), 'Lengths of time, input, and output signals must be equal.'
    print('Method: Actual Covariance')
    if num_dets['nreal'] < 20:
        tf, stats = deconv(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='neu_and_mars',method='actcov',add_error=add_error)
    else:
        tf, stats = deconv_parallel(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='neu_and_mars',method='actcov',add_error=add_error)
    exp_dict['neu_and_mars_actcov'] = {'tf': tf, 'stats': stats}
    print('Method: Cirpka et al. 2007')
    if num_dets['nreal'] < 20:
        tf, stats = deconv(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='neu_and_mars',method='olaf')
    else:
        tf, stats = deconv_parallel(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='neu_and_mars',method='olaf')
    exp_dict['neu_and_mars_olaf'] = {'tf': tf, 'stats': stats}
    print('Method: Linear Approximation')
    if num_dets['nreal'] < 20:
        tf, stats = deconv(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='neu_and_mars',method='linprox')
    else:
        tf, stats = deconv_parallel(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='neu_and_mars',method='linprox')
    exp_dict['neu_and_mars_linprox'] = {'tf': tf, 'stats': stats}
    
    # bimodal example:
    print('Running bimodal example...')
    time, in_signal,out_signal = bimodal(dt=0.1,add_error=add_error)#bimodal(dt=0.006)
    num_dets = {'theta': 0.1, 'corr_time': 6, 'sigma': 0.1, 'sigma_max': 0.15, 'n_h': None, 'nreal': 24}
    assert len(time) == len(in_signal) == len(out_signal), 'Lengths of time, input, and output signals must be equal.'
    print('Method: Actual Covariance')
    if num_dets['nreal'] < 20:
        tf, stats = deconv(num_dets,time, in_signal, out_signal,fit_ade=False,ex_nm='bimodal',method='actcov')
    else:
        tf, stats = deconv_parallel(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='bimodal',method='actcov')
    exp_dict['bimodal_actcov'] = {'tf': tf, 'stats': stats}
    print('Method: Cirpka et al. 2007')
    if num_dets['nreal'] < 20:
        tf, stats = deconv(num_dets,time, in_signal, out_signal,fit_ade=False,ex_nm='bimodal',method='olaf')
    else:
        tf, stats = deconv_parallel(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='bimodal',method='olaf')
    exp_dict['bimodal_olaf'] = {'tf': tf, 'stats': stats}
    print('Method: Linear Approximation')
    if num_dets['nreal'] < 20:
        tf, stats = deconv(num_dets,time, in_signal, out_signal,fit_ade=False,ex_nm='bimodal',method='linprox')
    else:
        tf, stats = deconv_parallel(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='bimodal',method='linprox')
    exp_dict['bimodal_linprox'] = {'tf': tf, 'stats': stats}
    
    
    # write RMSE table to latex:
    rmse_latex_table(exp_dict,fn_out='chapeau_retry_table.tex')
    
    print(asdf)
    #-------------------------------------------------------------------------------------------------------------------
    # RUN FIELD DATA EXAMPLES:
    #----------------------------------------------------------------
    # run Gambill data:
    # num_dets = {'theta': 5, 'corr_time': 24, 'sigma': 0.231, 'sigma_max': 0.4, 'n_h': 64, 'nreal': 10}
    # prefix = 'medQ_R2'
    # time, in_signal, out_signal = load_gambill_data(prefix=prefix)
    # tf, stats = deconv(num_dets, time, in_signal, out_signal,fit_ade=True,ex_nm=f'gambill_{prefix}',method='actcov')
    
    # run gooseff data:
    num_dets = {'theta': 10**-6, 'corr_time': 0.1, 'sigma': 0.4, 'sigma_max': 0.8, 'n_h': 500, 'nreal': 24}
    prefix = '1'
    time, in_signal, out_signal = load_gooseff_data(prefix=prefix)
    assert len(time) == len(in_signal) == len(out_signal), 'Lengths of time, input, and output signals must be equal.'
    if num_dets['nreal'] < 20:
        tf, stats = deconv(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm=f'gooseff_{prefix}',method='actcov')
    else:
        tf, stats = deconv_parallel(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm=f'gooseff_{prefix}',method='actcov')

    
    
    