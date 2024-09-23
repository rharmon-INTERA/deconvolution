import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.optimize import minimize
import scipy.io
from scipy.optimize import fmin
import time as tm
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
#%matplotlib widget
%matplotlib inline


# Seed based on the clock
np.random.seed(int(sum(100 * np.array(list(tm.localtime())))))

# Chapeau function:
def chapeau(dt=0.1):
    ex_nm = 'chapeau'
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

    # Generate figures:
    fig, [ax1, ax3] = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(t_x, x_t, 'b',label='x(t)')
    ax2 = ax1.twinx()
    ax2.plot(t_s, s_t, '-.g',label='h(t)')
    ax3.plot(t_y, y_t, 'k',label='C(t)')
    
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
    
    exdir = os.path.join('examples', ex_nm)
    if not os.path.exists(exdir):
        os.makedirs(exdir)
        os.makedirs(os.path.join(exdir, 'figs'))
        os.makedirs(os.path.join(exdir, 'data'))
    df.to_csv(os.path.join(exdir, 'data', f'{ex_nm}_data.csv'), index=False)
    plt.savefig(os.path.join(exdir, 'figs', f'{ex_nm}_fig.png'))
    
    return t_y, x_padded, y_t


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


def bimodal(dt=0.006):
    # Time vector for x(t) and s(t)
    t_x = np.linspace(0, 3, 500)

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
    s_t = norm.pdf(t_x, mean_val_1, sd_val_1) + (norm.pdf(t_x, mean_val_2, sd_val_2) / 2)

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

    # Plot the signals for visualization
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t_x, x_t)
    plt.title('Input Signal x(t)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(t_x, s_t)
    plt.title('Transfer Function s(t)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.plot(t_y, y_t)
    plt.title('Output Signal y(t) = x(t) * s(t)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

    # pad x(t) and s(t) to match the length of the convolution result
    x_t_padded = np.pad(x_t, (0, N_y - N_x), 'constant')
    s_t_padded = np.pad(s_t, (0, N_y - N_s), 'constant')
    
    # Return the signals and time vectors
    return t_y,x_t_padded,y_t


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


def deconv(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm=''):
    # Numerical details
    theta = num_dets['theta'] # max cov at 0 lag
    corr_time = num_dets['corr_time'] # correlation time
    sigma = num_dets['sigma'] # std dev of noise
    sigma_max = num_dets['sigma_max'] # max std dev of noise
    nreal = num_dets['nreal'] # number of realizations
    n_h = num_dets['n_h'] # number of convolution points

    sdir = os.path.join('examples',ex_nm, 'figs','solve')
    rdir = os.path.join('examples',ex_nm, 'figs','results')
    if not os.path.exists(sdir):
        os.makedirs(sdir)
    if not os.path.exists(rdir):
        os.makedirs(rdir)
        
    
    # Input and output signals
    x = in_signal
    y = out_signal
    t = time
    dt = time[1] - time[0]
    
    if n_h is None:
        max_transfer_time = 8.0  # Maximum expected duration of the transfer function
        n_h = int(np.ceil(max_transfer_time / dt))
    
    corr_time = min(n_h * dt, corr_time)
    n_corr_time = int(np.ceil(corr_time / dt))
    
    # Construction of Jacobian (convolution matrix)
    n_y = len(y)

    # Pad x to length n_y
    x_padded = np.pad(x, (0, n_y - len(x)), 'constant')

    # First column of the Toeplitz matrix for J
    c_J = dt * x_padded[:n_y]
    r_J = np.zeros(n_h)

    # Construct the convolution matrix J
    J = la.toeplitz(c_J, r_J)

    theta_old = 0

    while abs(theta_old - theta) / theta > 0.01:
        print(f'Current theta: {theta}, convergence: {abs(theta_old - theta) / theta}')
        theta_old = theta 
        # Generalized covariance matrix construction
        #co = np.arange(n_h, 0, -1) * dt * theta  # Olaf's method
        #co[:n_corr_time] = np.arange(n_corr_time, 0, -1) * dt * theta  # DAB's approx linear cov function
        
        co = np.zeros(n_h)
        co[:n_corr_time] = np.arange(n_corr_time, 0, -1) * dt * theta
        
        Q = la.toeplitz(co)
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
            print(abba)
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
            # if sigma is nan

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
        fout = os.path.join(sdir, f'realization_{ireal}')
        if not os.path.exists(fout):
            os.makedirs(fout)
        fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(10, 8))
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
            
            
            fig.suptitle(f'Realization {ireal}', fontsize=14)
            
            ax1.plot(t, y + me, '-r',label='measured')
            ax1.plot(t, sim, '--k',label='simulated')
            ax1.set_ylabel('Concentration, C(t)')
            ax1.set_title(f'Output after iteration {iter}')
            ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            ax1.tick_params(axis='both', direction='in',top=True,right=True)
            if iter == 0:
                ax1.legend() 
            ax2.plot(np.arange(n_h) * dt, h, 'k',label='transfer fx, h(t)',linewidth=.5,alpha=.5)
            ax2.set_title(f'Transfer function after iteration {iter}')
            ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            ax2.tick_params(axis='both', direction='in',top=True,right=True)
             
            plt.tight_layout()
            plt.savefig(os.path.join(fout, f'iteration_{iter}.png'))

            if not np.setdiff1d(hL_old, hL).size and not np.setdiff1d(hL, hL_old).size:
                print('breaking...')
                break
        h_all[:, ireal] = h
        
        ax3.plot(t_h, np.mean(h_all[:, :ireal + 1], axis=1), 'r', linewidth=1.5)
        plt.xlabel(r'$\tau$ [hr]')
        percentiles = np.percentile(h_all[:, :ireal + 1], [10, 90], axis=1)
        ax3.plot(t_h, percentiles[0, :], '--b', label="10th Percentile")  # 10th percentile
        ax3.plot(t_h, percentiles[1, :], '--b', label="90th Percentile")  # 90th percentile
        ax3.legend(['mean', '10%', '90%', 'min', 'max'])
        ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax3.tick_params(axis='both', direction='in',top=True,right=True)
        plt.tight_layout()
        plt.savefig(os.path.join(sdir, f'realization_{ireal}.png'))
        plt.close()
        
    # Define the objective function for theta optimization
    def sumprob(lntheta):
        theta_val = np.exp(lntheta)
        lnpsum = 0.0
        for ireal in range(nreal):
            h_i = h_all[:, ireal]
            nnz = np.sum(h_i > 0)
            ng = n_h
            nz = ng - nnz
            lnp_i = -nnz / 2 * np.log(4 * np.pi * theta_val) - (ng - 1) / 2 * np.log(1)
            # Loop over all entries
            for jj in range(ng - 1):
                lnp_i -= (h_i[jj + 1] - h_i[jj]) ** 2 / (4 * theta_val)
            lnpsum -= lnp_i
        return lnpsum

    # Update theta using optimization
    theta_old = theta
    res = minimize(sumprob, np.log(theta), method='Nelder-Mead')
    theta = np.exp(res.x[0])
        
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

    if fit_ade:
        dx = 24  # m downstream
        v = dx / m_1  # m/hr
        Disp = var * v**3 / dx  # m^2/hr

        tplot = np.linspace(0, 4, 500)
        InvG = (dx / np.sqrt(2 * np.pi * Disp * tplot**3)) * np.exp((dx - v * tplot) ** 2 / (-2 * Disp * tplot))
        InvG[0] = 0

    # Plot the transfer function
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_h, np.mean(h_all[:, :ireal + 1], axis=1), 'k', linewidth=1.5)
    ax.plot(t_h, np.percentile(h_all[:, :ireal + 1], 10, axis=1), '--r')
    ax.plot(t_h, np.percentile(h_all[:, :ireal + 1], 90, axis=1), '--r', label='10th and 90th percentiles')
    ax.legend()
    ax.set_title('Simulated Transfer Function')
    ax.set_xlabel(r'$\tau$ [hr]')
    ax.set_ylabel('h [1/hr]')
    ax.tick_params(axis='both', direction='in',top=True,right=True)
    txt = f"Kernel stats:\nmass: {np.round(m_0, 5)}\nmean: {np.round(m_1, 5)}\nvariance: {np.round(var, 5)}\nRMSE: {np.round(RMSE, 5)}"
    
    plt.text(0.01, -.35, txt, fontsize=12, transform=ax.transAxes,
         bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

    if fit_ade:
        txt = f"ADE parameters:\nv (m/hr): {np.round(v, 5)}\nDisp (m^2/hr): {np.round(Disp, 5)}"
        plt.text(0.51, -.35, txt, fontsize=12, transform=ax.transAxes,
            bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.tight_layout()
    plt.savefig(os.path.join(rdir, f'{ex_nm}_simulated_transfer_fx.png'))


def deconv_2(num_dets,time, in_signal, out_signal,fit_ade=False,ex_nm=''):
    # Numerical details
    theta = num_dets['theta'] # A first guess applied linearly to corr_time
    corr_time = num_dets['corr_time'] # Only applies if less than the total h(t) length
    sigma = num_dets['sigma'] # std dev of noise
    sigma_max = num_dets['sigma_max'] # Places a forced maximum on iterated sigma
    nreal = num_dets['nreal'] # number of realizations
    n_h = num_dets['n_h'] # Length of transfer-function vector (dt remains the same)

    sdir = os.path.join('examples',ex_nm, 'figs','solve')
    rdir = os.path.join('examples',ex_nm, 'figs','results')
    if not os.path.exists(sdir):
        os.makedirs(sdir)
    if not os.path.exists(rdir):
        os.makedirs(rdir)
        
    # Some reach details
    dx = 35  # m downstream of the upstream

    np.random.seed()   # Initialize random seed

    x = in_signal.copy()
    y = out_signal.copy()
    t = time.copy()

    # Time increment
    dt = t[1] - t[0]

    if n_h is None:
        max_transfer_time = 8.0  # Maximum expected duration of the transfer function
        n_h = int(np.ceil(max_transfer_time / dt))
    
    corr_time = min(n_h * dt, corr_time)  # Adjust corr_time
    n_corr_time = int(np.ceil(corr_time / dt))
    n_corr_time = 7
    # Initial covariance
    cov = np.zeros(n_h)
    cov[:n_corr_time] = (theta / n_corr_time) * np.arange(n_corr_time, 0, -1)

    # Construction of Jacobian (convolution matrix)
    cc = dt * x
    r = np.zeros(n_h)
    J = la.toeplitz(cc, r)

    # n_y = len(y)
    # x_padded = np.pad(x, (0, n_y - len(x)), 'constant')
    # c_J = dt * x_padded[:n_y]
    # r_J = np.zeros(n_h)
    # J = la.toeplitz(c_J, r_J)
    
    theta_old = 0
    while abs(theta_old - theta) / theta > 0.02:
        # Construction of generalized covariance matrix
        c = cov.copy()
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
        
            a = np.diag(mat)
            a = np.where(a != 0, a, 1)
            imat = la.inv(np.diag(1 / a) @ mat) @ np.diag(1 / a)
            sol = imat @ rhs
            h_be = sol[:n_h] + sol[n_h]
            h_be = h_be.flatten()
            h_be[hL] = 0
            nu = sol[n_h + 1:].flatten()
            sim = J @ h_be
            sigma = np.sqrt(np.sum((y - sim) ** 2) / (len(y) - n_h + nL - 1))
            sigma = min(sigma, sigma_max)
            print(f"Iteration {iter}: sigma = {sigma:.3g}, number of Lagrange multipliers {nL}")
            hLold = hL.copy()
            
            # Set of entries that need Lagrange multiplier
            hLadd = ii[h_be < 0]
            # Remove entries that don't need a Lagrange multiplier anymore
            hLrem = [hL[i] for i in range(len(nu)) if nu[i] > 0]
            hL = [h for h in hL if h not in hLrem]
            hL = sorted(set(hL) | set(hLadd))
            nL = len(hL)
            if not np.setdiff1d(hLold, hL).size and not np.setdiff1d(hL, hLold).size:
                print('breaking...')
                break
     
        # Initialize sum of h and sum of h squared
        h_all = np.zeros((n_h, nreal))

        # Loop over all realizations
        ireal = 0
        while ireal < nreal:
            # Unconditional realization
            h_uc = C.T @ np.random.randn(n_h)
            # Measurement error
            me = sigma * np.random.randn(len(y))
            # Initialization of constraints
            hL = []
            nL = 0
            iter = 0
            while iter < 50:
                iter += 1
                # Construction of unconstrained matrix
                JRJ = J.T @ J / sigma**2
                u = np.ones(n_h)
                #umat = np.block([[JRJ + iQ, JRJ @ u], [u.T @ JRJ, u.T @ JRJ @ u]])
                umat = np.block([[JRJ + iQ, JRJ @ u[:, None]], [u.T @ JRJ, u.T @ JRJ @ u[:, None]]])
                #urhs = np.concatenate([J.T @ (y + me) / sigma**2 - JRJ @ h_uc[:, None],
                #                u.T @ J.T @ (y + me) / sigma**2 - u.T @ JRJ @ h_uc[:, None]])
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
                a = np.where(a != 0, a, 1)
                imat = la.inv(np.diag(1 / a) @ mat) @ np.diag(1 / a)
                sol = imat @ rhs
                h = sol[:n_h] + sol[n_h] + h_uc
                h = h.flatten()
                h[hL] = 0
                nu = sol[n_h + 1:].flatten()
                sim = J @ h
                print(f"Iteration {iter}: number of Lagrange multipliers {nL}")

                # Plotting (optional)
                plt.figure(1)
                plt.clf()
                plt.subplot(3, 1, 1)
                plt.plot(t, y + me, '-r', label='meas.')
                plt.plot(t, sim, '-k', label='sim.')
                plt.xlabel('t [hr]')
                plt.legend()
                plt.title(f'Output after iteration {iter}')
                plt.subplot(3, 1, 2)
                plt.plot(np.arange(n_h) * dt, h, 'k')
                plt.title(f'Transfer function after iteration {iter}')
                plt.xlabel(r'$\tau$ [hr]')
                plt.draw()
                plt.pause(0.001)

                hLold = hL.copy()
                # Set of entries that need Lagrange multiplier
                hLadd = ii[h < 0]
                # Remove entries that don't need a Lagrange multiplier anymore
                hLrem = [hL[i] for i in range(len(nu)) if nu[i] > 0]
                hL = [h for h in hL if h not in hLrem]
                hL = sorted(set(hL) | set(hLadd))
                nL = len(hL)
                if set(hLold) == set(hL):
                    # Save realization if converged
                    h_all[:, ireal] = h
                    plt.subplot(3, 1, 3)
                    t_h = dt * np.arange(n_h)
                    plt.plot(t_h, np.mean(h_all[:, :ireal + 1], axis=1), 'r', linewidth=1.5)
                    plt.xlabel(r'$\tau$ [hr]')
                    plt.plot(t_h, np.percentile(h_all[:, :ireal + 1], [10, 90], axis=1).T, 'b')
                    #plt.plot(t_h, [np.min(h_all[:, :ireal + 1], axis=1),
                    #            np.max(h_all[:, :ireal + 1], axis=1)], 'b:')
                    plt.legend(['mean', '10%', '90%', 'min', 'max'])
                    plt.draw()
                    plt.pause(0.001)
                    ireal += 1
                    break

        theta_old = theta
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

        res = fmin(lambda lntheta: sumprob(lntheta), np.log(theta), disp=False)
        theta = np.exp(res[0])
        h = h_be
        # Save results
        #scipy.io.savemat(f'{prefix}_transfer_func_condreal.mat', {'t_h': t_h, 'h': h, 'h_all': h_all, 'theta': theta, 'sigma': sigma})

        h_mean = np.mean(h_all[:, :ireal], axis=1)
        h_fft = np.fft.fft(np.concatenate([h_mean, np.zeros_like(h_mean)]))
        cov = (1 / len(h)) * np.fft.ifft(h_fft * np.conj(h_fft))
        cov = np.real(cov[:len(h_mean)])
        int_cov = dt * np.sum(cov)
        theta = cov[0]
        var_exp = np.var(h_mean)
        corr_time = min(n_h * dt, 2 * int_cov / theta)
        theta = max(theta, 2 * int_cov / corr_time)
        n_corr_time = int(np.ceil(corr_time / dt))

        plt.figure(3)
        plt.plot(dt * np.arange(len(h)), cov, 'o', label='C(s)=<g(t+s)g(t)>')
        plt.plot(dt * np.arange(n_corr_time),
                (theta / (n_corr_time - 1)) * np.arange(n_corr_time - 1, -1, -1), '+-', label='Linear approx.')
        plt.legend()
        plt.xlabel('Time lag s (hr)')
        plt.ylabel('C(s) (1/hr^2)')
        plt.show()

    # Some stats
    trim_time = min(dt * n_h, 16)
    n_trim = min(n_h, 1 + int(np.ceil(trim_time / dt)))
    h_trim = h_mean[:n_trim]
    time_trim = dt * np.arange(len(h_trim))
    time_trim[0] = 1e-10
    m_0 = dt * np.sum(h_trim)
    m_1 = (dt / m_0) * np.sum(time_trim * h_trim)
    var_exp = (dt / m_0) * np.sum(((time_trim - m_1) ** 2) * h_trim)
    RMSE = np.sqrt(np.mean((J @ h - y) ** 2))

    # Generate an Inverse Gaussian with same parameters
    time_inv = np.linspace(0, (n_trim - 1) * dt, 1000)
    time_inv[0] = 1e-10
    h_inv = np.interp(time_inv, time_trim, h_trim)
    v = dx / m_1  # m/hr
    Disp = var_exp * v ** 3 / dx / 2  # m^2/hr

    # Best-fit using optimization
    from scipy.optimize import minimize

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

    # Plot final results
    plt.figure(2)
    plt.plot(t_h, np.mean(h_all[:, :ireal], axis=1), 'r', label='Mean')
    plt.xlabel(r'$\tau$ [hr]')
    plt.ylabel('g(τ) [1/hr]')
    plt.plot(t_h, np.percentile(h_all[:, :ireal], 10, axis=1), 'b', label='10%')
    plt.plot(t_h, np.percentile(h_all[:, :ireal], 90, axis=1), 'b', label='90%')
    plt.axis([0, 24, 0, 1])
    plt.text(12, 0.5, f'g(t) stats:\nkernel mass: {m_0:.3f}\nmean (hr): {m_1:.3f}\nVar (hr^2): {var_exp:.3f}')
    plt.legend()
    plt.show()

    plt.figure(22)
    plt.semilogy(t_h, np.mean(h_all[:, :ireal], axis=1), 'r', label='Mean')
    plt.xlabel(r'$\tau$ [hr]')
    plt.ylabel('g(τ) [1/hr]')
    plt.plot(t_h, np.percentile(h_all[:, :ireal], 10, axis=1), 'b', label='10%')
    plt.plot(t_h, np.percentile(h_all[:, :ireal], 90, axis=1), 'b', label='90%')
    plt.axis([0, 24, 1e-5, 1])
    plt.legend()
    plt.show()



if __name__ == '__main__':
    print('Running deconvolution...')
    
    # run Gambill data:
    num_dets = {'theta': 5, 'corr_time': 24, 'sigma': 0.231, 'sigma_max': 0.4, 'n_h': 64, 'nreal': 10}
    prefix = 'medQ_R2'
    time, in_signal, out_signal = load_gambill_data(prefix=prefix)
    deconv(num_dets, time, in_signal, out_signal,fit_ade=True,ex_nm=f'gambill_{prefix}')
    
    # run Gambill with deconv_2:
    num_dets = {'theta': 35, 'corr_time': 0.1, 'sigma': 0.231, 'sigma_max': 0.4, 'n_h': 128, 'nreal': 50}
    prefix = 'highQ_R2'
    time, in_signal, out_signal = load_gambill_data(prefix=prefix)
    assert len(time) == len(in_signal) == len(out_signal), 'Lengths of time, input, and output signals must be equal.'
    deconv_2(num_dets, time, in_signal, out_signal,fit_ade=True,ex_nm=f'gambill_{prefix}')
    
    # run chapeau function:
    time, in_signal,out_signal = chapeau()
    num_dets = {'theta': 2, 'corr_time': 8, 'sigma': 0.1, 'sigma_max': 0.15, 'n_h': None, 'nreal': 10}
    assert len(time) == len(in_signal) == len(out_signal), 'Lengths of time, input, and output signals must be equal.'
    deconv(num_dets, time, in_signal, out_signal,fit_ade=False,ex_nm='chapeau')
    
    # bimodal example:
    time, in_signal,out_signal = bimodal(dt=0.006)
    num_dets = {'theta': 2, 'corr_time': 6, 'sigma': 0.1, 'sigma_max': 0.15, 'n_h': None, 'nreal': 10}
    assert len(time) == len(in_signal) == len(out_signal), 'Lengths of time, input, and output signals must be equal.'
    deconv(num_dets,time, in_signal, out_signal,fit_ade=False,ex_nm='bimodal')
    