import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join('known_kernels','python_make')))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import pandas as pd
import json
import scipy.io as sio
import scipy.linalg as la
from scipy.linalg import toeplitz
from scipy.optimize import fmin
from multiprocessing import Pool
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

import kwn_kernel_make as make_kern 
import py_plotting as pyplt


def available_cpu_count():
    cpu_avail = multiprocessing.cpu_count()
    print(f"Available CPUs: {cpu_avail}")
    return cpu_avail


def compute_single_realization(args):
    # Unpack arguments
    C, J, iQ, y, sigma, sigma_max, n_h, dt = args
    # Unconditional realization:
    h_uc = C.T @ np.random.randn(n_h)
    # Add measurement error:
    me = sigma * np.random.randn(len(y))

    hL = []   # indices of active nonnegativity constraints
    nL = 0
    iter_r = 0
    while iter_r < 50:
        iter_r += 1

        JRJ = J.T @ J / sigma**2
        u = np.ones((n_h, 1))

        # augmented system for unknowns [h, alpha] (alpha scalar)
        umat = np.block([[JRJ + iQ, JRJ @ u],
                         [u.T @ JRJ, u.T @ JRJ @ u]])

        urhs = np.concatenate([
            J.T @ (y + me) / sigma**2 - JRJ @ h_uc,
            (u.T @ (J.T @ (y + me) / sigma**2 - JRJ @ h_uc)).ravel()
        ])

        if nL > 0:
            Lmat = np.zeros((n_h + 1, nL))
            Lrhs = np.zeros(nL)
            for j in range(nL):
                Lmat[hL[j], j] = 1.0
                Lmat[n_h, j] = 1.0
                Lrhs[j] = -h_uc[hL[j]]
            mat = np.block([[umat, Lmat],
                            [Lmat.T, np.zeros((nL, nL))]])
            rhs = np.concatenate([urhs, Lrhs])
        else:
            mat = umat
            rhs = urhs
        a = np.diag(mat).copy()
        if len(a) > n_h + 1:
            a[n_h + 1:] = 1.0
        imat = la.inv(np.diag(1 / a) @ mat) @ np.diag(1 / a)
        sol = imat @ rhs
        h = sol[:n_h] + sol[n_h] + h_uc
        if nL > 0:
            h[hL] = 0.0
        nu = sol[n_h + 1:]

        # active-set update
        hLold = list(hL)
        hLadd = list(np.where(h < 0)[0])
        hLrem = [hL[j] for j in range(len(nu)) if nu[j] > 0] if nL > 0 else []
        hL = sorted(list((set(hL) - set(hLrem)) | set(hLadd)))
        nL = len(hL)
        if sorted(hLold) == sorted(hL):
            break

    return h


def deconv_parallel(df, num_dets, prefix='bimodal'):
    # Load signals from dataframe
    t = df['time'].values.copy()          # time vector
    inp = df['input'].values.copy()         # input signal
    kernel = df['kernel'].values.copy()     # transfer function (kernel)
    out_signal = df['output'].values.copy() # output signal

    # out directories
    k_type = prefix.split('_')[1]
    noise_level = prefix.split('_')[-1]
    stats_out = os.path.join('known_kernels','python_make',k_type,'outputs',f'noise_added_{noise_level}')
    results_out = stats_out
    figs_out = os.path.join(stats_out,'figures')
    for d in (stats_out, figs_out):
        os.makedirs(d, exist_ok=True)

    # parallel settings
    if num_dets['parallel']:
        cpu_cnt = available_cpu_count() - 1
        print(f"Using all available {cpu_cnt} CPUs.")
    else:
        cpu_cnt = None

    # User-defined numerical details
    theta = float(num_dets['theta'])
    corr_time = float(num_dets['corr_time'])
    sigma = float(num_dets['sigma'])
    sigma_max = float(num_dets['sigma_max'])
    n_h = int(num_dets['n_h'])
    nreal = int(num_dets['nreal'])
    method = str(num_dets['method']).lower()

    np.random.seed()

    x = inp.copy()
    y = out_signal.copy()

    # Add noise to y
    if num_dets['sigma'] != 0:
        desired_std = num_dets['sigma']
        std_uniform_minus_05 = 1 / np.sqrt(12)
        noise_multiplier = desired_std / std_uniform_minus_05
        y = y + noise_multiplier * (np.random.rand(*y.shape) - 0.5)
        act_std = np.std(noise_multiplier * (np.random.rand(10000) - 0.5))        
        print("Actual noise std:", act_std)
    else:
        act_std = 0.0
        print("No noise added to output signal.")

    # Time increment and covariance setup
    dt = t[1] - t[0]
    corr_time = min(n_h * dt, corr_time)
    n_corr_time = int(np.ceil(corr_time / dt))


    # Initial triangled covariance (first guess)
    cov = np.zeros(n_h)
    if n_corr_time > 0:
        cov[:n_corr_time] = (theta / n_corr_time) * np.arange(n_corr_time, 0, -1)

    # Construct Jacobian (convolution matrix)
    input_fn = dt * x.copy()
    input_fn[input_fn < 1e-8] = 0.0
    r_vec = dt * np.zeros(n_h)
    J = toeplitz(input_fn, r_vec)

    rel_cov_change = 999.0
    outer_iter = 0

    while rel_cov_change > 0.1:
        outer_iter += 1
        print(f"Outer iteration {outer_iter}: rel_cov_change = {rel_cov_change:.3g}")

        # Rebuild J each iter
        input_fn = dt * x.copy()
        input_fn[input_fn < 1e-8] = 0.0
        r_vec = dt * np.zeros(n_h)
        J = toeplitz(input_fn, r_vec)

        # Olaf's linear with current theta, then overwrite with current cov
        c = dt * theta * np.arange(n_h, 0, -1)
        c[:len(cov)] = cov

        Q = toeplitz(c)
        epsilon = 1e-8
        Q += epsilon * np.eye(n_h)
        C = la.cholesky(Q, lower=False)
        iQ = la.inv(Q)

        # ----- Best-estimate (me=0, h_uc=0) with nonnegativity -----
        hL = []
        nL = 0
        iter_inner = 0
        while iter_inner < 50:
            iter_inner += 1
            JRJ = J.T @ J / sigma**2
            u = np.ones((n_h, 1))
            umat = np.block([[JRJ + iQ, JRJ @ u],
                             [u.T @ JRJ, u.T @ JRJ @ u]])
            urhs = np.concatenate([J.T @ y / sigma**2,
                                   (u.T @ J.T @ y / sigma**2).ravel()])
            if nL > 0:
                Lmat = np.zeros((n_h + 1, nL))
                Lrhs = np.zeros(nL)
                for j in range(nL):
                    Lmat[hL[j], j] = 1.0
                    Lmat[n_h, j] = 1.0
                    Lrhs[j] = 0.0
                mat = np.block([[umat, Lmat],
                                [Lmat.T, np.zeros((nL, nL))]])
                rhs = np.concatenate([urhs, Lrhs])
            else:
                mat = umat
                rhs = urhs
            a = np.diag(mat).copy()
            if len(a) > n_h + 1:
                a[n_h + 1:] = 1.0
            imat = la.inv(np.diag(1 / a) @ mat) @ np.diag(1 / a)
            sol = imat @ rhs

            h_be = sol[:n_h] + sol[n_h]
            if nL > 0:
                h_be[hL] = 0.0
            nu = sol[n_h + 1:n_h + 1 + nL]

            # Update sigma from best-estimate residuals (MATLAB)
            sim = J @ h_be
            denom = max(1, len(y) - n_h + nL - 1)
            sigma = float(np.sqrt(np.dot(y - sim, y - sim) / denom))
            sigma = min(sigma, sigma_max)

            # Active-set update
            hLold = list(hL)
            hLadd = list(np.where(h_be < 0)[0])
            hLrem = [hL[j] for j in range(len(nu)) if nu[j] > 0] if nL > 0 else []
            hL = sorted(list((set(hL) - set(hLrem)) | set(hLadd)))
            nL = len(hL)
            if sorted(hLold) == sorted(hL):
                break
        # ----- End best-estimate -----

        # ----- Parallel ensemble realizations -----
        args_list = [(C, J, iQ, y, sigma, sigma_max, n_h, dt) for _ in range(nreal)]
        if num_dets['parallel']:
            with Pool(cpu_cnt) as pool:
                h_all_list = pool.map(compute_single_realization, args_list)
            h_all = np.column_stack(h_all_list)
        else:
            h_all = np.column_stack([compute_single_realization(a) for a in args_list])

        # ----- Covariance Update -----
        old_cov = cov.copy()

        # --- theta via fminsearch analog (on ln(theta)) ---
        def sumprob(lntheta):
            theta_test = np.exp(lntheta)
            lnpsum = 0.0
            for i in range(nreal):
                h_i = h_all[:, i]
                nnz = np.sum(h_i > 0)
                ng = h_i.size
                lnp_i = -nnz / 2.0 * np.log(4.0 * np.pi * theta_test) - (ng - 1) / 2.0 * np.log(1.0)
                dif = np.diff(h_i)
                lnp_i -= (dif @ dif) / (4.0 * theta_test)
                lnpsum -= lnp_i
            return float(np.real(lnpsum))

        x0 = np.log(max(theta, 1e-12))
        maxfun = 200 * np.atleast_1d(x0).size
        res = fmin(lambda z: sumprob(z), x0, xtol=1e-4, ftol=1e-4,
                   maxiter=maxfun, maxfun=maxfun, disp=False)
        theta = float(np.exp(res[0]))

        # triangular cov from theta
        cov = np.zeros(n_h)
        if n_h > 1:
            cov[:n_h] = (theta / (n_h - 1)) * np.arange(n_h - 1, -1, -1)

        if method != 'cirpka':
            # empirical cov from mean h via FFT 
            h_mean_ext = np.concatenate([np.mean(h_all, axis=1), np.zeros(n_h)])
            h_fft = np.fft.fft(h_mean_ext)
            cov_fft = (1.0 / n_h) * np.fft.ifft(h_fft * np.conj(h_fft)).real
            cov_emp = cov_fft[:n_h].copy()

            cov_emp = np.maximum(0.0, cov_emp)
            int_cov = dt * np.sum(cov_emp)
            if int_cov > 0:
                cov_emp *= (int_cov / (dt * np.sum(cov_emp)))  # renorm (no-op but safe)

            cov = cov_emp
            theta = cov[0]
            var_exp = np.var(np.mean(h_all, axis=1))

            corr_time = min(n_h * dt, 2.0 * int_cov / max(theta, 1e-16))
            theta = max(theta, 2.0 * int_cov / max(corr_time, dt))
            n_corr_time = int(np.ceil(corr_time / dt))

            if method == 'linear':
                # Linear approximation: triangular over first n_corr_time, no resizing n_h
                cov = np.zeros(n_h)
                if n_corr_time > 1:
                    cov[:n_corr_time] = (theta / (n_corr_time - 1)) * np.arange(n_corr_time - 1, -1, -1)

        # Relative covariance change 
        int_cov = dt * np.sum(cov) if cov.size else 1.0
        length_comp = min(len(cov), len(old_cov))
        rel_cov_change = float(np.sqrt(dt * np.sum((cov[:length_comp] - old_cov[:length_comp]) ** 2)) /
                               max(int_cov, 1e-16) /
                               max(sigma / max(np.max(y), 1e-16), 1e-16))
        print("Relative covariance change:", rel_cov_change)

        # Finalize ensemble stats for this outer iter
        h_all = np.real(h_all)
        h_mean = np.mean(h_all, axis=1)
        L_2 = np.sqrt(dt * np.sum((h_mean - kernel[:len(h_mean)]) ** 2))

    # ===== Final stats & save (unchanged from your code) =====
    trim_time = min(dt * n_h, 30)
    n_trim = min(n_h, 1 + int(np.ceil(trim_time / dt)))
    h_trim = h_mean[:n_trim]
    time_trim = dt * np.arange(len(h_trim))
    time_trim[0] = 1e-10
    m_0 = dt * np.sum(h_trim)
    m_1 = (dt / m_0) * np.sum(time_trim * h_trim)
    var_exp = (dt / m_0) * np.sum(((time_trim - m_1) ** 2) * h_trim)
    RMSE = np.sqrt(np.mean((J @ h_mean - y) ** 2))
    print("m_0:", m_0, "m_1:", m_1, "var_exp:", var_exp, "RMSE:", RMSE)

    t_h = dt * np.arange(n_h)
    h_p10 = np.percentile(h_all, 10, axis=1)
    h_p90 = np.percentile(h_all, 90, axis=1)

    plt.figure(figsize=(6, 4))
    skyBlue = [0.35, 0.70, 0.90]
    orange = [0.90, 0.60, 0.00]
    bluishGreen = [0.00, 0.62, 0.45]
    plt.plot(t_h, kernel[:n_h], '-k', linewidth=1.5, label='Known transfer fx')
    plt.plot(t_h, h_mean, linestyle='--', color=bluishGreen, linewidth=1.5, label='Mean h(t)')
    p10_line, = plt.plot(t_h, h_p10, '-', color=orange, linewidth=1, label='10th pct.')
    p90_line, = plt.plot(t_h, h_p90, '-', color=orange, linewidth=1, label='90th pct.')
    p10_line.set_color((*orange, 0.5))
    p90_line.set_color((*orange, 0.5))
    plt.xlabel('τ [hr]', fontsize=14)
    plt.ylabel('g(τ) [1/hr]', fontsize=14)
    plt.xlim([0, num_dets['corr_time']])
    plt.legend(loc='best')

    rounded_std = round(act_std, 5)
    plt.title(method.capitalize() + f' derived transfer function\n Noise added {rounded_std}', fontsize=14)
    plt.savefig(os.path.join(figs_out,'g_of_t_' + method + f'_{rounded_std}.png'), dpi=300)
    plt.close()

    results_table = pd.DataFrame({
        'Method': [method.capitalize()],
        'AddedNoise': [act_std],
        'Mass_m0': [m_0],
        'MeanTravelTime_m1': [m_1],
        'SpreadOfTravelTimes_m2': [var_exp],
        'RMSE': [RMSE],
        'CorrelationTime': [corr_time],
        'FinalSigma': [sigma],
        'L2': [L_2]
    })
    csv_filename = prefix + '_' + method + '_stats.csv'
    results_table.to_csv(os.path.join(stats_out, csv_filename), index=False)

    df_subset = df.iloc[:300].copy()
    df_subset['time'] = t[:300]
    df_subset['input'] = x[:300]
    df_subset['output'] = y[:300]
    df_subset['kernel'] = kernel[:300]
    df_subset['transfer_func_mean'] = h_mean
    df_subset['transfer_func_p10'] = h_p10
    df_subset['transfer_func_p90'] = h_p90
    df_subset.to_csv(os.path.join(results_out, prefix + '_data_and_results.csv'), index=False)

    return df_subset, results_table


if __name__ == '__main__':
    
    # run control vars:
    run_knwn_kernels = True
    noise_type = 'on_out'  # must be 'on_out','on_in_before_conv', or 'on_in_after_conv', needed for knwn kernels
    
    run_gambill = False
    
    plot_figs = True
    
    run_in_parallel = False
    num_workers = 8  # only used if run_in_parallel is True
    
    if run_knwn_kernels:
        kernels = ['chapeau', 'gamma', 'bimodal']
        for kernel_shape in kernels:
            if kernel_shape == 'chapeau':
                # add_input_noise, enter noise if you want added before convolution with kernel
                df = make_kern.make_chapeau(add_input_noise=0.0)
            elif kernel_shape == 'gamma':
                df = make_kern.make_gamma(add_input_noise=0.0)
            elif kernel_shape == 'bimodal':
                df = make_kern.make_bimodal(add_input_noise=0.0)

            time, in_signal,out_signal = df['time'].values, df['input'].values, df['output'].values
            
            noise_levels = [0.001, 0.03,0.09] # this is noise added onto output signal
            results_dict = {}
            methods = ['learn','cirpka']
            for method in methods:
                run_key = f'{kernel_shape}_{method}'
                for nl in noise_levels:
                    sigma_max = 0.09
                    run_key += f'_{nl}'
                    if nl > sigma_max:
                        sigma_max = nl + 0.03
                    num_dets = {'theta': 0.02, 'corr_time': 40, 
                                'sigma': nl, 'sigma_max': sigma_max, 
                                'n_h': 300,'nreal': 56, 'parallel': run_in_parallel,
                                'method':method}
                    nl = f'{nl:.3f}'
                    results, stats = deconv_parallel(df, num_dets, prefix=f'{method}_{kernel_shape}_{nl}')
                    results_dict[run_key] = {
                        'results': results,
                        'stats': stats
                    }
                    print(f"Results for {run_key} saved.")
            if plot_figs:
                pyplt.plot_deconv_results(os.path.join('known_kernels','python_make'), noise_type=noise_type,inset_on=False)
    
    # 11/10 notes all examples run, but cirpka looks like it has an issue probs zero start. Next steps,
    # run gambill data and use parallel processing to confirm it works. Then need to get plottting and latex
    # tables upadted and in the draft. Then clean repo and add readme.
    
    # save to json:
    with open(f'{kernel_shape}_deconv_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    print("Results saved to deconv_results.json")
    print("Deconvolution completed.")


