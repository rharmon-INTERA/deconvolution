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
import time as time_mod
import warnings
warnings.filterwarnings("ignore")

import kwn_kernel_make as make_kern 
import py_plotting as pyplt


def available_cpu_count():
    cpu_avail = multiprocessing.cpu_count()
    print(f"Available CPUs: {cpu_avail}")
    return cpu_avail


def compute_single_realization_checked(args):
    """
    Returns:
        h (np.ndarray shape (n_h,)): realization
        converged (bool): True if active set stabilized within iter limit
    """
    C, J, iQ, y, sigma, sigma_max, n_h, dt = args

    # Unconditional realization:
    h_uc = C.T @ np.random.randn(n_h)
    # Add measurement error:
    me = sigma * np.random.randn(len(y))

    hL = []   # indices of active nonnegativity constraints
    nL = 0
    converged = False

    for _ in range(50):
        JRJ = J.T @ J / sigma**2
        u = np.ones((n_h, 1))

        # augmented system for unknowns [h, alpha] (alpha scalar)
        umat = np.block([[JRJ + iQ, JRJ @ u],
                         [u.T @ JRJ, u.T @ JRJ @ u]])

        urhs = np.concatenate([
            (J.T @ (y + me) / sigma**2 - JRJ @ h_uc).ravel(),
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

        nu = sol[n_h + 1:]  # Lagrange multipliers for active constraints

        # active-set update (same logic as your code)
        hLold = list(hL)
        hLadd = list(np.where(h < 0)[0])
        hLrem = [hL[j] for j in range(len(nu)) if nu[j] > 0] if nL > 0 else []
        hL = sorted(list((set(hL) - set(hLrem)) | set(hLadd)))
        nL = len(hL)

        if sorted(hLold) == sorted(hL):
            converged = True
            break

    if converged and nL > 0:
        h[hL] = 0.0  # final enforce

    return h, converged

def draw_accepted_realizations(C, J, iQ, y, sigma, sigma_max, n_h, dt,
                               nreal_target, max_attempts_factor=50,
                               parallel=False, cpu_cnt=None):
    """
    Returns:
      h_all_real: (n_h, nreal_target) array of accepted realizations
    """
    accepted = []
    attempts = 0
    max_attempts = max_attempts_factor * nreal_target

    if not parallel:
        while len(accepted) < nreal_target and attempts < max_attempts:
            attempts += 1
            h, ok = compute_single_realization_checked((C, J, iQ, y, sigma, sigma_max, n_h, dt))
            if ok:
                accepted.append(h)
        if len(accepted) < nreal_target:
            raise RuntimeError(
                f"Only accepted {len(accepted)}/{nreal_target} realizations after {attempts} attempts. "
                f"Consider increasing max_attempts_factor or checking conditioning/sigma."
            )
        return np.column_stack(accepted)

    # Parallel version: do in batches, filter accepted, repeat until filled
    from multiprocessing import Pool

    batch = max(nreal_target, (cpu_cnt or 1) * 2)
    with Pool(cpu_cnt) as pool:
        while len(accepted) < nreal_target and attempts < max_attempts:
            # submit a batch
            args_list = [(C, J, iQ, y, sigma, sigma_max, n_h, dt) for _ in range(batch)]
            results = pool.map(compute_single_realization_checked, args_list)
            attempts += batch
            for h, ok in results:
                if ok:
                    accepted.append(h)
                    if len(accepted) >= nreal_target:
                        break

    if len(accepted) < nreal_target:
        raise RuntimeError(
            f"Only accepted {len(accepted)}/{nreal_target} realizations after ~{attempts} attempts. "
            f"Consider increasing max_attempts_factor or checking conditioning/sigma."
        )

    return np.column_stack(accepted)

def deconv_parallel(df, num_dets, prefix='bimodal'):
    # Load signals from dataframe
    t = df['time'].values.copy()            # time vector
    inp = df['input'].values.copy()         # input signal
    out_signal = df['output'].values.copy() # output signal
    
    x = np.asarray(inp, dtype=float).copy()
    y = np.asarray(out_signal, dtype=float).copy()
    t = np.asarray(t, dtype=float).copy()

    # out directories
    k_type = prefix.split('_')[1] if '_' in prefix else prefix
    noise_level = prefix.split('_')[-1] if '_' in prefix else ""
    if k_type in ['bimodal', 'chapeau', 'gamma']:
        noise_type = prefix.split('_')[2]
        kernel = df['kernel'].values.copy()
        stats_out = os.path.join('known_kernels', 'python_make', k_type, 'outputs', noise_type, f'noise_added_{noise_level}')
    else:
        stats_out = os.path.join('field_studies', 'gambill', 'python_make', 'outputs', prefix)

    results_out = stats_out
    figs_out = os.path.join(stats_out, 'figures')
    for d in (stats_out, figs_out):
        os.makedirs(d, exist_ok=True)

    # parallel settings
    if num_dets.get('parallel', False):
        cpu_cnt = (os.cpu_count() or 2) - 1
        cpu_cnt = max(cpu_cnt, 1)
        print(f"Using {cpu_cnt} CPUs.")
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
    theta_converg = float(num_dets.get('theta_converg', 0.0))

    np.random.seed()

    x = inp.copy()
    y = out_signal.copy()

    # Add noise to y (known kernels only)
    if k_type in ['bimodal', 'chapeau', 'gamma']:
        if noise_type == 'on-out':
            desired_std = num_dets['sigma']
            std_uniform_minus_05 = 1 / np.sqrt(12)
            noise_multiplier = desired_std / std_uniform_minus_05
            y = y + noise_multiplier * (np.random.rand(*y.shape) - 0.5)
            act_std = np.std(noise_multiplier * (np.random.rand(10000) - 0.5))
            print("Actual noise std:", act_std)
        elif noise_type == 'on-in-after-conv':
            desired_std = num_dets['sigma']
            std_uniform_minus_05 = 1 / np.sqrt(12)
            noise_multiplier = desired_std / std_uniform_minus_05
            x = x + noise_multiplier * (np.random.rand(*x.shape) - 0.5)
            act_std = np.std(noise_multiplier * (np.random.rand(10000) - 0.5))
            print("Actual noise std on input after convolution:", act_std)
        else:
            act_std = num_dets['sigma']
            print("No noise added to output signal (field), noise was added to input before convolution.")
    else:
        act_std = 0.0
        print("No noise added to output signal (field).")

    # Time increment and covariance setup
    dt = t[1] - t[0]
    corr_time = min(n_h * dt, corr_time)
    n_corr_time = int(np.ceil(corr_time / dt))

    # ---- start wall-clock timing ----
    t_start = time_mod.perf_counter()

    # Initial triangular covariance (first guess)
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
    theta_old = 0.0

    while (rel_cov_change > 0.1) and (abs(theta_old - theta) / max(theta, 1e-16) >= theta_converg):
        outer_iter += 1
        print(f"Outer iteration {outer_iter}: rel_cov_change = {rel_cov_change:.3g}")
        print(f"theta change: {abs(theta_old - theta) / max(theta, 1e-16)}")

        # Rebuild J each iter
        input_fn = dt * x.copy()
        input_fn[input_fn < 1e-8] = 0.0
        r_vec = dt * np.zeros(n_h)
        J = toeplitz(input_fn, r_vec)

        # Olaf's linear with current theta, then overwrite with current cov
        if method == 'cirpka':
            c = dt * theta * np.arange(n_h, 0, -1)
        else:
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

            urhs = np.concatenate([
                (J.T @ y / sigma**2).ravel(),
                (u.T @ (J.T @ y / sigma**2)).ravel()
            ])

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

            # Update sigma from best-estimate residuals
            sim = J @ h_be
            denom = max(1, len(y) - n_h + nL - 1)
            sigma = float(np.sqrt(np.dot(y - sim, y - sim) / denom))
            sigma = min(sigma, sigma_max)
            
            if iter_inner == 1 or iter_inner % 2 == 0:
                print(f"  inner {iter_inner:02d}: sigma={sigma:.4g}, nL={nL}")

            # Active-set update
            hLold = list(hL)
            hLadd = list(np.where(h_be < 0)[0])
            hLrem = [hL[j] for j in range(len(nu)) if nu[j] > 0] if nL > 0 else []
            hL = sorted(list((set(hL) - set(hLrem)) | set(hLadd)))
            nL = len(hL)

            if sorted(hLold) == sorted(hL):
                break
        # ----- End best-estimate -----


        h_all_real = draw_accepted_realizations(
            C=C, J=J, iQ=iQ, y=y, sigma=sigma, sigma_max=sigma_max,
            n_h=n_h, dt=dt,
            nreal_target=nreal,
            max_attempts_factor=50,
            parallel=bool(num_dets.get('parallel', False)),
            cpu_cnt=cpu_cnt
        )
        nreal_eff = nreal  # accepted realizations count
        h_all = np.column_stack([h_be, h_all_real])  # include best-estimate as first column


        # ----- Covariance Update -----
        old_cov = cov.copy()

        def sumprob(lntheta):
            theta_test = np.exp(lntheta)
            lnpsum = 0.0
            ng = h_all.shape[0]
            # match your original Python behavior (uses first nreal columns of h_all)
            for i in range(nreal_eff):
                h_i = h_all[:, i]
                num_nonzero = np.sum(h_i > 0)
                lnp_i = -num_nonzero/2.0 * np.log(4.0*np.pi*theta_test) - (ng - 1)/2.0 * np.log(1.0)
                dif = np.diff(h_i)
                lnp_i -= (dif @ dif) / (4.0*theta_test)
                lnpsum -= lnp_i
            return float(np.real(lnpsum))

        x0 = np.log(max(theta, 1e-12))
        maxfun = 200 * np.atleast_1d(x0).size
        res = fmin(lambda z: sumprob(z), x0, xtol=1e-4, ftol=1e-4,
                   maxiter=maxfun, maxfun=maxfun, disp=False)
        theta_old = theta
        theta = float(np.exp(res[0]))

        # triangular cov from theta
        cov = np.zeros(n_h)
        if n_h > 1:
            cov[:n_h] = (theta / (n_h - 1)) * np.arange(n_h - 1, -1, -1)

        if method != 'cirpka':
            # empirical cov from mean h via FFT
            h_mean_ext = np.concatenate([np.mean(h_all, axis=1), np.zeros(n_h)])
            h_fft = np.fft.fft(h_mean_ext)
            cov_fft = (2.0 / n_h) * np.fft.ifft(h_fft * np.conj(h_fft)).real
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
                cov = np.zeros(n_h)
                if n_corr_time > 1:
                    cov[:n_corr_time] = (theta / (n_corr_time - 1)) * np.arange(n_corr_time - 1, -1, -1)

        # Relative covariance change (unchanged from your Python)
        int_cov = dt * np.sum(cov) if cov.size else 1.0
        length_comp = min(len(cov), len(old_cov))
        # --- Relative covariance change (MATCH MATLAB) ---
        length_comp = min(len(cov), len(old_cov))
        numer = np.sqrt(dt * np.sum((cov[:length_comp] - old_cov[:length_comp])**2))

        cov0 = float(cov[0]) if len(cov) else 1.0
        cov0 = max(cov0, 1e-16)

        snr = sigma / max(float(np.max(y)), 1e-16)
        snr = max(snr, 1e-16)

        rel_cov_change = float(numer / cov0 / snr)
        print("Relative covariance change:", rel_cov_change)

        # Finalize ensemble stats for this outer iter
        h_all = np.real(h_all)
        h_mean = np.mean(h_all, axis=1)
        if k_type in ['bimodal', 'chapeau', 'gamma']:
            L_2 = np.sqrt(dt * np.sum((h_mean - kernel[:len(h_mean)]) ** 2))

    # ===== Final stats & save (same as your code below) =====
    t_end = time_mod.perf_counter()
    solve_time_sec = t_end - t_start
    solve_time_min = solve_time_sec / 60.0

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
    if k_type in ['bimodal', 'chapeau', 'gamma']:
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

    # Build results table (same schema as your original)
    if k_type in ['bimodal', 'chapeau', 'gamma']:
        results_table = pd.DataFrame({
            'Method': [method.capitalize()],
            'AddedNoise': [act_std],
            'Mass_m0': [m_0],
            'MeanTravelTime_m1': [m_1],
            'SpreadOfTravelTimes_m2': [var_exp],
            'RMSE': [RMSE],
            'CorrelationTime': [corr_time],
            'FinalSigma': [sigma],
            'L2': [L_2],
            'SolveTime_sec': [solve_time_sec],
            'SolveTime_min': [solve_time_min],
        })
    else:
        results_table = pd.DataFrame({
            'Method': [method.capitalize()],
            'Mass_m0': [m_0],
            'MeanTravelTime_m1': [m_1],
            'SpreadOfTravelTimes_m2': [var_exp],
            'RMSE': [RMSE],
            'CorrelationTime': [corr_time],
            'FinalSigma': [sigma],
            'SolveTime_sec': [solve_time_sec],
            'SolveTime_min': [solve_time_min],
        })

    print(f"Solve time ({method}): {solve_time_sec:.2f} s ({solve_time_min:.2f} min)")

    csv_filename = prefix + '_' + method + '_stats.csv'
    results_table.to_csv(os.path.join(stats_out, csv_filename), index=False)

    # Minimal return consistent with your original
    df_subset = df.iloc[:min(300, len(df))].copy()
    df_subset['time'] = t_h[:len(df_subset)]
    df_subset['input'] = x[:len(df_subset)]
    df_subset['output'] = y[:len(df_subset)]
    if k_type in ['bimodal', 'chapeau', 'gamma']:
        df_subset['kernel'] = kernel[:len(df_subset)]
    df_subset['transfer_func_mean'] = h_mean[:len(df_subset)]
    df_subset['transfer_func_p10'] = h_p10[:len(df_subset)]
    df_subset['transfer_func_p90'] = h_p90[:len(df_subset)]
    df_subset.to_csv(os.path.join(results_out, prefix + '_data_and_results.csv'), index=False)
    
    # save best estimate simulation:
    y_pred = (J @ h_mean).ravel()   # ensure 1-D array
    bsim = pd.DataFrame({
        'time': df['time'].values.copy(),
        'input': x.ravel(),
        'output': y.ravel(),
        'simulated': y_pred
    })
    out_file = os.path.join(results_out, prefix + '_sim.csv')
    bsim.to_csv(out_file, index=False)


    return df_subset, results_table


if __name__ == '__main__':
    
    # run control vars:
    run_knwn_kernels = True
    noise_type = 'on-out' # must be 'on-out','on-in-before-conv', or 'on-in-after-conv', needed for knwn kernels
    
    run_gambill = False
    
    plot_figs = True
    
    run_in_parallel = False
    num_workers = 8  # only used if run_in_parallel is True
    
    if run_knwn_kernels:
        kernels = ['chapeau', 'gamma', 'bimodal']
        for kernel_shape in kernels:
            curdir = os.getcwd()
            os.chdir(os.path.join('known_kernels','python_make'))
            if kernel_shape == 'chapeau':
                df = make_kern.make_chapeau()
            elif kernel_shape == 'gamma':
                df = make_kern.make_gamma()
            elif kernel_shape == 'bimodal':
                df = make_kern.make_bimodal()
            os.chdir(curdir)
            time, in_signal,out_signal = df['time'].values, df['input'].values, df['output'].values
            
            noise_levels = [0.005, 0.03,0.09] # this is noise added onto output signal
            results_dict = {}
            methods = ['learn','cirpka']
            for method in methods:
                run_key = f'{kernel_shape}_{method}'
                for nl in noise_levels:
                    # if adding noise before convolution, need to remake df each time
                    os.chdir(os.path.join('known_kernels','python_make'))
                    if noise_type == 'on-in-before-conv':
                        if kernel_shape == 'chapeau':
                            df = make_kern.make_chapeau(add_input_noise=nl)
                        elif kernel_shape == 'gamma':
                            df = make_kern.make_gamma(add_input_noise=nl)
                        elif kernel_shape == 'bimodal':
                            df = make_kern.make_bimodal(add_input_noise=nl)
                    os.chdir(curdir)

                    sigma_max = 0.12
                    run_key += f'_{nl}'
                    if nl > sigma_max:
                        sigma_max = nl + 0.03
                    num_dets = {'theta': 0.02, 'corr_time': 40, 
                                'sigma': nl, 'sigma_max': sigma_max, 
                                'n_h': 300,'nreal': 56, 'parallel': run_in_parallel,
                                'method':method, 'theta_converg':0.0}
                    nl = f'{nl:.3f}'
                    results, stats = deconv_parallel(df, num_dets, prefix=f'{method}_{kernel_shape}_{noise_type}_{nl}')
                    results_dict[run_key] = {
                        'results': results,
                        'stats': stats
                    }
                    print(f"Results for {run_key} saved.")
    if plot_figs:
        pyplt.plot_deconv_results(os.path.join('known_kernels','python_make'), noise_type=noise_type,inset_on=False)


    
            
    if run_gambill:
        ws = os.path.join('field_studies', 'gambill', 'data')
        list_csv_files = [f for f in os.listdir(ws) if f.endswith('.csv')]
        # remove '.csv' from names
        sites = [f[:-4] for f in list_csv_files]
        # drop sites ending '_MIM'
        sites = [s for s in sites if not s.endswith('_MIM')]
        
        method = 'learn'  # 'learn' or 'cirpka'
        
        nl = 1
        sigma_max = 5.0
        
        # set theta converg to 0.0 for runs that have no problem converging under
        # the relative covariance change criterion.
        # sites_dict = {
        #     'highQ_R1': {'theta': 1e-4, 'corr_time': 10,
        #                 'sigma': 0.231, 'sigma_max': 1.0,
        #                 'n_h': 1600, 'nreal': 64, 'parallel': run_in_parallel,
        #                 'method': method,'theta_converg':0.0},

        #     'highQ_R2': {'theta': 0.02, 'corr_time': 40,
        #                 'sigma': nl, 'sigma_max': sigma_max,
        #                 'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
        #                 'method': method,'theta_converg':0.0},

        #     'highQ_long': {'theta': 0.02, 'corr_time': 40,
        #                 'sigma': nl, 'sigma_max': sigma_max,
        #                 'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
        #                 'method': method,'theta_converg':0.0},

        #     'lowQ_R1': {'theta': 0.02, 'corr_time': 40,
        #                 'sigma': nl, 'sigma_max': sigma_max,
        #                 'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
        #                 'method': method,'theta_converg':0.0},

        #     'lowQ_R2': {'theta': 12, 'corr_time': 10,
        #                 'sigma': 0.1, 'sigma_max': 0.52,
        #                 'n_h': 68, 'nreal': 16, 'parallel': run_in_parallel,
        #                 'method': method, 'theta_converg':0.0},

        #     'lowQ_long': {'theta': 0.02, 'corr_time': 40,
        #                 'sigma': nl, 'sigma_max': sigma_max,
        #                 'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
        #                 'method': method, 'theta_converg':0.0},

        #     'medQ_R1': {'theta': 0.02, 'corr_time': 40,
        #                 'sigma': nl, 'sigma_max': sigma_max,
        #                 'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
        #                 'method': method, 'theta_converg':0.0},

        #     'medQ_R2': {'theta': 1e-4, 'corr_time': 10,
        #                 'sigma': 0.231, 'sigma_max': 1.0,
        #                 'n_h': 300, 'nreal': 16, 'parallel': run_in_parallel,
        #                 'method': method, 'theta_converg':0.0},

        #     'medQ_long': {'theta': 0.02, 'corr_time': 40,
        #                 'sigma': nl, 'sigma_max': sigma_max,
        #                 'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
        #                 'method': method, 'theta_converg':0.0}
        #     }
        sites_dict = {
            'lowQ_R1': {'theta': 1e-4, 'corr_time': 10,
                        'sigma': 0.231, 'sigma_max': 1.,
                        'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
                        'method': method,'theta_converg':0.0001},
            'medQ_R1': {'theta': 1e-4, 'corr_time': 10,
                        'sigma': 0.231, 'sigma_max': 1.0,
                        'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
                        'method': method, 'theta_converg':0.0},
            'highQ_R1': {'theta': 1e1, 'corr_time': 8,
                        'sigma': 0.0001, 'sigma_max': 5,
                        'n_h': 400, 'nreal': 60, 'parallel': run_in_parallel,
                        'method': method,'theta_converg':0.0},

            }        
        
        for site in sites:
            if site in sites_dict:
                file_path = os.path.join(ws, site + '.csv')
                df = pd.read_csv(file_path)
                df = df.rename(columns={'in': 'input', 'out': 'output'})
                num_dets = sites_dict[site]
                results, stats = deconv_parallel(df, num_dets, prefix=f'{site}_{method}')

                print(f"Deconvolution for site {site} completed.")
                
    # 11/10 notes all examples run, but cirpka looks like it has an issue probs zero start. Next steps,
    # run gambill data and use parallel processing to confirm it works. Then need to get plottting and latex
    # tables upadted and in the draft. Then clean repo and add readme.

    # 12/1-8 notes: 
    #   - re-test bef_conv noise addition (done)
    #   - add tim it and save n outer iters to stats
    #   - finalize latex tabl gen 
    #   - mention to Dave the converg issue with no noise case, seems that atleast 0.005 noise is needed for stable runs across all cases
    #   - comapre Cirpka and learn on Gambill data, quantify differences from obs out, comp time, and # outter iters 
    #   - get the final set of matlab run scripts organized and into git repo

