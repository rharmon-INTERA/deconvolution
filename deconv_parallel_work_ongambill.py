import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('known_kernels', 'python_make')))
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

import logging
import logging.handlers
from datetime import datetime

import kwn_kernel_make as make_kern
import py_plotting as pyplt



_worker_logger = None  # set in _worker_init on each worker process


def _start_logger(log_path):
    """
    Configure a single file logger on the main process.
    Workers will send records into a Queue handled by a QueueListener here.
    Returns: (logger, listener, queue)
    """
    logger = logging.getLogger("deconv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Set up a file handler (the listener will write to file)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(processName)s[%(process)d] | job=%(job_id)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(fmt)

    # Create the shared queue and attach a QueueHandler to the logger
    q = multiprocessing.Queue()
    qh = logging.handlers.QueueHandler(q)
    logger.addHandler(qh)

    # Listener consumes records from queue and writes via file_handler
    listener = logging.handlers.QueueListener(q, file_handler, respect_handler_level=True)
    listener.start()
    return logger, listener, q


def _worker_init(log_queue):
    """
    Run in each worker process to attach a QueueHandler that forwards logs
    to the main process's listener.
    """
    global _worker_logger
    _worker_logger = logging.getLogger("deconv")
    _worker_logger.setLevel(logging.INFO)
    _worker_logger.handlers.clear()
    _worker_logger.addHandler(logging.handlers.QueueHandler(log_queue))


def compute_single_realization_logged(job_id, args):
    """
    Wrapper around compute_single_realization that logs start/finish
    and any exceptions. Requires _worker_init() to have run in the worker.
    """
    extra = {"job_id": job_id}
    try:
        _worker_logger.info("received", extra=extra)
    except Exception:
        pass  # in case someone calls without initializer

    try:
        h = compute_single_realization(args)
        try:
            _worker_logger.info("finished", extra=extra)
        except Exception:
            pass
        return h
    except Exception as e:
        try:
            _worker_logger.exception(f"failed: {e}", extra=extra)
        except Exception:
            pass
        raise


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
    t = df['time'].values.copy()          # time vector
    inp = df['input'].values.copy()       # input signal
    out_signal = df['output'].values.copy()  # output signal

    # out directories
    k_type = prefix.split('_')[1]
    noise_level = prefix.split('_')[-1]
    if k_type in ['bimodal', 'chapeau', 'gamma']:
        kernel = df['kernel'].values.copy()     # transfer function (kernel)
        stats_out = os.path.join('known_kernels', 'python_make', k_type, 'outputs', f'noise_added_{noise_level}')
    else:
        stats_out = os.path.join('field_studies', 'gambill', 'python_make', 'outputs', prefix)

    results_out = stats_out
    figs_out = os.path.join(stats_out, 'figures')
    for d in (stats_out, figs_out):
        os.makedirs(d, exist_ok=True)

    # --- start logging (single file per run directory) ---
    log_path = os.path.join(stats_out, "parallel_runs.log")
    logger, listener, log_queue = _start_logger(log_path)
    logger.info(f"start run | prefix={prefix}", extra={"job_id": "-"})

    # pre-create pool reference (for cleanup)
    pool = None

    try:
        # parallel settings
        if num_dets['parallel']:
            max_cpus = available_cpu_count()
            if 'nworkers' in num_dets:
                req = int(num_dets['nworkers'])
                cpu_cnt = max(1, min(req, max_cpus))
                print(f"Using {cpu_cnt} CPUs as specified.")
            else:
                cpu_cnt = max(1, max_cpus - 1)
                print(f"Using {cpu_cnt} CPUs as available_cpu_count()-1.")
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
        if k_type in ['bimodal', 'chapeau', 'gamma']:
            if num_dets['sigma'] != 0:
                desired_std = num_dets['sigma']
                std_uniform_minus_05 = 1 / np.sqrt(12)
                noise_multiplier = desired_std / std_uniform_minus_05
                y = y + noise_multiplier * (np.random.rand(*y.shape) - 0.5)
                act_std = np.std(noise_multiplier * (np.random.rand(10000) - 0.5))
                print("Actual noise std:", act_std)
                logger.info(f"noise added | std={act_std:.6g}", extra={"job_id": "-"})
            else:
                act_std = 0.0
                print("No noise added to output signal.")
                logger.info("noise added | std=0", extra={"job_id": "-"})
        else:
            act_std = 0.0
            print("No noise added to output signal (field).")

        # Time increment and covariance setup
        dt = t[1] - t[0]
        corr_time = min(n_h * dt, corr_time)
        n_corr_time = int(np.ceil(corr_time / dt))

        # init triangled covariance (first guess)
        cov = np.zeros(n_h)
        if n_corr_time > 0:
            cov[:n_corr_time] = (theta / n_corr_time) * np.arange(n_corr_time, 0, -1)

        # construct Jacobian (convolution matrix)
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

        # --- create pool ONCE if parallel ---
        if num_dets['parallel']:
            logger.info(f"pool create | workers={cpu_cnt}", extra={"job_id": "-"})
            ctx = multiprocessing.get_context("spawn")
            pool = ctx.Pool(cpu_cnt, initializer=_worker_init, initargs=(log_queue,))

        rel_cov_change = 999.0
        outer_iter = 0

        while rel_cov_change > 0.1:
            outer_iter += 1
            print(f"Outer iteration {outer_iter}: rel_cov_change = {rel_cov_change:.3g}")
            logger.info(f"outer_iter start | iter={outer_iter} rel_cov_change={rel_cov_change:.3g}",
                        extra={"job_id": "-"})

            # rebuild J each iter
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

            # ----- best-estimate (me=0, h_uc=0) with nonnegativity -----
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

                # update sigma from best-estimate resids 
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
            # ----- end best-estimate -----

            # ----- parallel ensemble realizations -----
            args_list = [(C, J, iQ, y, sigma, sigma_max, n_h, dt) for _ in range(nreal)]
            indexed_args = list(enumerate(args_list, start=1))  # (job_id, args)

            if num_dets['parallel']:
                logger.info(f"pool map start | workers={cpu_cnt}", extra={"job_id": "-"})
                h_all_list = pool.starmap(compute_single_realization_logged, indexed_args)
                logger.info("pool map complete", extra={"job_id": "-"})
                h_all = np.column_stack(h_all_list)
            else:
                _worker_init(log_queue)
                cols = []
                for job_id, a in indexed_args:
                    cols.append(compute_single_realization_logged(job_id, a))
                h_all = np.column_stack(cols)

            # ----- cov pdate -----
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
                    # linear approx: triangular over first n_corr_time, no resizing n_h
                    cov = np.zeros(n_h)
                    if n_corr_time > 1:
                        cov[:n_corr_time] = (theta / (n_corr_time - 1)) * np.arange(n_corr_time - 1, -1, -1)

            # relative covariance change
            int_cov = dt * np.sum(cov) if cov.size else 1.0
            length_comp = min(len(cov), len(old_cov))
            rel_cov_change = float(np.sqrt(dt * np.sum((cov[:length_comp] - old_cov[:length_comp]) ** 2)) /
                                   max(int_cov, 1e-16) /
                                   max(sigma / max(np.max(y), 1e-16), 1e-16))
            print("Relative covariance change:", rel_cov_change)
            logger.info(f"outer_iter end | iter={outer_iter} rel_cov_change={rel_cov_change:.6g}",
                        extra={"job_id": "-"})

            # get ensemble stats for this outer iter
            h_all = np.real(h_all)
            h_mean = np.mean(h_all, axis=1)
            L_2 = np.sqrt(dt * np.sum((h_mean - kernel[:len(h_mean)]) ** 2))

        # ===== fin stats & save =====
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
        logger.info(f"final stats | m0={m_0:.6g} m1={m_1:.6g} m2={var_exp:.6g} RMSE={RMSE:.6g}",
                    extra={"job_id": "-"})

        t_h = dt * np.arange(n_h)
        h_p10 = np.percentile(h_all, 10, axis=1)
        h_p90 = np.percentile(h_all, 90, axis=1)

        plt.figure(figsize=(6, 4))
        skyBlue = [0.35, 0.70, 0.90]
        orange = [0.90, 0.60, 0.00]
        bluishGreen = [0.00, 0.62, 0.45]
        if k_type in ['bimodal', 'chapeau', 'gamma']:
            plt.plot(t_h, kernel[:n_h], '-k', linewidth=1.5, label='Known transfer fx')
        else:
            plt.plot(t_h, kernel[:n_h], '-k', linewidth=1.5, label='Estimated transfer fx')
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
        plt.savefig(os.path.join(figs_out, 'g_of_t_' + method + f'_{rounded_std}.png'), dpi=300)
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

        logger.info("end run", extra={"job_id": "-"})
        return df_subset, results_table

    finally:
        # stop the log
        if pool is not None:
            pool.close()
            pool.join()
        listener.stop()


if __name__ == '__main__':
    
    # run control vars:
    run_knwn_kernels = False
    noise_type = 'on_out'  # must be 'on_out','on_in_before_conv', or 'on_in_after_conv', needed for knwn kernels
    
    run_gambill = True
    
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

            time, in_signal, out_signal = df['time'].values, df['input'].values, df['output'].values
            
            noise_levels = [0.001, 0.03, 0.09]  # this is noise added onto output signal
            results_dict = {}
            methods = ['learn', 'cirpka']
            for method in methods:
                run_key = f'{kernel_shape}_{method}'
                for nl in noise_levels:
                    sigma_max = 0.09
                    run_key += f'_{nl}'
                    if nl > sigma_max:
                        sigma_max = nl + 0.03
                    num_dets = {'theta': 0.02, 'corr_time': 40, 
                                'sigma': nl, 'sigma_max': sigma_max, 
                                'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
                                'method': method}
                    nl = f'{nl:.3f}'
                    results, stats = deconv_parallel(df, num_dets, prefix=f'{method}_{kernel_shape}_{nl}')
                    results_dict[run_key] = {
                        'results': results,
                        'stats': stats
                    }
                    print(f"Results for {run_key} saved.")
        if plot_figs:
            pyplt.plot_deconv_results(os.path.join('known_kernels', 'python_make'), noise_type=noise_type, inset_on=False)
            
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
        
        sites_dict = {
            'highQ_R1': {'theta': 1e-4, 'corr_time': 10,
                        'sigma': 0.231, 'sigma_max': 1.0,
                        'n_h': 1600, 'nreal': 64, 'parallel': run_in_parallel,
                        'nworkers': num_workers, 'method': method},

            'highQ_R2': {'theta': 0.02, 'corr_time': 40,
                        'sigma': nl, 'sigma_max': sigma_max,
                        'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
                        'method': method},

            'highQ_long': {'theta': 0.02, 'corr_time': 40,
                        'sigma': nl, 'sigma_max': sigma_max,
                        'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
                        'method': method},

            'lowQ_R1': {'theta': 0.02, 'corr_time': 40,
                        'sigma': nl, 'sigma_max': sigma_max,
                        'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
                        'method': method},

            'lowQ_R2': {'theta': 0.02, 'corr_time': 40,
                        'sigma': nl, 'sigma_max': sigma_max,
                        'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
                        'method': method},

            'lowQ_long': {'theta': 0.02, 'corr_time': 40,
                        'sigma': nl, 'sigma_max': sigma_max,
                        'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
                        'method': method},

            'medQ_R1': {'theta': 0.02, 'corr_time': 40,
                        'sigma': nl, 'sigma_max': sigma_max,
                        'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
                        'method': method},

            'medQ_R2': {'theta': 0.02, 'corr_time': 40,
                        'sigma': nl, 'sigma_max': sigma_max,
                        'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
                        'method': method},

            'medQ_long': {'theta': 0.02, 'corr_time': 40,
                        'sigma': nl, 'sigma_max': sigma_max,
                        'n_h': 300, 'nreal': 56, 'parallel': run_in_parallel,
                        'method': method}
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

