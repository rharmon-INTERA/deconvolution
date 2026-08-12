"""
Appendix figure: sensitivity of the estimated transfer function to the
user-defined kernel length n_g for three methods on gambill highQ_R1.

  - new Cirpka  : updated sumprob counts all (n_g-1) increments -> theta (and
                  hence the prior/smoothing) depends on n_g
  - Fienen      : sumprob counts only nonzero kernel entries -> invariant to n_g
  - COV-Learn   : empirical covariance of the kernel itself   -> invariant to n_g

Runs all six cases (3 methods x n_g in {40, 400}) with every other setting held
at the professor-validated highQ_R1 test configuration, then draws a 2x2 figure
(linear + semilog-y, per n_g) for the paper appendix.

Cirpka and COV-Learn go through deconv_parallel.deconv_parallel unchanged.
The Fienen variant is implemented HERE ONLY (new_cirpka_fienen.m sumprob),
so deconv_parallel.py is not modified in any way.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('known_kernels', 'python_make')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.linalg import toeplitz
from scipy.optimize import fminLook forward to it

from deconv_parallel import deconv_parallel, draw_accepted_realizations

# highQ_R1 settings (new_cirpka_gambill_test.m) — only n_h varies
BASE_SETTINGS = {'theta': 1.0, 'corr_time': 0.2, 'sigma': 0.03, 'sigma_max': 0.5,
                 'nreal': 20, 'parallel': False}
NG_CASES = [40, 400]

COLORS = {'learn': '#0A6092', 'fienen': '#009E73', 'cirpka': '#E69F00'}
STYLES = {'learn': '-', 'fienen': '-.', 'cirpka': '--'}
PLOT_ORDER = ['learn', 'fienen', 'cirpka']
# 'fienen' curve = num_nonzero counting = what deconv_parallel.py now implements
# as Modified-Cirpka. 'cirpka' curve = the (n_g-1) form being called out as flawed.
LABELS = {'learn': 'COV-Learn', 'fienen': 'Modified-Cirpka',
          'cirpka': r'Cirpka ($n_g-1$)'}

FIGDIR = os.path.join('field_studies', 'gambill', 'python_make', 'gambill_figs')
CURVE_CACHE = os.path.join(FIGDIR, 'appendix_ng_curves.csv')


def run_fienen(df, num_dets):
    """Replica of deconv_parallel's updated-cirpka outer loop with the Fienen
    sumprob (new_cirpka_fienen.m): num_nonzero/2*log(4*pi*theta*dt) instead of
    0.5*(n_g-1)*log(4*pi*theta*dt). Everything else identical."""
    t = np.asarray(df['time'].values, dtype=float)
    x = np.asarray(df['input'].values, dtype=float)
    y = np.asarray(df['output'].values, dtype=float)

    theta = float(num_dets['theta'])
    corr_time = float(num_dets['corr_time'])
    sigma = float(num_dets['sigma'])
    sigma_max = float(num_dets['sigma_max'])
    n_h = int(num_dets['n_h'])
    nreal = int(num_dets['nreal'])
    max_outer = int(num_dets.get('max_outer', 25))

    np.random.seed()

    dt = t[1] - t[0]
    corr_time = min(n_h * dt, corr_time)
    n_corr_time = int(np.ceil(corr_time / dt))

    cov = np.zeros(n_h)
    if n_corr_time > 0:
        cov[:n_corr_time] = (theta / n_corr_time) * np.arange(n_corr_time, 0, -1)

    input_fn = dt * x.copy()
    input_fn[input_fn < 1e-8] = 0.0
    r_vec = dt * np.zeros(n_h)
    J = toeplitz(input_fn, r_vec)

    rel_cov_change = 999.0
    outer_iter = 0

    while rel_cov_change > 1 and outer_iter < max_outer:
        outer_iter += 1

        c = dt * theta * np.arange(n_h, 0, -1)
        c[:len(cov)] = cov

        Q = toeplitz(c)
        Q += 1e-8 * np.eye(n_h)
        try:
            C = la.cholesky(Q, lower=False)
        except np.linalg.LinAlgError:
            d, v = np.linalg.eigh(Q)
            d[d < 0] = 1e-10
            Q = (v * d) @ v.T
            Q = 0.5 * (Q + Q.T)
            C = la.cholesky(Q, lower=False)
        iQ = la.inv(Q)

        # best estimate (me=0, h_uc=0) with nonnegativity + sigma update
        hL, nL = [], 0
        for _ in range(50):
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
                for j in range(nL):
                    Lmat[hL[j], j] = 1.0
                    Lmat[n_h, j] = 1.0
                mat = np.block([[umat, Lmat],
                                [Lmat.T, np.zeros((nL, nL))]])
                rhs = np.concatenate([urhs, np.zeros(nL)])
            else:
                mat, rhs = umat, urhs

            a = np.diag(mat).copy()
            if len(a) > n_h + 1:
                a[n_h + 1:] = 1.0
            imat = la.inv(np.diag(1 / a) @ mat) @ np.diag(1 / a)
            sol = imat @ rhs

            h_be = sol[:n_h] + sol[n_h]
            if nL > 0:
                h_be[hL] = 0.0
            nu = sol[n_h + 1:n_h + 1 + nL]

            sim = J @ h_be
            denom = max(1, len(y) - n_h + nL - 1)
            sigma = min(float(np.sqrt(np.dot(y - sim, y - sim) / denom)), sigma_max)

            hLold = list(hL)
            hLadd = list(np.where(h_be < 0)[0])
            hLrem = [hL[j] for j in range(len(nu)) if nu[j] > 0] if nL > 0 else []
            hL = sorted(list((set(hL) - set(hLrem)) | set(hLadd)))
            nL = len(hL)
            if sorted(hLold) == sorted(hL):
                break

        h_all_real = draw_accepted_realizations(
            C=C, J=J, iQ=iQ, y=y, sigma=sigma, sigma_max=sigma_max,
            n_h=n_h, dt=dt, nreal_target=nreal, max_attempts_factor=50,
            parallel=False, cpu_cnt=None)
        nreal_eff = nreal
        h_all = np.column_stack([h_be, h_all_real])

        old_cov = cov.copy()

        def sumprob_fienen(lnval):
            # new_cirpka_fienen.m: num_nonzero/2*log(4*pi*theta*dt) + sum(diff^2)/(4*theta*dt)
            theta_test = np.exp(lnval)
            thetadt = theta_test * dt
            lnpsum = 0.0
            for i in range(nreal_eff):
                h_i = h_all[:, i]
                num_nonzero = np.sum(h_i > 0)
                dif = np.diff(h_i)
                lnpsum += 0.5 * num_nonzero * np.log(4.0 * np.pi * thetadt)
                lnpsum += (dif @ dif) * 0.25 / thetadt
            return float(np.real(lnpsum))

        x0 = np.log(max(theta, 1e-12))
        maxfun = 200
        res = fmin(sumprob_fienen, x0, xtol=1e-4, ftol=1e-4,
                   maxiter=maxfun, maxfun=maxfun, disp=False)
        theta = float(np.exp(res[0]))

        cov = dt * theta * np.arange(n_h, 0, -1)

        length_comp = min(len(cov), len(old_cov))
        snr = max(sigma / max(float(np.max(y)), 1e-16), 1e-16)
        numer = np.sqrt(n_h * np.sum((cov[:length_comp] - old_cov[:length_comp])**2))
        den = max(float(np.sum(cov[:length_comp])), 1e-16)
        rel_cov_change = float(numer / den / snr)
        print(f"  fienen outer {outer_iter}: theta={theta:.4g}, sigma={sigma:.4g}, "
              f"rel_cov_change={rel_cov_change:.3g}")

    h_all = np.real(h_all)
    g_mean = np.mean(h_all, axis=1)
    t_g = dt * np.arange(n_h)
    return {'time': t_g, 'mean': g_mean, 'theta': theta, 'sigma': sigma,
            'outer_iters': outer_iter, 'converged': rel_cov_change <= 1}


def compute_curves():
    df = pd.read_csv(os.path.join('field_studies', 'gambill', 'data', 'highQ_R1.csv'))
    df = df.rename(columns={'in': 'input', 'out': 'output'})

    curves = {}   # (method, n_g) -> dict(time, mean)
    summary = []
    for n_g in NG_CASES:
        for method in ['cirpka', 'learn']:
            num_dets = dict(BASE_SETTINGS, n_h=n_g, method=method)
            print(f"\n===== {method}, n_g={n_g} =====")
            results, stats = deconv_parallel(df, num_dets, prefix=f'highQ_R1-ng{n_g}-{method}')
            curves[(method, n_g)] = {'time': results['time'].values,
                                     'mean': results['transfer_func_mean'].values}
            summary.append({'method': LABELS[method], 'n_g': n_g,
                            'sigma': float(stats['FinalSigma'].iloc[0]),
                            'outer_iters': int(stats['OuterIters'].iloc[0])})
        print(f"\n===== fienen, n_g={n_g} =====")
        fr = run_fienen(df, dict(BASE_SETTINGS, n_h=n_g))
        curves[('fienen', n_g)] = {'time': fr['time'], 'mean': fr['mean']}
        summary.append({'method': LABELS['fienen'], 'n_g': n_g,
                        'sigma': fr['sigma'], 'outer_iters': fr['outer_iters'],
                        'theta': fr['theta']})

    # cache curves so formatting tweaks can re-plot without re-running
    rows = []
    for (method, n_g), cu in curves.items():
        for tt, gg in zip(cu['time'], cu['mean']):
            rows.append({'method': method, 'n_g': n_g, 'time': tt, 'mean': gg})
    os.makedirs(FIGDIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(CURVE_CACHE, index=False)
    print(f"\nCurves cached to {CURVE_CACHE}")
    return curves, summary


def load_cached_curves():
    dfc = pd.read_csv(CURVE_CACHE)
    curves = {}
    for (method, n_g), grp in dfc.groupby(['method', 'n_g']):
        grp = grp.sort_values('time')
        curves[(method, int(n_g))] = {'time': grp['time'].values,
                                      'mean': grp['mean'].values}
    return curves


def main():
    rerun = '--rerun' in sys.argv
    if not rerun and os.path.isfile(CURVE_CACHE):
        print(f"Re-plotting from cache {CURVE_CACHE} (pass --rerun to recompute)")
        curves, summary = load_cached_curves(), []
    else:
        curves, summary = compute_curves()

    # ---- 2x2 appendix figure ----
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    panel = ['(a)', '(b)', '(c)', '(d)']
    for row, n_g in enumerate(NG_CASES):
        for col, logy in enumerate([False, True]):
            ax = axes[row, col]
            for method in PLOT_ORDER:
                cu = curves[(method, n_g)]
                yv = np.maximum(1e-10, cu['mean']) if logy else cu['mean']
                ax.plot(cu['time'], yv, STYLES[method], color=COLORS[method],
                        lw=1.8, label=LABELS[method])
            if logy:
                ax.set_yscale('log')
                ax.set_ylim(1e-2, 300)
            else:
                ax.set_ylim(0, 40)
            ax.set_xlim(0, 0.25)
            ax.set_ylabel(r'$\mathbf{g}$ [1/hr]' + (' (log)' if logy else ''))
            if row == 1:
                ax.set_xlabel('Time lag [hr]')
            ax.text(0.03, 0.97, f"{panel[row*2+col]} $n_g$ = {n_g}",
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=10, fontweight='bold')
            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', length=0)
            ax.tick_params(axis='both', which='major', direction='in',
                           length=6, width=1, top=True, bottom=True, left=True, right=True)
            ax.tick_params(axis='both', which='minor', direction='in',
                           length=3, width=0.8, top=True, bottom=True, left=True, right=True)
            ax.grid(True, alpha=0.4)
    axes[0, 0].legend(loc='upper right', fontsize=8, frameon=True)

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, 'appendix_ng_sensitivity')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out}.pdf/.png")
    if summary:
        print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
