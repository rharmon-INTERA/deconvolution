import os
import sys
sys.path.insert(0,os.path.abspath(os.path.join('..','..')))
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import py_plotting as pyplt
pyplt.set_graph_specifications()


def make_chapeau(add_input_noise=0.0):
    """
    Generate a chapeau kernel, convolve with a Gaussian-mix input,
    optionally add uniform input noise (with std = add_input_noise),
    and save results as PNG, PDF, and CSV.

    Parameters
    ----------
    add_input_noise : float
        Desired standard deviation of uniform noise added to the input signal.
        Use 0.0 for no noise.
    """
    
    # --- setup ---
    prefix = "chapeau"
    ins_pth = os.path.join(prefix, "py_inputs")
    os.makedirs(ins_pth, exist_ok=True)

    #if add_input_noise > 0.0:
    #    prefix += f'_in-noise_bef_conv_{add_input_noise:.3f}'

    # --- parameters ---
    N = 512       # number of points
    dt = 0.1      # time step
    L = dt * (N - 1)  # length (not directly used but kept for parity)

    # --- allocate ---
    out = np.zeros(N)
    in_signal = np.zeros(N)
    kernel = np.zeros(N)
    pad = np.zeros(N)

    # --- chapeau kernel ---
    peak = N // 20  # floor(N/20)

    kernel[:peak] = np.arange(0, peak)                     # rising
    kernel[peak:peak + peak + 1] = np.arange(peak, -1, -1) # falling incl. peak
    kernel = kernel / (dt * kernel.sum())                  # normalize (integral = 1)

    # --- signals ---
    t = dt * np.arange(N)
    in_signal = (
        np.exp(((t - (dt * N / 5)) ** 2) / (-4)) +
        0.25 * np.exp(((t - (dt * N / 3)) ** 2) / (-8))
    )
    in_signal *= 10.0

    if add_input_noise > 0.0:
        desired_std = add_input_noise  
        std_uniform_minus_05 = 1 / np.sqrt(12)
        noise_mult = desired_std / std_uniform_minus_05
        noise_vec = noise_mult * (np.random.rand(*in_signal.shape) - 0.5)
        in_signal = in_signal + noise_vec

    # --- convo ---
    full_conv = dt * np.convolve(
        np.concatenate([in_signal, pad]),
        np.concatenate([kernel, pad]),
        mode="full"
    )
    out = full_conv[:N]

    pdf_path = os.path.join(ins_pth, f"{prefix}_in_tfx_out.pdf")
    with PdfPages(pdf_path) as pdf:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot(t, kernel, linewidth=1.5)
        ax1.set_ylabel("Amplitude")
        if add_input_noise > 0.0:
            ax1.set_title(f"Chapeau Kernel with Input Noise = {add_input_noise}")
        else:
            ax1.set_title("Chapeau Kernel")
        ax2.plot(t, in_signal, label="Input Signal", linewidth=1.5)
        ax2.plot(t, out, label="Output (Convolved)", linewidth=1.5)
        ax2.legend()
        ax2.set_title("Input Signal and Output")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Amplitude")

        fig.tight_layout()
        fig.savefig(os.path.join(ins_pth, f"{prefix}_in_tfx_out.png"), dpi=300, bbox_inches="tight")
        pdf.savefig(fig)
        plt.close(fig)

    # --- moments ---
    m_0 = dt * kernel.sum()
    m_1 = (dt / m_0) * np.sum(t * kernel)
    var_exp = (dt / m_0) * np.sum(((t - m_1) ** 2) * kernel)

    print(f"m_0 = {m_0:.6f}")
    print(f"m_1 = {m_1:.6f}")
    print(f"var_exp = {var_exp:.6f}")

    moments_df = pd.DataFrame({
        "m_0": [m_0],
        "m_1": [m_1],
        "var_exp": [var_exp]
    })
    moments_df.to_csv(os.path.join(ins_pth, f"{prefix}_moments.csv"), index=False)

    df = pd.DataFrame({
        "time": t,
        "input": in_signal,
        "output": out,
        "kernel": kernel
    })
    df.to_csv(os.path.join(ins_pth, f"{prefix}.csv"), index=False)

    return df

def make_gamma(add_input_noise=0.0):
    """
    Generate a gamma kernel, convolve with a Gaussian-mix input,
    optionally add uniform input noise (with std = add_input_noise),
    and save results as PNG, PDF, and CSV.

    Parameters
    ----------
    add_input_noise : float
        Desired standard deviation of uniform noise added to the input signal.
        Use 0.0 for no noise.
    """

    # --- setup ---
    prefix = "gamma"
    ins_pth = os.path.join(prefix, "py_inputs")
    os.makedirs(ins_pth, exist_ok=True)

    
    #if add_input_noise > 0.0:
    #    prefix += f'_in_noise_bef_conv_{add_input_noise:.3f}'

    # --- parameters ---
    N = 512       # number of points
    dt = 0.1      # time step
    k_shape = 3.0 # shape parameter
    theta = 2.0   # scale parameter

    # --- time array ---
    t = dt * np.arange(N)
    pad = np.zeros(N)

    # --- gamma kernel (vectorized) ---
    coef = 1.0 / (math.gamma(k_shape) * (theta ** k_shape))
    kernel = coef * np.power(t, k_shape - 1, where=(t > 0)) * np.exp(-t / theta)
    if k_shape == 1.0:
        kernel[t == 0] = 1.0 / theta
    else:
        kernel[t == 0] = 0.0

    kernel = kernel / (dt * np.sum(kernel))

    # --- input signal ---
    in_signal = (
        np.exp(((t - (dt * N / 5)) ** 2) / (-4))
        + 0.25 * np.exp(((t - (dt * N / 3)) ** 2) / (-8))
    ) * 10.0

    # --- optional input noise ---
    if add_input_noise > 0.0:
        std_uniform_minus_05 = 1.0 / np.sqrt(12.0)
        noise_mult = add_input_noise / std_uniform_minus_05
        noise_vec = noise_mult * (np.random.rand(*in_signal.shape) - 0.5)
        in_signal = in_signal + noise_vec

    # --- convolution (pad + full, then crop first N) ---
    full_conv = dt * np.convolve(
        np.concatenate([in_signal, pad]),
        np.concatenate([kernel, pad]),
        mode="full"
    )
    out = full_conv[:N]

    # --- moments ---
    m_0 = dt * kernel.sum()
    m_1 = (dt / m_0) * np.sum(t * kernel)
    var_exp = (dt / m_0) * np.sum(((t - m_1) ** 2) * kernel)

    print(f"[gamma] m_0={m_0:.6f}  m_1={m_1:.6f}  var_exp={var_exp:.6f}")

    moments_df = pd.DataFrame({
        "m_0": [m_0],
        "m_1": [m_1],
        "var_exp": [var_exp]
    })
    moments_df.to_csv(os.path.join(ins_pth, f"{prefix}_moments.csv"), index=False)

    # --- plotting ---
    pdf_path = os.path.join(ins_pth, f"{prefix}_in_tfx_out.pdf")
    with PdfPages(pdf_path) as pdf:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        if add_input_noise > 0.0:
            title1 = f"Gamma Kernel (k={k_shape}, θ={theta}) | Input Noise σ={add_input_noise}"
        else:
            title1 = f"Gamma Kernel (k={k_shape}, θ={theta})"

        ax1.plot(t, kernel, linewidth=1.5)
        ax1.set_title(title1)
        ax1.set_ylabel("Amplitude")

        ax2.plot(t, in_signal, label="Input Signal", linewidth=1.5)
        ax2.plot(t, out, label="Output (Convolved)", linewidth=1.5)
        ax2.legend()
        ax2.set_title("Input Signal and Output")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Amplitude")

        fig.tight_layout()
        fig.savefig(os.path.join(ins_pth, f"{prefix}_in_tfx_out.png"),
                    dpi=300, bbox_inches="tight")
        pdf.savefig(fig)
        plt.close(fig)

    df = pd.DataFrame({
        "time": t,
        "input": in_signal,
        "output": out,
        "kernel": kernel
    })
    df.to_csv(os.path.join(ins_pth, f"{prefix}.csv"), index=False)

    return df

def make_bimodal(add_input_noise=0.0):
    """
    Create a bimodal (two-Gaussian) transfer function kernel, convolve with a
    standard Gaussian-mix input, optionally add uniform input noise with target
    std, and save PNG/PDF/CSV outputs (no .mat).

    Parameters
    ----------
    add_input_noise : float
        Target standard deviation for added uniform noise to the input signal.
        Uses scaled U(-0.5, 0.5). Set 0.0 for no noise.
    N : int
        Number of time points.
    dt : float
        Time step.
    mu1, sigma1 : float
        Mean and standard deviation of the first Gaussian peak. If mu1 is None,
        defaults to dt * N / 5 (to match your MATLAB).
    mu2, sigma2 : float
        Mean and standard deviation of the second Gaussian peak. If mu2 is None,
        defaults to dt * 2.25 * N / 6 (to match your MATLAB).
    second_peak_scale : float
        Amplitude multiplier for the second Gaussian (MATLAB uses 0.4).
    """
  
    prefix = "bimodal"
    ins_pth = os.path.join(prefix, "py_inputs")
    os.makedirs(ins_pth, exist_ok=True)

    #if add_input_noise > 0.0:
    #    prefix += f'_in_noise_bef_conv_{add_input_noise:.3f}'

    # --- parameters ---
    N = 512       # number of points
    dt = 0.1      # time step   
    mu1=None 
    sigma1=2.75
    mu2=None
    sigma2=2.5
    second_peak_scale=0.4

    # --- time vector ---
    t = dt * np.arange(N)
    pad = np.zeros(N)

    if mu1 is None:
        mu1 = dt * N / 5.0
    if mu2 is None:
        mu2 = dt * 2.25 * N / 6.0

    # --- bimodal kernel (two Gaussians, second scaled by 0.4) ---
    g1 = np.exp(-((t - mu1) ** 2) / (2.0 * sigma1 ** 2))
    g2 = np.exp(-((t - mu2) ** 2) / (2.0 * sigma2 ** 2))
    kernel = g1 + second_peak_scale * g2

    # normalize so dt * sum(kernel) = 1
    kernel = kernel / (dt * kernel.sum())

    # --- input signal (same Gaussian mix used in your other functions) ---
    in_signal = (
        np.exp(((t - (dt * N / 5)) ** 2) / (-4))
        + 0.25 * np.exp(((t - (dt * N / 3)) ** 2) / (-8))
    ) * 10.0

    # --- optional uniform noise with desired std ---
    if add_input_noise > 0.0:
        std_uniform_minus_05 = 1.0 / np.sqrt(12.0)
        noise_mult = add_input_noise / std_uniform_minus_05
        in_signal = in_signal + noise_mult * (np.random.rand(*in_signal.shape) - 0.5)

    # --- convolution (pad + full, then crop first N) ---
    full_conv = dt * np.convolve(
        np.concatenate([in_signal, pad]),
        np.concatenate([kernel, pad]),
        mode="full"
    )
    out = full_conv[:N]

    # --- moments ---
    m_0 = dt * kernel.sum()
    m_1 = (dt / m_0) * np.sum(t * kernel)
    var_exp = (dt / m_0) * np.sum(((t - m_1) ** 2) * kernel)
    print(f"[bimodal] m_0={m_0:.6f}  m_1={m_1:.6f}  var_exp={var_exp:.6f}")

    moments_df = pd.DataFrame({
        "m_0": [m_0],
        "m_1": [m_1],
        "var_exp": [var_exp]
    })
    moments_df.to_csv(os.path.join(ins_pth, f"{prefix}_moments.csv"), index=False)

    pdf_path = os.path.join(ins_pth, f"{prefix}_in_tfx_out.pdf")
    with PdfPages(pdf_path) as pdf:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        if add_input_noise > 0.0:
            title = (f"Bimodal Transfer Function | μ1={mu1:.2f}, σ1={sigma1:.2f}; "
                     f"{second_peak_scale}×[μ2={mu2:.2f}, σ2={sigma2:.2f}] | "
                     f"Input noise σ={add_input_noise}")
        else:
            title = (f"Bimodal Transfer Function | μ1={mu1:.2f}, σ1={sigma1:.2f}; "
                     f"{second_peak_scale}×[μ2={mu2:.2f}, σ2={sigma2:.2f}]")

        ax1.plot(t, kernel, linewidth=1.5)
        ax1.set_title(title)
        ax1.set_ylabel("Amplitude")

        ax2.plot(t, in_signal, label="Input Signal", linewidth=1.5)
        ax2.plot(t, out, label="Output (Convolved)", linewidth=1.5)
        ax2.legend()
        ax2.set_title("Input Signal and Output")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Amplitude")

        fig.tight_layout()
        fig.savefig(os.path.join(ins_pth, f"{prefix}_in_tfx_out.png"),
                    dpi=300, bbox_inches="tight")
        pdf.savefig(fig)
        plt.close(fig)

    df = pd.DataFrame({
        "time": t,
        "input": in_signal,
        "output": out,
        "kernel": kernel
    })
    df.to_csv(os.path.join(ins_pth, f"{prefix}.csv"), index=False)

    return df

if __name__ == '__main__':
    print('Making known kernels')
    pyplt.set_graph_specifications()
    
    # no input noise:
    make_chapeau(add_input_noise=0.0)
    make_gamma(add_input_noise=0.0)
    make_bimodal(add_input_noise=0.0)

    # add input noise before convolution:
    make_chapeau(add_input_noise=0.03)
    make_gamma(add_input_noise=0.03)
    make_bimodal(add_input_noise=0.03)
