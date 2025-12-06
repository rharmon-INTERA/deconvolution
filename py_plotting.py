import os 
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def set_graph_specifications():
    rc_dict = {'font.family': 'DejaVu Sans',
               'axes.labelsize': 9,
               'axes.titlesize': 9,
               'axes.linewidth': 0.5,
               'xtick.labelsize': 8,
               'xtick.top': True,
               'xtick.bottom': True,
               'xtick.major.size': 7.2,
               'xtick.minor.size': 3.6,
               'xtick.major.width': 0.5,
               'xtick.minor.width': 0.5,
               'xtick.direction': 'in',
               'ytick.labelsize': 8,
               'ytick.left': True,
               'ytick.right': True,
               'ytick.major.size': 7.2,
               'ytick.minor.size': 3.6,
               'ytick.major.width': 0.5,
               'ytick.minor.width': 0.5,
               'ytick.direction': 'in',
               'pdf.fonttype': 42,
               'savefig.dpi': 300,
               'savefig.transparent': True,
               'legend.fontsize': 9,
               'legend.frameon': False,
               'legend.markerscale': 1.
               }
    mpl.rcParams.update(rc_dict)


def plot_known_kernels(figdir='.'):
    """
    Function to plot known kernels generated in python_make folder.
    
    Parameters:
    - figdir: Directory containing the input, kernel, and output files.
    
    Returns:
    - None, but will generate and save a PDF and PNG of the known kernels.
    """
    kernel_types = ['chapeau', 'gamma', 'bimodal']

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    # ensure that csv files exist
    for kernel in kernel_types:
        rdir = os.path.join('known_kernels','python_make', kernel,'py_inputs')
        csv_path = os.path.join(rdir, f'{kernel}.csv')
        if not os.path.exists(csv_path):
            print(f"Error: Required file {csv_path} not found. Please run kwn_kernel_make.py first.")
            sys.exit(1)

    pdf_path = os.path.join(figdir, 'known_kernels.pdf')
    subplot_labels = ['(a)', '(b)', '(c)']
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(len(kernel_types), 1, figsize=(5.5, 8), sharex=True)
        cnt= 0
        for kernel in kernel_types:
            rdir = os.path.join('known_kernels','python_make', kernel,'py_inputs')
            data = pd.read_csv(os.path.join(rdir, f'{kernel}.csv'))
            ax = axes[kernel_types.index(kernel)]
            # main y axis:
            l1, = ax.plot(data['time'], data['input'], linewidth=1.25, label='Input ignal',color='grey')
            l2, = ax.plot(data['time'], data['output'], linewidth=1.25, label='Output signal',color='black')
            
            # add to secondary y axis:
            ax2 = ax.twinx()
            l3, = ax2.plot(data['time'], data['kernel'],'k--', linewidth=1.25,color='black', label='Known Kernel')
            
            ax.set_title(f'{kernel.capitalize()}', fontsize=10)
            
     
            ax.set_ylabel('Concentration\n[ppb]')
            ax2.set_ylabel('Kernel $k(t)$\n[1/hr]', rotation=270, labelpad=22)
            

            ax.tick_params(axis='both', which='both', direction='in', colors='black', top=True)
                    
            lines = [l1, l2, l3]
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels, loc='upper right')

            ax.text(
                    0.08, 0.95, subplot_labels[cnt],
                    transform=ax.transAxes,
                    ha='right', va='top',
                    fontsize=11, fontweight='bold',
                    zorder=50, clip_on=False
                )
            if cnt == 2:
                ax.set_xlabel('Time lag [hr]')
            cnt+=1
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"known_kernels.png"),
                dpi=300, bbox_inches="tight")
    pdf.savefig(fig)
    plt.close(fig)
    pdf.close()
            
           
def plot_deconv_results(results_dir, noise_type='on_out',inset_on=False):
    """
    Function to plot results from deconv of knwon kernels.
    
    Parameters:
    - results_dir: directory containing the result files.
    - save_dir: fig out directory.
    - noise_type: just a string to indicate when noise was added in the deconv process. Must be 'on_out',
      'on_in_before_conv', or 'on_in_after_conv'.
    
    Returns:
    - None, but will genreate figs
    """
    kernel_types = ['chapeau', 'gamma', 'bimodal']
    method_keys = ['cirpka' ,'learn']  # substrings in filenames
    methods = ['Modified Cirpka' ,'Machine Learning']  # labels

    xmx = [10, 25,30] # x axis maximums ordered by kernel_types

    save_dir = os.path.join(results_dir, 'figs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if noise_type == 'on_out':
        noise_sub = 'y'
    elif noise_type == 'on_in_before_conv':
        noise_sub = 'x'
    elif noise_type == 'on_in_after_conv':
        noise_sub = r'x^{\ast}'  
    
    if not inset_on:
        pdf_path = os.path.join(save_dir, f'deconv_results_noise_{noise_type}.pdf')
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        with PdfPages(pdf_path) as pdf:
            for kernel in kernel_types:
                rdir = os.path.join(results_dir, kernel, 'outputs')
                files = glob.glob(os.path.join(rdir, '**', '*_data_and_results.csv'), recursive=True)
                # get noise levels from files:
                noise_lvls = [os.path.basename(f).split('_')[2] for f in files]
                noise_lvls = sorted(list(set(noise_lvls)))
                fig, axes = plt.subplots(len(methods), len(noise_lvls), 
                                        figsize=(8.5, 8), sharex=True, sharey=True)
                
                for i, (mkey, mlabel) in enumerate(zip(method_keys, methods)):
                    for j, noise in enumerate(noise_lvls):
                        ax = axes[i, j]
                        # find file:
                        match = [f for f in files if (noise in f) and (mkey in f)]
                        if not match:
                            ax.set_visible(False)
                            continue
                        data = pd.read_csv(os.path.join(match[0]))
                        
                        ax.plot(data['time'], data['kernel'],color='black',linewidth=1.25, label='Constructed Kernel', zorder=3)
                        ax.plot(data['time'], data['transfer_func_mean'], '--', color='navy', linewidth=0.75,label='Ensemble Mean', zorder=4)
                        #ax.plot(data['time'], data['transfer_func_p10'],color='lightgrey', zorder=2)
                        #ax.plot(data['time'], data['transfer_func_p90'],color='lightgrey', zorder=2)
                        ax.fill_between(data['time'], data['transfer_func_p10'], data['transfer_func_p90'], color='grey', alpha=0.65, zorder=1, label='10th/90th Percentiles')
                        
                        ax.tick_params(axis='both', which='both', direction='in', length=6, width=1, colors='black')
                        ax.tick_params(axis='both', which='both', direction='in', length=3, width=1, colors='black', top=True, right=True)
                        
                        if i == 0 and i <= len(noise_lvls)-1:
                            ax.set_title(f'$\sigma_{noise_sub}$ = {noise}',fontsize=10)

                        if j == 0:
                            ax.set_ylabel(f"{mlabel}\n$k(t)$ [1/hr]")

                        # if i == 0 and j == len(noise_lvls)-1:
                        #     ax.legend(loc='upper right',fontsize=8)
                        
                        ax.set_xlim(0, xmx[kernel_types.index(kernel)])
                        
                        idx = i * len(noise_lvls) + j
                        ax.text(
                            0.13, 0.98, subplot_labels[idx],
                            transform=ax.transAxes,
                            ha='right', va='top',
                            fontsize=11, fontweight='bold',
                            zorder=50, clip_on=False
                        )
                        
                        pretty_labels = {
                            'Constructed Kernel': 'Constructed Kernel',
                            'Ensemble Mean': 'Ensemble Mean',
                            '10th/90th Percentiles': r'Area between 10$^{\mathrm{th}}$ and 90$^{\mathrm{th}}$ percentiles',
                        }
                        desired_keys = list(pretty_labels.keys())

                        handle_map = {}
                        for ax_row in np.ravel(axes):
                            if not ax_row.get_visible():
                                continue
                            h, l = ax_row.get_legend_handles_labels()
                            for handle, label in zip(h, l):
                                if label in desired_keys and label not in handle_map:
                                    handle_map[label] = handle

                        handles = [handle_map[k] for k in desired_keys if k in handle_map]
                        labels  = [pretty_labels[k] for k in desired_keys if k in handle_map]

                        if handles:
                            fig.legend(
                                handles, labels,
                                loc='lower center',
                                bbox_to_anchor=(0.5, 0.01),
                                ncol=max(1, len(handles)),   # one row
                                fontsize=9,
                                frameon=True
                            )
                            fig.tight_layout()
                            fig.subplots_adjust(bottom=0.08)  
                pdf.savefig(fig)

                # save png
                png_path = os.path.join(save_dir, f'{kernel}_noise_type_{noise_type}.png')
                fig.savefig(png_path, dpi=300)
            
                plt.close(fig)

        print(f"Saved PDF to {pdf_path}")

    if inset_on:
        pdf_path = os.path.join(save_dir, 'kernels_by_method_with_insets.pdf')
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
        with PdfPages(pdf_path) as pdf:
            for kernel in kernel_types:
                rdir = os.path.join(results_dir, kernel, 'outputs')
                files = sorted(f for f in os.listdir(rdir) if f.endswith('_h_results.csv'))
            
                fig, axes = plt.subplots(len(methods), len(noise_lvls),
                                        figsize=(8.5, 9), sharex=True, sharey=True)
                
                for i, (mkey, mlabel) in enumerate(zip(method_keys, methods)):
                    for j, noise in enumerate(noise_lvls):
                        ax = axes[i, j]
                        
                        match = [f for f in files if (noise in f) and (mkey in f)]
                        if not match:
                            ax.set_visible(False)
                            continue
                        
                        data = pd.read_csv(os.path.join(rdir, match[0]))
                    
                        pct_lower = data.iloc[:, 2:].quantile(0.1, axis=1)
                        pct_upper = data.iloc[:, 2:].quantile(0.9, axis=1)
           
                        ax.plot(data['time'], data['known_kernel'],
                                color='black', linewidth=1.25,
                                label='Constructed Kernel', zorder=3)
                        ax.plot(data['time'], data['h_mean'],
                                '--', color='navy', linewidth=0.75,
                                label='Ensemble Mean', zorder=4)
        
                        ax.fill_between(data['time'], pct_lower, pct_upper,
                                        color='grey', alpha=0.65, zorder=1,
                                        label=r'Area between $10^{\mathrm{th}}$ and $90^{\mathrm{th}}$ percentiles'
                                        )
                    
                        ax.tick_params(axis='both', which='both',
                                    direction='in', length=6, width=1)
                        ax.tick_params(top=True, right=True)
                        ax.set_xlim(0, xmx[kernel_types.index(kernel)])

                        idx = i * len(noise_lvls) + j
                        ax.text(
                            0.1, 0.98, subplot_labels[idx],
                            transform=ax.transAxes,
                            ha='right', va='top',
                            fontsize=11, fontweight='bold',
                            zorder=50, clip_on=False
                        )

                    
                        if j == 0:
                            ax.set_ylabel(mlabel)
                        if i == 0:
                            ax.set_title(f'Noise: {noise}')
                        if i == len(methods)-1:
                            ax.set_xlabel('Time lag [hr]')
                    
                        axins = inset_axes(ax, width="40%", height="40%", loc='upper right')
                    
                        # get peak
                        peak_idx = data['known_kernel'].idxmax()
                        t_peak = data['time'].iat[peak_idx]
                        total_span = data['time'].iloc[-1] - data['time'].iloc[0]
                        if kernel == 'chapeau':
                            zoom_frac = 0.03   # 2% of total span → tighter zoom
                        else:
                            zoom_frac = 0.05   # 5% default
                        half_win = zoom_frac * total_span  # zoom ±5% of span
                        
                        mask = (data['time'] >= t_peak - half_win) & (data['time'] <= t_peak + half_win)
                        
                        axins.plot(data['time'][mask], data['known_kernel'][mask],
                                color='black', lw=1.25)
                        axins.plot(data['time'][mask], data['h_mean'][mask],
                                '--', color='navy', lw=0.75)
                        axins.fill_between(data['time'][mask],
                                        pct_lower[mask], pct_upper[mask],
                                        color='grey', alpha=0.65)
                        
                        axins.set_xlim(t_peak - half_win, t_peak + half_win)
                        ymin = data['known_kernel'][mask].min()
                        ymax = data['known_kernel'][mask].max()
                        padding = 0.2 * (ymax - ymin)    # 20% vertical padding
                        axins.set_ylim(ymin - padding, ymax + padding)
                        
                        from matplotlib.patches import Rectangle, ConnectionPatch

                        x0 = t_peak + half_win                  # right edge of the zoom window
                        y_top   = data['known_kernel'][mask].max() 
                        y_bottom= data['known_kernel'][mask].min() 
                        rect = Rectangle(
                                (t_peak - half_win, ymin- padding),       # lower‐left corner of the box
                                2*half_win,                      # width
                                (ymax + padding) - (ymin - padding),                     # height
                                fill=False,
                                edgecolor='0.5',
                                linewidth=1
                            )
                        ax.add_patch(rect)
                        con1 = ConnectionPatch(
                            xyA=(x0, y_top), coordsA=ax.transData,
                            xyB=(0, 1),      coordsB=axins.transAxes,
                            arrowstyle='-', lw=1, edgecolor='0.5'
                        )
                        ax.add_artist(con1)

                        con2 = ConnectionPatch(
                            xyA=(x0, y_bottom), coordsA=ax.transData,
                            xyB=(0, 0),         coordsB=axins.transAxes,
                            arrowstyle='-', lw=1, edgecolor='0.5'
                        )
                        ax.add_artist(con2)
                
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(
                    handles, labels,
                    loc='lower center',
                    ncol=len(labels),        
                    bbox_to_anchor=(0.555, 0.02),  
                    frameon=True 
                )

            
                fig.supylabel('Convolution Kernels\n $\\mathit{k}(t)$ [1/hr]',
                                x=0.05, y=0.54, rotation='vertical',
                                ha='center', va='center')

                fig.tight_layout(rect=[0, 0.05, 1, 1])
                pdf.savefig(fig)
                
                png_path = os.path.join(save_dir, f'{kernel}_by_method_with_insets.png')
                fig.savefig(png_path, dpi=300)
                plt.close(fig)

        print(f"Saved PDF to {pdf_path}")
    

if __name__ == "__main__":
    set_graph_specifications()
    figdir = os.path.join('known_kernels','python_make','figs')

    plot_known_kernels(figdir=figdir)

    results_dir = os.path.join('known_kernels','python_make')