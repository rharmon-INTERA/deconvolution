import os 
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%matplotlib widget


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
            l1, = ax.plot(data['time'], data['input'], linewidth=1.25, label='Input signal',color='grey')
            l2, = ax.plot(data['time'], data['output'], linewidth=1.25, label='Output signal',color='black')
            
            # add to secondary y axis:
            ax2 = ax.twinx()
            l3, = ax2.plot(data['time'], data['kernel'],'k--', linewidth=1.25,color='black', label='Selected Kernel')
            
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
            
           
def plot_deconv_results(results_dir, noise_type='on-out',inset_on=False):
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
    methods = ['Modified Cirpka' ,'COV-Learn']  # labels

    xmx = [10, 25,30] # x axis maximums ordered by kernel_types

    save_dir = os.path.join(results_dir, 'figs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if noise_type == 'on-out':
        noise_sub = 'y'
    elif noise_type == 'on-in-before-conv':
        noise_sub = r'x^{\ast}'
    elif noise_type == 'on-in-after-conv':
        noise_sub =  'x' 
    
    if not inset_on:
        pdf_path = os.path.join(save_dir, f'deconv_results_noise_{noise_type}.pdf')
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        with PdfPages(pdf_path) as pdf:
            for kernel in kernel_types:
                rdir = os.path.join(results_dir, kernel, 'outputs',noise_type)
                files = glob.glob(os.path.join(rdir, '**', '*_data_and_results.csv'), recursive=True)
                # get noise levels from files:
                noise_lvls = [os.path.basename(f).split('_')[3] for f in files]
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
                            ax.set_title(f'$\sigma_{noise_sub}$ = {noise}',fontsize=12)

                        if j == 0:
                            ax.set_ylabel(f"{mlabel}\n$k(t)$ [1/hr]")
                        if i == len(methods)-1:
                            ax.set_xlabel('Time lag [hr]')
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
                                bbox_to_anchor=(0.5, -0.005),
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

    
def plot_gambill(ws='.',rchnm='R2',dlvl='medQ',axes_pair=None,leg=False):
    discharge_dict = {'lowQ': r'$Q = 0.2\ \mathrm{m^3/s}$',
                    'medQ': r'$Q = 0.5\ \mathrm{m^3/s}$',
                    'highQ': r'$Q = 1.0\ \mathrm{m^3/s}$',}

    
    sim_fp = os.path.join(ws, f'{dlvl}_{rchnm}_learn_sim.csv')
    data_fp = os.path.join(ws, f'{dlvl}_{rchnm}_learn_data_and_results.csv')
    if not (os.path.isfile(sim_fp) and os.path.isfile(data_fp)):
        print(f"plot_gambill: missing outputs for {dlvl}_{rchnm}_learn — skipping panel")
        return False
    sim = pd.read_csv(sim_fp)
    data = pd.read_csv(data_fp)
    er = False
    if 'A_MIM' in rchnm or 'B_MIM' in rchnm:
        er = True

        if 'A_MIM' in rchnm:
            rchnm = rchnm.replace('A_MIM', '')
        if 'B_MIM' in rchnm:
            rchnm = rchnm.replace('B_MIM', '')
    ax = axes_pair[0]
    
    ax.plot(data['time'], data['transfer_func_mean'], '--', color='navy', linewidth=0.75,label='Ensemble mean', zorder=4)
    ax.plot(data['time'], data['transfer_func_p10'].values,color='lightgrey', zorder=2)
    ax.plot(data['time'], data['transfer_func_p90'].values,color='lightgrey', zorder=2)
    ax.set_ylabel(f'{discharge_dict[dlvl]}\n'+r'$\mathbf{g}$ [1/hr]')

    ax.fill_between(data['time'], data['transfer_func_p10'].values, data['transfer_func_p90'].values, color='grey', alpha=0.65, zorder=1, label='10th/90th\nprecentiles')

    ax.minorticks_on()

    ax.tick_params(axis='both', which='both', length=0)

    ax.tick_params(axis='both', which='major',
                direction='in',
                length=6, width=1,
                top=True, bottom=True, left=True, right=True)

    ax.tick_params(axis='both', which='minor',
                direction='in',
                length=3, width=0.8,
                top=True, bottom=True, left=True, right=True)

    ax.grid(True,alpha=0.5)

    ax.set_xlim(0,0.501)
    ax.set_ylim(0,20)
    if dlvl == 'highQ':
        ax.set_xlabel('Time lag [hr]')
    if dlvl != 'highQ':
        ax.set_xticklabels([])
    
    if leg:
        ax.legend(loc='upper right', fontsize=8, frameon=True)

    ax = axes_pair[1]
    
    if er:
        ax.plot(sim['time'], sim['simulated'], label='Simulated\n bulk EC',linestyle='--',color='grey',zorder=4)
        ax.plot(sim['time'], sim['output'], label='Measured\n bulk EC',color='k',zorder=3)
    else:
        ax.plot(sim['time'], sim['simulated'], label='Simulated\n fluid EC',linestyle='--',color='grey',zorder=4)
        ax.plot(sim['time'], sim['output'], label='Measured\n fluid EC',color='k',zorder=3)        
    
    if leg:
        ax.legend(loc='upper right', fontsize=8, frameon=True)

    ax.minorticks_on()

    ax.tick_params(axis='both', which='both', length=0)

    ax.tick_params(axis='both', which='major',
                direction='in',
                length=6, width=1,
                top=True, bottom=True, left=True, right=True)

    ax.tick_params(axis='both', which='minor',
                direction='in',
                length=3, width=0.8,
                top=True, bottom=True, left=True, right=True)
    if er:
        ax.set_ylabel('Bulk fluid electrical \nconductivity [$\\mu$S/cm]')
    else:
        ax.set_ylabel('Fluid electrical \nconductivity [$\\mu$S/cm]')
    if dlvl == 'highQ':
        ax.set_xlabel('Time lag [hr]')
    ax.set_ylim(0,60)
    ax.set_xlim(0,15)
    ax.grid(True,alpha=0.5)
 
    if dlvl != 'highQ':
        ax.set_xticklabels([])

    return True


def plot_reach(pdf_nm='reach#.pdf',rchnm='R2'):
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    with PdfPages(pdf_nm) as pdf:
        mws = os.path.join('field_studies','gambill','python_make','outputs')
        fig, axes = plt.subplots(3, 2, figsize=(5.5, 8))

        ax12 = [axes[0, 0], axes[0, 1]]
        rchnm=rchnm
        dlvl='lowQ'
        ws = os.path.join(mws,f'{dlvl}_{rchnm}_learn')
        ok_low = plot_gambill(ws=ws, rchnm=rchnm, dlvl=dlvl, axes_pair=ax12,leg=True)
        axes[0, 0].text(0.15, 0.98,
                subplot_labels[0],
                transform=axes[0, 0].transAxes,
                ha='right', va='top',
                fontsize=11, fontweight='bold',
                zorder=50, clip_on=False
            ) 
        axes[0, 1].text(0.15, 0.98,
                subplot_labels[1],
                transform=axes[0, 1].transAxes,
                ha='right', va='top',
                fontsize=11, fontweight='bold',
                zorder=50, clip_on=False
            )   
                      
        ax34 = [axes[1, 0], axes[1, 1]]
        rchnm=rchnm
        dlvl='medQ'
        ws = os.path.join(mws,f'{dlvl}_{rchnm}_learn')
        ok_med = plot_gambill(ws=ws, rchnm=rchnm, dlvl=dlvl, axes_pair=ax34)
        axes[1, 0].text(0.15, 0.98,
                subplot_labels[2],
                transform=axes[1, 0].transAxes,
                ha='right', va='top',
                fontsize=11, fontweight='bold',
                zorder=50, clip_on=False
            ) 
        axes[1, 1].text(0.15, 0.98,
                subplot_labels[3],
                transform=axes[1, 1].transAxes,
                ha='right', va='top',
                fontsize=11, fontweight='bold',
                zorder=50, clip_on=False
            )

        ax56 = [axes[2, 0], axes[2, 1]]
        rchnm=rchnm
        dlvl='highQ'
        ws = os.path.join(mws,f'{dlvl}_{rchnm}_learn')
        ok_high = plot_gambill(ws=ws, rchnm=rchnm, dlvl=dlvl, axes_pair=ax56)
        axes[2, 0].text(0.15, 0.98,
                subplot_labels[4],
                transform=axes[2, 0].transAxes,
                ha='right', va='top',
                fontsize=11, fontweight='bold',
                zorder=50, clip_on=False
            ) 
        axes[2, 1].text(0.15, 0.98,
                subplot_labels[5],
                transform=axes[2, 1].transAxes,
                ha='right', va='top',
                fontsize=11, fontweight='bold',
                zorder=50, clip_on=False
            )

        if not (ok_low or ok_med or ok_high):
            print(f"plot_reach: no outputs found for reach {rchnm} — skipping {pdf_nm}")
            plt.close(fig)
            return

        plt.tight_layout()
        # save:
        pdf.savefig(fig, bbox_inches='tight')
        fig.savefig(pdf_nm.replace('.pdf','.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_gambill_compare_highQ_R1(
    mws_learn=os.path.join('field_studies','gambill','python_make','outputs'),
    mws_cirpka=os.path.join('field_studies','gambill','python_make','outputs'),
    rchnm='R1',
    dlvl='highQ',
    outpath=None,
    figsize=(10, 7.2),
    tf_xlim=(0, 0.25),
    tf_ylim=(0, 35),
    tf_log_ylim=(1e-2, 100),
    ec_xlim=(0, 15),
    ec_ylim=(0, 25),
    add_legend=True,

    ec_inset=False,
    ec_inset_xlim=None,   
    ec_inset_ylim=None,   
    ec_inset_bbox=(0.62, 0.33, 0.36, 0.34),  # (x0, y0, w, h) in ax_c axes fraction
    ec_inset_label="",
    ec_inset_grid=True,

    show_tf_pbands=True,
    pb_ls=":",         
    pb_lw=1.5,
    pb_alpha=0.9,
    log_floor=1e-12,  
):
    """
    Creates a 3-panel figure:
      (a) transfer functions (linear y): mean + optional p10/p90
      (b) transfer functions (semilog-y): mean + optional p10/p90
      (c) measured downstream EC + simulated convolution (learn vs cirpka)
          with optional inset zoom.

    Expected folder structure (per method workspace):
      {mws_method}/{dlvl}_{rchnm}_{method}/
        - {dlvl}_{rchnm}_{method}_sim.csv
        - {dlvl}_{rchnm}_{method}_data_and_results.csv
    """

    def _load(method: str, mws_base: str):
        ws = os.path.join(mws_base, f"{dlvl}_{rchnm}_{method}")
        sim_fp  = os.path.join(ws, f"{dlvl}_{rchnm}_{method}_sim.csv")
        data_fp = os.path.join(ws, f"{dlvl}_{rchnm}_{method}_data_and_results.csv")

        if not os.path.isfile(sim_fp):
            raise FileNotFoundError(f"Missing sim CSV: {sim_fp}")
        if not os.path.isfile(data_fp):
            raise FileNotFoundError(f"Missing data/results CSV: {data_fp}")

        sim = pd.read_csv(sim_fp)
        data = pd.read_csv(data_fp)
        return ws, sim, data


    _, sim_learn,  data_learn  = _load("learn",  mws_learn)
    _, sim_cirpka, data_cirpka = _load("cirpka", mws_cirpka)

    needed_base = ["time", "transfer_func_mean"]
    needed_p = ["transfer_func_p10", "transfer_func_p90"]
    for nm, d in [("learn", data_learn), ("cirpka", data_cirpka)]:
        missing = [c for c in needed_base if c not in d.columns]
        if missing:
            raise KeyError(f"{nm} data/results CSV missing columns: {missing}")
        if show_tf_pbands:
            missing_p = [c for c in needed_p if c not in d.columns]
            if missing_p:
                raise KeyError(f"{nm} data/results CSV missing columns: {missing_p}")

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        nrows=2, ncols=2,
        width_ratios=[1.0, 1.55],
        height_ratios=[1, 1],
        wspace=0.28, hspace=0.22
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1])

    # Style constants
    col_learn = "#0A6092"
    col_cirp  = "#E69F00"
    ls_mean_learn = "-"
    ls_mean_cirp  = "--"

    # -------------------------
    # (a) Transfer functions (linear y)
    # -------------------------
    ax_a.plot(
        data_learn["time"], data_learn["transfer_func_mean"],
        linestyle=ls_mean_learn, linewidth=1.2, color=col_learn, label="COV-Learn"
    )
    ax_a.plot(
        data_cirpka["time"], data_cirpka["transfer_func_mean"],
        linestyle=ls_mean_cirp, linewidth=1.2, color=col_cirp, label="Cirpka"
    )

    if show_tf_pbands:
        # learn p10/p90
        ax_a.plot(
            data_learn["time"], data_learn["transfer_func_p10"],
            linestyle=pb_ls, linewidth=pb_lw, color=col_learn, alpha=pb_alpha,
            label="COV-Learn p10/p90"
        )
        ax_a.plot(
            data_learn["time"], data_learn["transfer_func_p90"],
            linestyle=pb_ls, linewidth=pb_lw, color=col_learn, alpha=pb_alpha
        )
        # cirpka p10/p90
        ax_a.plot(
            data_cirpka["time"], data_cirpka["transfer_func_p10"],
            linestyle=pb_ls, linewidth=pb_lw, color=col_cirp, alpha=pb_alpha,
            label="Cirpka p10/p90"
        )
        ax_a.plot(
            data_cirpka["time"], data_cirpka["transfer_func_p90"],
            linestyle=pb_ls, linewidth=pb_lw, color=col_cirp, alpha=pb_alpha
        )

    ax_a.set_xlim(*tf_xlim)
    ax_a.set_ylim(*tf_ylim)
    ax_a.set_ylabel(r'$\mathbf{g}$ [1/hr]')
    ax_a.grid(True, alpha=0.5)
    ax_a.minorticks_on()

    # -------------------------
    # (b) Transfer functions (semilog-y)
    # -------------------------
    ax_b.semilogy(
        data_learn["time"], data_learn["transfer_func_mean"],
        linestyle=ls_mean_learn, linewidth=1.2, color=col_learn
    )
    ax_b.semilogy(
        data_cirpka["time"], data_cirpka["transfer_func_mean"],
        linestyle=ls_mean_cirp, linewidth=1.2, color=col_cirp
    )

    if show_tf_pbands:
        lp10 = np.maximum(data_learn["transfer_func_p10"].to_numpy(), log_floor)
        lp90 = np.maximum(data_learn["transfer_func_p90"].to_numpy(), log_floor)
        cp10 = np.maximum(data_cirpka["transfer_func_p10"].to_numpy(), log_floor)
        cp90 = np.maximum(data_cirpka["transfer_func_p90"].to_numpy(), log_floor)

        ax_b.semilogy(
            data_learn["time"], lp10,
            linestyle=pb_ls, linewidth=pb_lw, color=col_learn, alpha=pb_alpha
        )
        ax_b.semilogy(
            data_learn["time"], lp90,
            linestyle=pb_ls, linewidth=pb_lw, color=col_learn, alpha=pb_alpha
        )

        ax_b.semilogy(
            data_cirpka["time"], cp10,
            linestyle=pb_ls, linewidth=pb_lw, color=col_cirp, alpha=pb_alpha
        )
        ax_b.semilogy(
            data_cirpka["time"], cp90,
            linestyle=pb_ls, linewidth=pb_lw, color=col_cirp, alpha=pb_alpha
        )

    ax_b.set_xlim(*tf_xlim)
    ax_b.set_ylim(*tf_log_ylim)
    ax_b.set_xlabel("Time [hr]")
    ax_b.set_ylabel(r'$\mathbf{g}$ [1/hr] (log)')
    ax_b.grid(True, which="both", alpha=0.5)
    ax_b.minorticks_on()

    # -------------------------
    # (c) Downstream EC: measured + simulated convolutions
    # -------------------------
    ax_c.plot(
        sim_learn["time"], sim_learn["output"],
        color="k", linestyle="-", linewidth=1.3, label="Measured EC"
    )
    ax_c.plot(
        sim_learn["time"], sim_learn["simulated"],
        color=col_learn, linestyle="-.", linewidth=1.2, label="Simulated (COV-Learn)"
    )
    ax_c.plot(
        sim_cirpka["time"], sim_cirpka["simulated"],
        color=col_cirp, linestyle="--", linewidth=1.2, label="Simulated (Cirpka)"
    )

    ax_c.set_xlim(*ec_xlim)
    ax_c.set_ylim(*ec_ylim)
    ax_c.set_xlabel("Time [hr]")
    ax_c.set_ylabel('Fluid electrical\nconductivity [$\\mu$S/cm]')
    ax_c.grid(True, alpha=0.5)
    ax_c.minorticks_on()

    # -------------------------
    # inset zoom inside panel (c)
    # -------------------------
    ax_c_inset = None
    if ec_inset:
        if ec_inset_xlim is None or ec_inset_ylim is None:
            raise ValueError("If ec_inset=True, provide ec_inset_xlim=(x1,x2) and ec_inset_ylim=(y1,y2).")

        x0, y0, w, h = ec_inset_bbox
        ax_c_inset = ax_c.inset_axes([x0, y0, w, h], transform=ax_c.transAxes)

        ax_c_inset.plot(sim_learn["time"], sim_learn["output"],
                        color="k", linestyle="-", linewidth=1.0)
        ax_c_inset.plot(sim_learn["time"], sim_learn["simulated"],
                        color=col_learn, linestyle="-.", linewidth=1.0)
        ax_c_inset.plot(sim_cirpka["time"], sim_cirpka["simulated"],
                        color=col_cirp, linestyle="--", linewidth=1.0)

        ax_c_inset.set_xlim(*ec_inset_xlim)
        ax_c_inset.set_ylim(*ec_inset_ylim)
        ax_c_inset.tick_params(labelsize=8)
        ax_c_inset.minorticks_on()
        if ec_inset_grid:
            ax_c_inset.grid(True, alpha=0.35)

        if ec_inset_label:
            ax_c_inset.text(0.02, 0.98, ec_inset_label,
                            transform=ax_c_inset.transAxes,
                            ha="left", va="top", fontsize=8, fontweight="bold")

        ax_c.indicate_inset_zoom(ax_c_inset, edgecolor="0.3", linewidth=0.8)


    for ax in (ax_a, ax_b, ax_c):
        ax.tick_params(axis='both', which='both', length=0)
        ax.tick_params(axis='both', which='major',
                       direction='in', length=6, width=1,
                       top=True, bottom=True, left=True, right=True)
        ax.tick_params(axis='both', which='minor',
                       direction='in', length=3, width=0.8,
                       top=True, bottom=True, left=True, right=True)

    if ax_c_inset is not None:
        ax_c_inset.tick_params(axis='both', which='both', length=0)
        ax_c_inset.tick_params(axis='both', which='major',
                               direction='in', length=4, width=0.9,
                               top=True, bottom=True, left=True, right=True)
        ax_c_inset.tick_params(axis='both', which='minor',
                               direction='in', length=2, width=0.7,
                               top=True, bottom=True, left=True, right=True)

    # panel labels
    ax_a.text(0.03, 0.98, "(a)", transform=ax_a.transAxes,
              ha="left", va="top", fontsize=11, fontweight="bold", zorder=50)
    ax_b.text(0.03, 0.98, "(b)", transform=ax_b.transAxes,
              ha="left", va="top", fontsize=11, fontweight="bold", zorder=50)
    ax_c.text(0.02, 0.98, "(c)", transform=ax_c.transAxes,
              ha="left", va="top", fontsize=11, fontweight="bold", zorder=50)

    if add_legend:
        ax_a.legend(loc="upper right", fontsize=8, frameon=True)
        ax_c.legend(loc="upper right", fontsize=8, frameon=True)

    if outpath is not None:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    return fig, (ax_a, ax_b, ax_c)


def plot_estimated_covariance_by_kernel(
    results_dir,
    kernel='gamma',
    noise_type='on-out',
    noise_level='0.030',
    outname=None,
    use_abs=True
):
    """
    Plot estimated covariance for a single kernel, with both
    Modified Cirpka and COV-Learn overlaid on the same axes.

    Parameters
    ----------
    results_dir : str
        Base results directory, e.g. 'known_kernels/python_make'
    kernel : str
        'chapeau', 'gamma', or 'bimodal'
    noise_type : str
        One of 'on-out', 'on-in-before-conv', 'on-in-after-conv'
    noise_level : str
        Noise level string as used in filenames, e.g. '0.030'
    outname : str or None
        Optional output filename. If None, a default is used.
    use_abs : bool
        If True, plot abs(covariance) so semilogy works even if values
        are slightly negative from numerical noise.
    """
    method_keys = ['cirpka', 'learn']
    method_labels = ['Modified Cirpka', 'COV-Learn']
    method_styles = {
        'cirpka': {'color': 'grey', 'linestyle': '--', 'linewidth': 1.25},
        'learn':  {'color': 'navy', 'linestyle': '-',  'linewidth': 1.25},
    }

    save_dir = os.path.join(results_dir, 'figs')
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))

    rdir = os.path.join(results_dir, kernel, 'outputs', noise_type)

    for mkey, mlabel in zip(method_keys, method_labels):
        pattern = os.path.join(
            rdir,
            f'noise_added_{noise_level}',
            f'{mkey}_{kernel}_{noise_type}_{noise_level}_{mkey}_covariance.csv'
        )
        matches = glob.glob(pattern)

        if not matches:
            print(f'Could not find covariance file for {kernel}, {mkey}: {pattern}')
            continue

        cov_df = pd.read_csv(matches[0])

        if 'lag_time' not in cov_df.columns or 'covariance' not in cov_df.columns:
            print(f"Missing required columns in {matches[0]}")
            continue

        x = cov_df['lag_time'].values
        y = cov_df['covariance'].values

        if use_abs:
            y = np.abs(y)

        y = np.maximum(y, 1e-12)

        dt_cov = x[1] - x[0] if len(x) > 1 else 1.0
        cov_raw = cov_df['covariance'].values
        cov0 = cov_raw[0] if cov_raw[0] > 1e-16 else 1e-16
        corr_time = dt_cov * np.sum(cov_raw) / cov0

        ax.semilogy(
            x, y,
            label=f'{mlabel}  ($T_c$ = {corr_time:.3g} hr)',
            **method_styles[mkey]
        )

    ax.set_xlabel('Time lag [hr]')
    ax.set_ylabel('Estimated covariance [1/hr$^2$]')
    ax.set_title(f'Known {kernel.capitalize()} kernel', fontsize=10)
    ax.set_ylim(bottom=1e-7)
    ax.grid(True, which='both', alpha=0.4)
    ax.minorticks_on()

    ax.tick_params(axis='both', which='both', length=0)
    ax.tick_params(axis='both', which='major',
                   direction='in', length=6, width=1,
                   top=True, bottom=True, left=True, right=True)
    ax.tick_params(axis='both', which='minor',
                   direction='in', length=3, width=0.8,
                   top=True, bottom=True, left=True, right=True)

    ax.legend(loc='best', frameon=True)

    fig.tight_layout()

    if outname is None:
        outname = f'estimated_covariance_{kernel}_{noise_type}_{noise_level}.pdf'

    outpath_pdf = os.path.join(save_dir, outname)
    outpath_png = outpath_pdf.replace('.pdf', '.png')

    fig.savefig(outpath_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(outpath_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved covariance plot to {outpath_pdf}")


if __name__ == "__main__":
    set_graph_specifications()
    figdir = os.path.join('known_kernels','python_make','figs_known_kernels')

    plot_known_kernels(figdir=figdir)
    
    results_dir = os.path.join('known_kernels','python_make')
    plot_deconv_results(results_dir, noise_type='on-out',inset_on=False)
    plot_deconv_results(results_dir, noise_type='on-in-before-conv',inset_on=False)
    plot_deconv_results(results_dir, noise_type='on-in-after-conv',inset_on=False)
    
    figdir = os.path.join('field_studies','gambill','python_make','gambill_figs')
    if not os.path.exists(figdir):
         os.makedirs(figdir)
    plot_reach(pdf_nm=os.path.join(figdir,'reach2_FEC.pdf'),rchnm='R2')
    plot_reach(pdf_nm=os.path.join(figdir,'reach1_FEC.pdf'),rchnm='R1')
  
    plot_gambill_compare_highQ_R1(
        rchnm="R1",
        dlvl="highQ",
        outpath=os.path.join(figdir, "gambill_highQ_R1_compare.pdf"),
        ec_inset=True,
        ec_inset_xlim=(3.62, 3.82),
        ec_inset_ylim=(14.8, 15.5),
        # right side, vertically centered (tweak if needed)
        ec_inset_bbox=(0.57, 0.34, 0.35, 0.32),
        ec_inset_label="",
    )
    
    plot_estimated_covariance_by_kernel(
        results_dir=os.path.join('known_kernels', 'python_make'),
        kernel='chapeau',
        noise_type='on-out',
        noise_level='0.030'
    )

    plot_estimated_covariance_by_kernel(
        results_dir=os.path.join('known_kernels', 'python_make'),
        kernel='gamma',
        noise_type='on-out',
        noise_level='0.030'
    )

    plot_estimated_covariance_by_kernel(
        results_dir=os.path.join('known_kernels', 'python_make'),
        kernel='bimodal',
        noise_type='on-out',
        noise_level='0.030'
    )
        