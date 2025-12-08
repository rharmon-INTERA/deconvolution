# Deconvolution
-----------------------------------------------
This repository contains the complete workflow, Python environment, and datasets used to develop, test, and apply an improved time-domain deconvolution method for estimating transfer functions in hydrologic systems.

In simple terms, deconvolution attempts to recover how a tracer signal is transformed as it travels from an upstream input location to a downstream output location. Traditional approaches often assume a fixed covariance structure for (i.e, a linear covariance structure), which can constrain the recovered kernels—especially under high noise conditions.

The method implemented here removes that assumption by directly solving for the actual autocovariance from the data. It also incorporates a machine-learning–based covariance estimator to improve the estimated covariance upon each iteration. The new method also allows for added error bounds on the recovered kernels. Together, these improvements make the deconvolution method more robust and more stable across both synthetic and field datasets.

The repository includes the following key directories:

known_kernels/ — Contains the chapeau, gamma, and bimodal kernels used for synthetic testing and validation of the improved method.

field_studies/ — Contains input data results from applying the machine-learning approach to the Gambill et al. (2025) field dataset.

## Getting Started with Python
-----------------------------------------------
The Python workflow was used to generate every figure and table in Harmon et al. (YYYY). We validated the Python implementation of the improved deconvolution method by comparing it against the original MATLAB version using the synthetic test cases from the manuscript. Both implementations recovered nearly identical transfer functions and error bounds. Minor deviations between the two are expected and stem primarily from (i) the inability to enforce identical random number seeds for generating synthetic noise across platforms, and (ii) subtle numerical differences between SciPy and MATLAB’s optimization and linear-algebra routines.
### Installing Python Dependencies with Conda or Mamba
There are several options for installing the Python environments needed to run the workflow:

Anaconda Distribution (https://www.anaconda.com/download)
Full installation with many scientific packages included by default that are not relevant to this workflow.

Miniconda (https://docs.anaconda.com/miniconda/)
A minimal installer used to create custom Python environments.

Mambaforge (https://github.com/conda-forge/miniforge#mambaforge)
A lightweight installer preconfigured to use the faster Mamba solver and the conda-forge channel.
Recommended — it is open-source, avoids Anaconda’s new potentital paywall adds, uses fewer resources, and solves environments faster.

Once one of these is installed, create the environment using the provided file deconv_env.yml.

Using mamba (recommended):

```bash
	$ mamba env create -f deconv_env.yml
	$ mamba activate deconv
```
 
```bash
	$ conda env create -f deconv_env.yml
	$ conda activate deconv
```
    
## Python Scripts

###deconv_parallel.py

This is the main driver script for the workflow. It handles:
- Serial execution of deconvoltion method
- Parallel execution of the deconvolution method


###py_plotting.py

Contains all plotting functions used throughout the workflow, including:

- Kernel comparison plots

- Input/output time-series figures

- Generation of latex tables

Figures for the known kernels that are shown in Harmon et al. (YYYY) are saved to ./deconvolution/known_kernels/python_make/figs, tables are saved to deconvolution/known_kernels/python_make/tables

Figures of the Gambill et al. (2025) are located in ./deconvolution/field_studies/gambill/python_make/gambill_figs

## Getting Started with MATLAB
-----------------------------------------------
The MATLAB version of the repo is significantly less automated. There is a stand alone script to run each of known transfer function examples under the varying degrees of noise. There are also standalone scripts to run each of the Gambill et al. (2025) input and output tracer pairs. While the Matlab scripts do not contain the functions to generate figures and tables seen in the report, all relevant results are still saved to ".csv" and/or ".mat" files. The MATLAB scripts also include active plotting, that plot results at each sub-iteration within the deconvolution method. A lot can be learned about your problem by watching the improvement or derailing of your solution as you step through these sub-iterations.  
