# Deconvolution
-----------------------------------------------
This repository contains the complete workflow, Python environment, and datasets used to develop, test, and apply an improved time-domain deconvolution method for estimating transfer functions (residence-time distributions, RTDs) in hydrologic systems.

In simple terms, deconvolution attempts to recover how a tracer signal is transformed as it travels from an upstream input location to a downstream output location. Traditional approaches often assume a fixed covariance structure for the noise, which can bias the recovered kernels—especially under high noise conditions or in real field data.

The method implemented here removes that assumption by directly solving for the actual autocovariance from the data. It also incorporates a machine-learning–based covariance estimator and provides error bounds on the recovered kernels. Together, these improvements make the method significantly more robust, less biased, and more stable across both synthetic and field datasets.

The repository includes the following key directories:

known_kernels/ — Contains the chapeau, gamma, and bimodal kernels used for synthetic testing and validation of the improved method.

field_studies/ — Contains results from applying the machine-learning approach to the Gambill et al. (2025) field dataset.

## Getting Started
-----------------------------------------------
### Installing Python Dependencies with Conda or Mamba
There are several options for installing Conda-based Python environments:

Anaconda Distribution (https://www.anaconda.com/download)
Full installation with many scientific packages included by default.

Miniconda (https://docs.anaconda.com/miniconda/)
A minimal installer used to create custom Python environments for specific workflows.

Mambaforge (https://github.com/conda-forge/miniforge#mambaforge)
A lightweight installer preconfigured to use the faster Mamba solver and the conda-forge channel.
Recommended — it is open-source, avoids Anaconda’s new paywall risks, uses fewer resources, and solves environments faster.

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

- Parallel execution of the deconvolution algorithm



###py_plotting.py

Contains all plotting functions used throughout the workflow, including:

- Kernel comparison plots

- Input/output time-series figures