## REDLY Overview

REDLY is a Python and Julia-based codebase for the design, training, and validation of physics-informed neural networks (PINNs) for power system applications.  The provided code and datasets are specifically tailored for the use of PINNs as surrogates for ACPF prediction, but they can be extended to support more general economic dispatch problems (e.g., DCOPF, ACOPF).

REDLY is available under the BSD License (see [LICENSE.txt](LICENSE.txt)).

This code is based on that developed for the paper *Physics-Informed Machine Learning with Optimization-Based Guarantees:  Applications to AC Power Flow*, J. Jalving, et al., International Journal of Electric Power and Energy Systems (157), 2024.  Please cite if you find this code useful in your research.

## Features

* **src/acpf**: Python packages for PINN training with physics-informed regularization constraints
  * Provides example callbacks for Lagrangian Dual training of physics-informed regularizations (F. Fioretto, T. W. Mak, P. V. Hentenryck, Predicting AC Optimal Power Flows: Combining Deep Learning and Lagrangian Dual Methods, in: AAAI Conf on AI, 2020) and Lottery Ticket pruning to compress PINN sizes (J. Frankle, M. Carbin, The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks, in: 7th International Conference on Learning Representations, ICLR 2019)
  * Provides example primitives for physics-based regularization constraints in neural network models, including Kirchhoff's laws, operational limits, and flow balance
* **verification**: Julia scripts to measure output violation of trained models
  * Gives methods to compare violations of physics-regularized and non-regularized models under a range of conditions
* **data**:  Provides sample datasets for ACPF prediction using the IEEE Case-14 and Case-118 test configurations
  * Will need to download the pglib_opf_case118_ieee.m and pglib_opf_case14_ieee.m case files from [pglib](https://github.com/power-grid-lib/pglib-opf/tree/master) and place into the corresponding data directories

## Requirements

#### Python
We recommend using [Anaconda](https://www.anaconda.com/distribution/) to handle Python dependencies:
* Tensorflow 2.X
* [Egret](https://github.com/grid-parity-exchange/Egret/tree/main)
* Numpy
* Scipy
* Pandas
* Matplotlib
* Seaborn
* Jupyter

#### Julia
The verification scripts require Jump wrappers for [IPOPT](https://github.com/jump-dev/Ipopt.jl) and [GUROBI](https://github.com/jump-dev/Gurobi.jl).  Users will need to provide their own licenses for the underlying solvers, as needed.  Additional requirements include:
* Formatting
* CSV
* JSON
* DataFrames
* Statistics
* Plots
* JuMP
* SparseArrays
* Complementarity
* Flux

The Python modules for designing and training PINNs can be run seperately from the Julia modules for verification.

## Getting Started

#### PINN design and training (Python)
The following scripts provide example end-to-end workflows to define and train both physics-regularized and non-regularized neural networks for ACPF prediction:
*  acpf_train_example.py - basic PINN training with Lagrangian Dual callback
*  acpf_train_and_prune_example.py - includes LTH pruning outer loop

These scripts also have corresponding Jupyter notebook variations
  
#### PINN verification (Julia)
The scripts to validate model violations under a range of conditions are located in **verification/verification_scripts**.  These should be run *after* models have been created and saved using the Python scripts above:
* run_acpf_input_verification.jl - compare model solutions to true ACPF solutions
* run_acpf_verifier_cases.jl - estimate model violations under a range of optimization conditions
* run_acpf_wcs_gurobi_outputs.jl - obtain per-output violations
* run_acpf_wcs_gurobi.jl - obtain per-bus violations

The scripts to generate visualizations from these results are also located in **verification/verification_scripts**.



