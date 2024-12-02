# Active-Learning Assisted Framework for Efficient Parameterization of Force Fields

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.7%2B-green.svg)](https://www.python.org/)
[![GROMACS](https://img.shields.io/badge/GROMACS-blue.svg)](http://www.gromacs.org/)

This repository contains code and documentation for the paper:

**Active-Learning Assisted General Framework for Efficient Parameterization of Force Fields** submitted to the *Journal of Chemical Theory and Computation*.

**Authors**: Yati<sup>1</sup>, Yash Kokane<sup>2</sup>, and Anirban Mondal<sup>1,*</sup>  
<sup>1</sup>Department of Chemistry, Indian Institute of Technology Gandhinagar, Gujarat, 382355, India  
<sup>2</sup>Department of Materials Engineering, Indian Institute of Technology Gandhinagar, Gujarat, 382355, India  
*Email*: [amondal@iitgn.ac.in](mailto:amondal@iitgn.ac.in)

---

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)

---

## Introduction

This repository provides the implementation of an efficient approach to optimizing Lennard-Jones (LJ) force field parameters for sulfone molecules. The framework combines Genetic Algorithms (GA) with Gaussian Process Regression (GPR) to significantly reduce the computational expense associated with traditional force field parameterization methods.

The code is designed to:

- Start with initial LJ parameter guesses near the OPLS (Optimized Potentials for Liquid Simulations) values.
- Use GROMACS molecular dynamics simulations to evaluate the fitness of these parameters by comparing simulated densities and radial distribution functions (RDFs) to reference data obtained from Ab Initio Molecular Dynamics (AIMD) simulations.
- Train a GPR model on the evaluated parameters and their fitness values.
- Use a GA to predict new parameter sets that minimize the fitness function.
- Iterate this process to efficiently converge on optimized force field parameters.

## Prerequisites

To run this repository, the following prerequisites are required:

### System Requirements
- **Operating System**: Linux or macOS (Windows with WSL is supported but not tested extensively).
- **Python Version**: Python 3.7 or higher.
- **GROMACS**: GROMACS 2022.4 was used during development. Compatibility with other versions has not been evaluated. Please ensure GROMACS is installed separately.
### Python Dependencies
The following Python libraries are required and are listed in `requirements.txt`:
- `numpy`
- `pandas`
- `scikit-learn`
- `deap`
- `argparse`

You can install these dependencies using the command:
```bash
pip install -r requirements.txt
```

## Installation

### Clone the Repository
```bash
git clone https://github.com/cocokane/LJ_paramopt_framework.git
cd LJ_paramopt_framework
```

### Install Python Dependencies
It is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```
### Install required Python packages:
```bash
pip install -r requirements.txt
```
### GROMACS Installation
Ensure that GROMACS 2022.4 is installed and properly configured in your environment. Detailed installation instructions can be found on the GROMACS website.
## Usage

### Input Data

The reference RDF data for OO and SS pairs from AIMD simulations must be stored in the `reference/` directory. The initial parameter guesses and fitness data are stored in `reference/MD_data.csv`.

### Required File Structure 

```plaintext
├── README.md                   # Project documentation
├── main.py                     # Main Python script
├── requirements.txt            # Python dependencies
├── reference/
│   ├── OO_target.out           # Reference OO RDF data from AIMD
│   ├── SS_target.out           # Reference SS RDF data from AIMD
│   └── MD_data.csv             # Initial parameter guesses and fitness data
├── output.txt                  # Generated LJ parameters from GA-GPR
└── LICENSE                     # Project license
```

As the framework was developed primarily for sulfone molecules, the main script `main.py` is configured to optimize LJ parameters for OO and SS pairs. The script can be modified to optimize parameters for other atom pairs by changing the variables in the `main.py` script.

Ensure that the data files are correctly placed in the `reference/` directory before running the main script.

To run the main script, use the following command:

```bash
python main.py
```

The script will execute the GA-GPR optimization process and output the predicted optimal LJ parameters to `output.txt`.

The next step involves performing MD simulations in GROMACS using the parameters stored in output.txt. Sample files, including the training dataset (MD_data.csv), AIMD reference RDFs (OO_target.out and SS_target.out), and the output file (output.txt), are provided as an example to familiarize users with the expected data structure.

## Steps to run the framework

1. Use OPLS non bonded paremeters to generate new 200 parameters within ±5% of deviation
2. Perform classical MD on 200 new parameters to extract density and required RDFs and create MD_data.csv
3. Run main.py to generate output.txt, which contains optimized LJ parameters. This script uses MD_data.csv to train the GPR model and predict new parameters, and completes one iteration of the optimization process.
4. Best parameters from this iteration are stored in output.txt, perform classical MD using GROMACS for these parameters.
5. Based on your accuracy requirements, either finish or repeat step 3 and 4 by updating MD_data.csv until desired accuracy is achieved.
