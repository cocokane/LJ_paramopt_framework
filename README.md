# Active-Learning Assisted Framework for Efficient Parameterization of Non-Polarizable Force Fields

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.7%2B-green.svg)](https://www.python.org/)
[![GROMACS](https://img.shields.io/badge/GROMACS-2022.4-blue.svg)](http://www.gromacs.org/)

This repository contains code and documentation for the paper:

**Active-Learning Assisted General Framework for Efficient Parameterization of Non-Polarizable Force Fields** submitted to the *Journal of Chemical Theory and Computation*.

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
- **GROMACS**: Version 2022.4 (must be installed separately).

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
## Repository Structure

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

## Installation

### Clone the Repository
```bash
git clone https://github.com/your_username/active-learning-force-fields.git
cd active-learning-force-fields
```

### Install Python Dependencies
It is recommended to use a virtual environment:

```bash
Copy code
python3 -m venv venv
source venv/bin/activate
```
### Install required Python packages:
```bash
Copy code
pip install -r requirements.txt
```
### GROMACS Installation
Ensure that GROMACS 2022.4 is installed and properly configured in your environment. Detailed installation instructions can be found on the GROMACS website.
## Usage

To run the main script, use the following command:

```bash
python main.py
```
