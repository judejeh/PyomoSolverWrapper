# Pyomo Solver Wrapper

## Introduction

A third-party python-based solver wrapper for [Pyomo](http://www.pyomo.org/about).

## Requirements
1. [Python v3.5+](https://www.python.org/downloads/release/python-370/) 

## Installation

The package can be installed using the command:
```
pip install PyoSolveWrapper
```

Also by directly downloading an appropriate release or cloning the main development repository from GitHub:
```
git clone https://github.com/judejeh/PyomoSolverWrapper.git
```

Change into the "PyomoSolverWrapper" directory created by Git or extracted from the downloaded zip file and run:
```
python setup.py install
```

## Usage
Import the solve wrapper:
```
import PyoSolveWrapper as slw 
```
This package solely aids the optimisation model solve process. Hence, having created an appropriate pyomo model object (further referred to as 'model'), an instance of the solver wrapper is first created:
```
Solver = slw.SolverWrapper()
```
Basic solver options can be changed where appropriate:
```
Solver.solver_name = 'cbc'
Solver.threads = 2
Solver.time_limit = 1200
Solver.rel_gap = 0.1
Solver.solver_progress = True
```
The package attempts to find the appropriate solver executable from the system PATH. However, should the solver path be in a different location, this path may be included as:
``` 
Solver.solver_path = '<PATH_TO_SOLVER>'
```
Note that whatever solver supplied should be compatible with Pyomo.

Where solvers are not locally installed, optimisation models may be solved using the [NEOS server](https://neos-server.org/neos/) if doing so for non-commercial purposes.
Note that for now, results are not post-processed.

**Finally**, the model can be solved as:
```
Solver.solve_model(model)
```

Results obtained from the model solution is automatically post-processed and saved with 'Solver' and can be accessed via the dictionary:
``` 
Solver.final_results[<type>][<name>]
```
where `<type>` is one of of 'sets', 'parameters' or 'variables'; and `<name>` is the actual model component name used in 'model'.

For e.g., a parameter and variable name 'A' and 'y' in the Pyomo model object 'model' can be accessed after successful model solution via:
``` 
Solver.final_results['parameters']['A']
```
and
``` 
Solver.final_results['variables']['y']
```
respectively.

Where parameters or variables are index of set(s), they can be accessed in the order of the sets they were declared by when creating the model component.

## Disclaimer
At the time of writing, the authors of this package have no affiliation with Pyomo or NEOS server. The package was written to simplify an optimisation model solving process using Pyomo, especially relating to post-processing results and dealing with multiple solvers.