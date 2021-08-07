[![Documentation Status](https://readthedocs.org/projects/simplenlopt/badge/?version=latest)](https://simplenlopt.readthedocs.io/en/latest/?badge=latest)

## Overview
A simple, SciPy like interface for the excellent nonlinear optimization library [NLopt](https://github.com/stevengj/nlopt) to make switching between SciPy and NLopt a piece of cake. SimpleNLopt's functions can act as a drop-in replacement for SciPy functions. Major differences compared to plain NLopt:

* SciPy like minimize(method='NLopt algorithm') API for NLopt's local optimizers
* Automatic numerical approximation of the gradient if analytical gradient is not available
* Automatic handling of constraints via the augmented lagrangian method without boilerplate code
* Scipy like interfaces to NLopt's global optimizers with hard stopping criteria
* SciPy like curve fitting using NLopt's algorithms

## Documentation
Refer to the online [documentation](https://simplenlopt.readthedocs.io/en/latest/index.html) for detailed description of the API and examples 

## Installation
```bash
pip install simplenlopt
```

## Example: Minimizing the Rosenbrock function in simplenlopt and scipy
```python
import simplenlopt
from scipy.optimize import rosen, rosen_der
import scipy
import numpy as np

x0 = np.array([0.5, 1.8])

res = simplenlopt.minimize(rosen, x0, jac = rosen_der)
print("Found optimum: ", res.x)

res_scipy = scipy.optimize.minimize(rosen, x0, jac = rosen_der)
print("Found optimum: ", res_scipy.x)
```