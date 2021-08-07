.. simplenlopt documentation master file, created by
   sphinx-quickstart on Sun May  9 21:58:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to simplenlopt's documentation!
=======================================

Features
--------
- SciPy like minimize(method='NLopt algorithm') API for NLopt's local optimizers
- Automatic numerical approximation of the gradient if analytical gradient is not available
- Automatic handling of constraints via the augmented lagrangian method without boilerplate code
- Scipy like interfaces to NLopt's global optimizers with hard stopping criteria
- SciPy like curve fitting using NLopt's algorithms

Installation
--------

.. code:: bash

   pip install simplenlopt

Example: Minimizing the Rosenbrock function in simplenlopt and scipy
-----

.. code-block:: python

   import simplenlopt
   from scipy.optimize import rosen, rosen_der
   import scipy
   import numpy as np

   x0 = np.array([0.5, 1.8])

   res = simplenlopt.minimize(rosen, x0, jac = rosen_der)
   print("Found optimum: ", res.x)

   res_scipy = scipy.optimize.minimize(rosen, x0, jac = rosen_der)
   print("Found optimum: ", res_scipy.x)

Documentation
-----
.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Quickstart
   Constrained_Optimization
   Global_Opt
   InDepth_Gradients
   Curve_Fitting
   simplenlopt

   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
