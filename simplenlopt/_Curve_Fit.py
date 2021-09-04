import numpy as np
from simplenlopt._Core import minimize, is_gradient_based
from simplenlopt._Global_Optimization import direct, mlsl, stogo, isres, esch, crs
from inspect import signature
from scipy.linalg import svd
from scipy.optimize._numdiff import approx_derivative
from time import time
#from numba import njit
from warnings import warn

def is_gradient_based_global(method):

    if method in ['mlsl', 'MLSL']:

        return True

    else:

        return is_gradient_based(method)

def curve_fit(f, xdata, ydata, p0=None, sigma=None, bounds = None, 
        loss = 'squared', method = 'auto', jac = None, f_scale = 1, **minimize_kwargs):
    '''
    Curve fitting using NLopt's local optimizers in SciPy style

    Parameters
    --------
    f : callable
        Must be of the form ``f(xdata, *params)`` 
    xdata : ndarray (n, )
        Predictor variable values
    ydata : ndarray (n, )
        Response variable values
    p0 : ndarray (n, ), optional
        If None, defaults to 1 for all parameters
    sigma : ndarray (n, ), optional
        Typically uncertainties for each data point. The objective will be multiplied by 1/sigma
    bounds : two-tuple of array-like, optional
        Determines the bounds on the fitting parameters ([lower_bounds], [upper_bounds])
    loss : string, optional, default 'linear'
        Should be one of

            - 'squared': minimizes ``sum(residual**2)``
            - 'absolute': minimizes ``sum(abs(residual))``
            - 'cauchy': minimizes ``sum(f_scale**2 * ln(1 + residual**2/f_scale**2))``

    method : string or 'auto', optional, default 'auto'
        Optimization algorithm to use.\n
        Local optimizers: 

            - 'lbfgs'
            - 'slsqp'
            - 'mma'
            - 'ccsaq'
            - 'tnewton'
            - 'tnewton_restart'
            - 'tnewton_precond'
            - 'tnewton_precond_restart'
            - 'var1'
            - 'var2'
            - 'bobyqa'
            - 'cobyla'
            - 'neldermead'
            - 'sbplx'
            - 'praxis'
        
        Global optimizers require ``bounds!=None``. Possible algorithms:

            - 'crs'
            - 'direct'
            - 'esch'
            - 'isres'
            - 'stogo'
            - 'mlsl'

        If 'auto', defaults to 'slsqp' if ``jac != None`` and 'bobyqa' if ``jac == None``
    jac : callable, optional
        Must be of the form ``jac(xdata)`` and return a N x m numpy array for
        N data points and m fitting parameters

    Returns
    -------
    popt : ndarray (m, )
        Array of best fit parameters.
    pcov : ndarray (m, m)
        Approximated covariance matrix of the fit parameters. 
        To compute one standard deviation errors on the parameters use
        ``perr = np.sqrt(np.diag(pcov))``. In general, these are not very accurate
        estimates of the parameter uncertainty as the method relies on approximations.

    Raises
    ------
    ValueError
        if either `ydata` or `xdata` contain NaNs, or if incompatible options
        are used.
    '''

    #check that bounds are provided for global optimizers
    global_opt_list =['direct', 'mlsl', 'crs', 'stogo', 'isres', 'esch']
    global_opt_list_upper = [x.upper() for x in global_opt_list]

    if method in global_opt_list + global_opt_list_upper:
        if bounds == None:
            raise ValueError("method={} requires bounds.".format(method))

    if loss not in ['squared', 'absolute', 'cauchy']:
        raise ValueError("loss must be one of 'squared', 'absolute', 'cauchy'.")
        
    #convert xdata and ydata to float and make sure that no NaNs are present
    xdata = np.asarray_chkfinite(xdata, float)
    ydata = np.asarray_chkfinite(ydata, float)

    #extract number of parameters
    sig = signature(f)
    num_params = len(sig.parameters) -1
    if p0 is None:
        p0 = np.ones((num_params, ))

    if method == 'auto':
        if callable(jac):
            method = 'slsqp'

        else:
            method = 'bobyqa'

    if bounds:

        lower_bounds = bounds[0]
        upper_bounds = bounds[1]

        minimize_bounds = list(zip(lower_bounds, upper_bounds))

    else:
        minimize_bounds = None

    if callable(jac) and is_gradient_based_global(method):

        if loss == 'squared':

            def objective(p):

                prediction = f(xdata, *p)
                residuals = prediction - ydata
                if sigma is not None:
                    residuals = residuals/sigma
                obj = np.sum(np.square(residuals))

                jac_matrix = jac(xdata, *p)
                gradresiduals = residuals[:, None] * jac_matrix
                grad = 2 * np.sum(gradresiduals, axis=0)
                #print(grad)
                return obj, grad

        elif loss == 'absolute':

            def objective(p):

                prediction = f(xdata, *p)
                residuals = prediction - ydata
                if sigma is not None:
                    residuals = residuals/sigma
                obj = np.sum(np.abs(residuals))

                jac_matrix = jac(xdata, *p)
                signs_residuals = np.sign(residuals)
                gradresiduals = signs_residuals[:, None] * jac_matrix
                grad = np.sum(gradresiduals, axis=0)
            
                return obj, grad

        elif loss == 'cauchy':
            
            def objective(p):

                prediction = f(xdata, *p)
                residuals = prediction - ydata
                if sigma is not None:
                    residuals = residuals/sigma

                fscaled_squared = f_scale * f_scale
                squared_residuals = np.square(residuals)/fscaled_squared
                obj = np.sum(fscaled_squared * np.log1p(squared_residuals))

                jac_matrix = jac(xdata, *p)
                gradresiduals = jac_matrix/(squared_residuals[:, None] +1)*2*residuals[:, None]
                grad = np.sum(fscaled_squared * gradresiduals, axis = 0)

                return obj, grad

        if method in ['mlsl', 'MLSL']:
            res = mlsl(objective, bounds = minimize_bounds, jac = True, **minimize_kwargs)
        elif method in ['stogo', 'STOGO']:
            res = stogo(objective, bounds = minimize_bounds, jac = True, **minimize_kwargs)
        else:
            res = minimize(objective, p0, method = method, bounds = minimize_bounds, jac = True, **minimize_kwargs)
        
    else:

        if loss == 'absolute':

            def objective(p):

                prediction = f(xdata, *p)
                residuals = prediction - ydata
                if sigma is not None:
                    residuals = residuals/sigma
                obj = np.sum(np.abs(residuals))

                return obj

        if loss == 'squared':
            
            def objective(p):

                prediction = f(xdata, *p)
                residuals = prediction - ydata
                if sigma is not None:
                    residuals = residuals/sigma
                obj = np.sum(np.square(residuals))

                return obj

        if loss == 'cauchy':

            fscaled_squared = f_scale * f_scale

            def objective(p):

                prediction = f(xdata, *p)
                residuals = prediction - ydata
                if sigma is not None:
                    residuals = residuals/sigma
                obj = np.sum(fscaled_squared * np.log1p(np.square(residuals)/fscaled_squared))

                #print(obj)
                return obj

        if method in ['mlsl', 'MLSL']:
            res = mlsl(objective, bounds = minimize_bounds, **minimize_kwargs)
        elif method in ['stogo', 'STOGO']:
            res = stogo(objective, bounds = minimize_bounds, **minimize_kwargs)
        elif method in ['crs', 'CRS']:
            res = crs(objective, bounds = minimize_bounds, **minimize_kwargs)
        elif method in ['direct', 'DIRECT']:
            res = direct(objective, bounds = minimize_bounds, **minimize_kwargs)
        elif method in ['isres', 'ISRES']:
            res = isres(objective, bounds = minimize_bounds, **minimize_kwargs)
        elif method in ['esch', 'ESCH']:
            res = esch(objective, bounds = minimize_bounds, **minimize_kwargs)
        else:
            res = minimize(objective, p0, method = method, bounds = minimize_bounds, **minimize_kwargs)

    popt = res.x

    #Covariance matrix estimation borrowed from SciPy
    
    if callable(jac):
        jac_min = jac(xdata, *popt)
    else:
        def wrapped_func(pos, xdata):
            return f(xdata, *pos)

        if bounds is None:
            bounds = (-np.inf, np.inf)

        jac_min = approx_derivative(wrapped_func, popt, method='3-point', 
            bounds = bounds, args=(xdata,))
    
    _, s, VT = svd(jac_min, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac_min.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)
    s_sq = res.fun / (ydata.size - p0.size)
    pcov = pcov * s_sq

    warn_cov = False
    ndatapoints = xdata.size

    if pcov is None:
        # indeterminate covariance
        pcov = np.zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(np.inf)
        warn_cov = True
    
    elif ndatapoints > p0.size:
            s_sq = res.fun / (ndatapoints - p0.size)
            pcov = pcov * s_sq
    else:
        pcov.fill(np.inf)
        warn_cov = True

    if warn_cov:
        warn('Covariance of the parameters could not be estimated!',
                    RuntimeWarning)
    return popt, pcov