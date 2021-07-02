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
        loss = 'linear', method = 'auto', jac = None, **minimize_kwargs):
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

            - 'linear' (yields squared residuals)
            - 'absolute' (minimizes absolute residuals)

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

        if loss == 'linear':

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

        if loss == 'absolute':

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

        if loss == 'linear':
            
            def objective(p):

                prediction = f(xdata, *p)
                residuals = prediction - ydata
                if sigma is not None:
                    residuals = residuals/sigma
                obj = np.sum(np.square(residuals))

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

'''
#@njit
def func(x, a, b, c):

    return a * np.exp(-b * x) + c 

#@njit
def jac_func(x, a, b, c):

    da = np.exp(-b * x)
    db = -x * a * np.exp(-b * x)
    dc = np.full_like(x, 1.).ravel()
    return np.stack((da.ravel(), db.ravel(), dc), axis = 1)


xdata = np.linspace(0, 4,50)
y = func(xdata, 2., 0.8, 2.)
jac = jac_func(xdata, 2., 0.8, 2.)
#np.random.seed(1729)
y_noise = 0.1 * np.random.normal(size=xdata.size)

ydata = y + y_noise

prediction_array = np.empty_like(xdata)
grad_array = np.empty((xdata.size, 3))

def func_fast(x, a, b, c):
    prediction_array[:] = a * np.exp(-b * x) + c
    return prediction_array

def jac_fast(x, a, b, c):

    grad_array[:, 0] = np.exp(-b * x)
    grad_array[:, 1] = -x * a * grad_array[:, 0]#np.exp(-b * x)
    grad_array[:, 2] = np.ones_like(x)

    return grad_array

p0 = np.array([1., 1., 1.])
#print(jac_0.shape)

t0 = time()
#for i in range(100):

testbounds = ([1., 0.9, 1.], [1.9, 1., 3])
params, pcov = curve_fit(func_fast, xdata, ydata, p0, method='slsqp', jac=jac_fast, bounds = testbounds, loss = 'squared')
print(time() - t0)
print(params)
print(pcov)

t0 = time()
#for i in range(100):

params, pcov = curve_fit(func, xdata, ydata, p0, method='slsqp', bounds = testbounds, loss = 'squared')
print(time() - t0)
print(params)
print(pcov)

#print("SciPy")
from scipy.optimize import curve_fit as sc_curve_fit
from scipy.optimize import least_squares

def ls_obj(x):

    return func(xdata, x[0], x[1], x[2]) -y

def jac_func_scipy(x):

    da = np.exp(-x[1] * xdata)
    db = -xdata * x[0] * np.exp(-x[1] * xdata)
    dc = np.full_like(xdata, 1.).ravel()
    return np.array([da.ravel(), db.ravel(), dc]).T

#ls_res = least_squares(ls_obj, p0, jac = jac_func_scipy)
#print("Least Squares jac: ", ls_res.jac)

#print("Scipy curve fit")
params, pcov = sc_curve_fit(func, xdata, ydata, p0,jac = jac_func)
print(params)
print(pcov)
'''