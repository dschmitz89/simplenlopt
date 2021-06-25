import nlopt
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._numdiff import approx_derivative
from warnings import warn

_NLOPT_ALG_NAMES = ['MLSL', 'MLSL_LDS', 'STOGO', 'STOGO_RAND', 'AGS', 'CRS2_LM', 
    'DIRECT', 'DIRECT_L', 'DIRECT_L_NOSCAL', 'DIRECT_L_RAND',
    'DIRECT_L_RAND_NOSCAL', 'DIRECT_NOSCAL', 'ESCH', 'ISRES', 'MLSL', 'MLSL_LDS',
    'ORIG_DIRECT', 'ORIG_DIRECT_L', 'AUGLAG', 'AUGLAG_EQ', 'CCSAQ', 'LBFGS', 'LBFGS_NOCEDAL',
    'MMA', 'SLSQP', 'TNEWTON', 'TNEWTON_PRECOND', 'TNEWTON_PRECOND_RESTART', 'TNEWTON_RESTART',
    'VAR1', 'VAR2', 'AUGLAG', 'AUGLAG_EQ', 'BOBYQA', 'COBYLA', 'NELDERMEAD', 'NEWUOA',
    'NEWUOA_BOUND', 'PRAXIS', 'SBPLX']

_NLOPT_ALGORITHMS = {'GD_MLSL': 21, 'GD_MLSL_LDS': 23,  'GD_STOGO': 8, 'GD_STOGO_RAND': 9,
 'GN_AGS': 43, 'GN_CRS2_LM': 19, 'GN_DIRECT': 0, 'GN_DIRECT_L': 1, 'GN_DIRECT_L_NOSCAL': 4,
 'GN_DIRECT_L_RAND': 2, 'GN_DIRECT_L_RAND_NOSCAL': 5, 'GN_DIRECT_NOSCAL': 3, 'GN_ESCH': 42,
 'GN_ISRES': 35, 'GN_MLSL': 20, 'GN_MLSL_LDS': 22, 'GN_ORIG_DIRECT': 6, 'GN_ORIG_DIRECT_L': 7,
 'LD_AUGLAG': 31, 'LD_AUGLAG_EQ': 33, 'LD_CCSAQ': 41, 'LD_LBFGS': 11, 'LD_LBFGS_NOCEDAL': 10,
 'LD_MMA': 24, 'LD_SLSQP': 40, 'LD_TNEWTON': 15, 'LD_TNEWTON_PRECOND': 17, 'LD_TNEWTON_PRECOND_RESTART': 18,
 'LD_TNEWTON_RESTART': 16, 'LD_VAR1': 13, 'LD_VAR2': 14, 'LN_AUGLAG': 30, 'LN_AUGLAG_EQ': 32,
 'LN_BOBYQA': 34, 'LN_COBYLA': 25, 'LN_NELDERMEAD': 28, 'LN_NEWUOA': 26, 'LN_NEWUOA_BOUND': 27,
 'LN_PRAXIS': 12, 'LN_SBPLX': 29}

_NLOPT_ALGORITHMS_KEYS = ['GD_MLSL', 'GD_MLSL_LDS', 'GD_STOGO', 'GD_STOGO_RAND', 'GN_AGS',
 'GN_CRS2_LM', 'GN_DIRECT', 'GN_DIRECT_L', 'GN_DIRECT_L_NOSCAL', 'GN_DIRECT_L_RAND',
  'GN_DIRECT_L_RAND_NOSCAL', 'GN_DIRECT_NOSCAL', 'GN_ESCH', 'GN_ISRES', 'GN_MLSL',
   'GN_MLSL_LDS', 'GN_ORIG_DIRECT', 'GN_ORIG_DIRECT_L', 'LD_AUGLAG', 'LD_AUGLAG_EQ',
    'LD_CCSAQ', 'LD_LBFGS', 'LD_LBFGS_NOCEDAL', 'LD_MMA', 'LD_SLSQP', 'LD_TNEWTON',
     'LD_TNEWTON_PRECOND', 'LD_TNEWTON_PRECOND_RESTART', 'LD_TNEWTON_RESTART', 'LD_VAR1',
      'LD_VAR2', 'LN_AUGLAG', 'LN_AUGLAG_EQ', 'LN_BOBYQA', 'LN_COBYLA', 'LN_NELDERMEAD',
       'LN_NEWUOA', 'LN_NEWUOA_BOUND', 'LN_PRAXIS', 'LN_SBPLX']

NLOPT_MESSAGES = {
    nlopt.SUCCESS: 'Success',
    nlopt.STOPVAL_REACHED: 'Optimization stopped because stopval (above) '
                           'was reached.',
    nlopt.FTOL_REACHED: 'Optimization stopped because ftol_rel or ftol_abs '
                        '(above) was reached.',
    nlopt.XTOL_REACHED: 'Optimization stopped because xtol_rel or xtol_abs '
                        '(above) was reached.',
    nlopt.MAXEVAL_REACHED: 'Optimization stopped because maxeval (above) '
                           'was reached.',
    nlopt.MAXTIME_REACHED: 'Optimization stopped because maxtime (above) '
                           'was reached.',
    nlopt.FAILURE: 'Failure',
    nlopt.INVALID_ARGS: 'Invalid arguments (e.g. lower bounds are bigger '
                        'than upper bounds, an unknown algorithm was '
                        'specified, etcetera).',
    nlopt.OUT_OF_MEMORY: 'Ran out of memory.',
    nlopt.ROUNDOFF_LIMITED: 'Halted because roundoff errors limited progress. '
                            '(In this case, the optimization still typically '
                            'returns a useful result.)',
    nlopt.FORCED_STOP: "Halted because of a forced termination: the user "
                       "called nlopt_force_stop(opt) on the optimization's "
                       "nlopt_opt object opt from the user's objective "
                       "function or constraints."
}


def get_nlopt_enum(method_name=None, default=nlopt.LN_BOBYQA):
    """
    Get NLOpt algorithm object by name. If the algorithm is not found,
    defaults to `nlopt.LN_BOBYQA`.

    Notes
    -----

    From http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms#Nomenclature:

        Each algorithm in NLopt is identified by a named constant, which
        is passed to the NLopt routines in the various languages in
        order to select a particular algorithm. These constants are
        mostly of the form `NLOPT_{G,L}{N,D}_xxxx`, where G/L denotes
        global/local optimization and N/D denotes derivative-free/
        gradient-based algorithms, respectively.

        For example, the NLOPT_LN_COBYLA constant refers to the COBYLA
        algorithm (described below), which is a local (L)
        derivative-free (N) optimization algorithm.

        Two exceptions are the MLSL and augmented Lagrangian algorithms,
        denoted by NLOPT_G_MLSL and NLOPT_AUGLAG, since whether or not
        they use derivatives (and whether or not they are global, in
        AUGLAG's case) is determined by what subsidiary optimization
        algorithm is specified.

    Equivalent to::

        partial(NLOPT_ALGORITHMS.get, default=nlopt.LN_BOBYQA)

    Examples
    --------
    >>> get_nlopt_enum('LN_NELDERMEAD') == nlopt.LN_NELDERMEAD
    True

    >>> get_nlopt_enum('ln_neldermead') == nlopt.LN_NELDERMEAD
    True

    One is permitted to be cavalier with these method names.

    >>> get_nlopt_enum('ln_NelderMead') == nlopt.LN_NELDERMEAD
    True

    >>> get_nlopt_enum() == nlopt.LN_BOBYQA
    True

    >>> get_nlopt_enum('foobar') == nlopt.LN_BOBYQA
    True

    .. todo:: Exceptional cases (low-priority)

    >>> get_nlopt_enum('G_MLSL') == nlopt.G_MLSL # doctest: +SKIP
    True

    >>> get_nlopt_enum('AUGLAG') == nlopt.AUGLAG # doctest: +SKIP
    True
    """
    if method_name is None:
        method_name = 'LN_BOBYQA'

    try:
        return _NLOPT_ALGORITHMS[method_name.upper()]
    except KeyError:
        pass

    try:
        algorithm_index = _NLOPT_ALG_NAMES.index(method_name.upper())
        full_algorithm_name = _NLOPT_ALGORITHMS_KEYS[algorithm_index]
        return _NLOPT_ALGORITHMS[full_algorithm_name]

    except KeyError:
        warn('Method {name} could not be found. Defaulting to '
             '{default}'.format(name=method_name, default=default),
             RuntimeWarning)
        return default

def setup_optimizer(method, dim):

    if isinstance(method, str):
        method = get_nlopt_enum(method)

    opt = nlopt.opt(method, dim)

    return opt

def set_constraints(constraints, optimizer):

    for constr in constraints:

        fun = generate_nlopt_objective(fun=constr['fun'],
                             jac=constr.get('jac', False),
                             args=constr.get('args', ()))

        constr_tol = constr.get('tol')
        if not constr_tol:
            constr_tol = 1e-8
        if constr['type'] == 'eq':
            optimizer.add_equality_constraint(fun, constr_tol)
        elif constr['type'] == 'ineq':
            optimizer.add_inequality_constraint(fun, constr_tol)
        elif constr['type'] in ('eq_m', 'ineq_m'):
            # TODO: Define '_m' as suffix for now.
            # TODO: Add support for vector/matrix-valued constraints
            raise NotImplementedError('Vector-valued constraints currently '
                                      'not supported.')
        else:
            raise ValueError('Constraint type not recognized')

def generate_nlopt_objective(fun, jac_required = None, jac = None, args=(), bounds = None, path = None):

    assert (callable(fun)), "fun must be a callable!"

    #default to central difference approx. if required
    if jac_required and jac == None:

        jac = '3-point'
        warn('Using gradient-based optimization'
            ', but no gradient information is '
            'available. Gradient will be approximated '
            'by central difference. Consider using a '
            'derivative-free optimizer or supplying '
            'gradient information.', RuntimeWarning)

    if jac_required:

        #objective and grad provided by same function
        if isinstance(jac, bool) and jac == True:
            #print("Combined")#if jac == True:
            def objective(x, grad):

                if path is not None:
                    path.append(x.copy())

                if grad.size > 0:
                    val, grad[:] = fun(x, *args)

                else:
                    val, grad_temp = fun(x, *args)

                return val

        #finite difference approximation
        elif jac in ['2-point', '3-point']:

            if bounds:
                lower_bounds = [bound[0] for bound in bounds]
                upper_bounds = [bound[1] for bound in bounds]

                numdiff_bounds = (lower_bounds, upper_bounds)

            else:
                numdiff_bounds = (-np.inf, np.inf)

            def objective(x, grad):

                if path is not None:
                    path.append(x.copy())

                val = fun(x, *args)
                
                if grad.size > 0:
                    grad[:] = approx_derivative(fun, x, method = jac, f0 = val, 
                    bounds = numdiff_bounds, args=args)
                return val        

        #function already in Nlopt style
        elif jac == 'nlopt':

            def objective(x, grad):

                if path is not None:
                    path.append(x.copy())

                return fun(x, grad)

        #separate callable for grad
        elif callable(jac):

            def objective(x, grad):

                if path is not None:
                    path.append(x.copy())

                val = fun(x, *args)
                if grad.size > 0:
                    grad[:] = jac(x, *args)
                return val            

        else:
            raise ValueError("Unknown method for Jacobian: jac='%s'. Should be one of "
            " None, True, '2-point', '3-point', 'nlopt' or a callable. " % jac )
    
    #no gradient information required
    else:

        def objective(x, grad):

            if path is not None:
                path.append(x.copy())

            return fun(x, *args)

    return objective

def normalize_bound(bound):
    """
    Examples
    --------
    >>> normalize_bound((2.6, 7.2))
    (2.6, 7.2)

    >>> normalize_bound((None, 7.2))
    (-inf, 7.2)

    >>> normalize_bound((2.6, None))
    (2.6, inf)

    >>> normalize_bound((None, None))
    (-inf, inf)

    This operation is idempotent:

    >>> normalize_bound((-float("inf"), float("inf")))
    (-inf, inf)
    """
    min_, max_ = bound

    if min_ is None:
        min_ = -float('inf')

    if max_ is None:
        max_ = float('inf')

    return min_, max_


def normalize_bounds(bounds=[]):
    """
    Examples
    --------
    >>> bounds = [(2.6, 7.2), (None, 2), (3.14, None), (None, None)]
    >>> list(normalize_bounds(bounds))
    [(2.6, 7.2), (-inf, 2), (3.14, inf), (-inf, inf)]
    """
    return map(normalize_bound, bounds)


def get_nlopt_message(ret_code):
    """
    Notes
    -----
    Identical to ``NLOPT_MESSAGES.get``

    Examples
    --------
    >>> get_nlopt_message(nlopt.SUCCESS)
    'Success'
    >>> get_nlopt_message(nlopt.INVALID_ARGS)
    'Invalid arguments (e.g. lower bounds are bigger than upper bounds, an unknown algorithm was specified, etcetera).'
    """
    return NLOPT_MESSAGES.get(ret_code)

def execute_optimization(optimizer, x0, path):

    try:
        x = optimizer.optimize(x0)

    except nlopt.RoundoffLimited:
        # If we encounter a RoundoffLimited exception, simply return last point
        x = path[-1]

    return OptimizeResult(
        x=x,
        fun=optimizer.last_optimum_value(),
        nfev=optimizer.get_numevals(),
        message=get_nlopt_message(optimizer.last_optimize_result()),
        success=(optimizer.last_optimize_result() > 0),
    )

def equality_constraints_set(constraints):

    for constr in constraints:
        if constr['type'] == 'eq':
            break
            return True

def random_initial_point(lower, upper):

    lower_bounds = np.asarray(lower)
    upper_bounds = np.asarray(upper)

    x0 =  lower_bounds + (upper_bounds - lower_bounds) * np.random.rand(len(lower_bounds))

    return x0

def is_gradient_based(method_name):
    '''
    Checks if solver uses derivatives based on NLOPT code
    '''
    if not method_name[2] == '_':

        algorithm_index = _NLOPT_ALG_NAMES.index(method_name.upper())
        method_name = _NLOPT_ALGORITHMS_KEYS[algorithm_index]

    method_name = method_name.upper()
    
    if method_name[1] == 'D':
        return True
    else:
        return None

def is_global(method_name):
    '''
    Checks if solver is a global optimizer based on NLOPT code
    '''
    if not method_name[2] == '_':

        algorithm_index = _NLOPT_ALG_NAMES.index(method_name.upper())
        method_name = _NLOPT_ALGORITHMS_KEYS[algorithm_index]

    method_name = method_name.upper()
    
    if method_name[0] == 'G':
        return True
    else:
        return None

def minimize(fun, x0, args=(), method='auto', jac=None, bounds=None,
             constraints=[], ftol_rel = 1e-8, xtol_rel = 1e-6, 
             ftol_abs = 1e-14, xtol_abs = 1e-8, maxeval=None, 
            maxtime=None, solver_options={}):
    """
    Local minimization function for NLopt's algorithm in SciPy style

    Parameters
    ----------
    fun : callable 
        The objective function to be minimized. Must be in the form ``fun(x, *args)``, where ``x`` is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function
    x0 : ndarray
        Starting guess for the decision variable
    args : tuple, optional, default ()
        Further arguments to describe the objective function
    method : string or 'auto', optional, default 'auto'
        Optimization algorithm to use. If string, Should be one of 

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
            - 'newuoa_bound'
            - 'newuoa'

        If 'auto', will be set to 'bobyqa'/'cobyla' if jac=None ('cobyla' if constraints are set) 
        or 'lbfgs'/'slsqp' if jac != None ('slsqp' if constraints set)

        If the chosen method does not support the required constraints, the augmented lagrangian 
        is called to handle them.
    bounds : tuple of array-like
        Bounds for variables. ``(min, max)`` pairs for each element in ``x``,
        defining the finite lower and upper bounds for the optimizing argument of ``fun``. 
        It is required to have ``len(bounds) == len(x)``.
    constraints: list, optional, default ()
        List of constraint functions. Constraints must be of the form ``f(x)`` for a constraint of the form f(x) <= 0.
    ftol_rel : float, optional, default 1e-8
        Relative function tolerance to signal convergence 
    xtol_rel : float, optional, default 1e-6
        Relative parameter vector tolerance to signal convergence
    ftol_abs : float, optional, default 1e-14
        Absolute function tolerance to signal convergence
    xtol_abs : float, optional, default 1e-8
        Absolute parameter vector tolerance to signal convergence
    maxeval : {int, 'auto'}, optional, default 'auto'
        Number of maximal function evaluations.
        If 'auto', set to 1.000 * dimensions
    maxtime : float, optional, default None
        maximum absolute time until the optimization is terminated.
    solver_options: dict, optional, default None
        Dictionary of additional options supplied to the solver.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import rosen, rosen_der
    >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    >>> res = minimize(rosen, x0, method='lbfgs', jac=rosen_der)
    >>> res.success
    True
    >>> res.message
    'Success'
    >>> np.isclose(res.fun, 0)
    True
    >>> res.x
    array([ 1.,  1.,  1.,  1.,  1.])

    >>> res = minimize(rosen, x0, method='lbfgs', jac=rosen_der,
    ...                ftol_abs=1e-5)
    >>> res.success
    True
    >>> res.message
    'Optimization stopped because ftol_rel or ftol_abs (above) was reached.'

    >>> res = minimize(rosen, x0, method='lbfgs', jac=rosen_der, foo=3)
    Traceback (most recent call last):
        ...
    ValueError: Parameter foo could not be recognized.

    .. todo:: Some sensible way of testing this.

    >>> x0 = np.array([-1., 1.])
    >>> fun = lambda x: - 2*x[0]*x[1] - 2*x[0] + x[0]**2 + 2*x[1]**2
    >>> dfun = lambda x: np.array([2*x[0] - 2*x[1] - 2, - 2*x[0] + 4*x[1]])
    >>> cons = [{'type': 'eq',
    ...           'fun': lambda x: x[0]**3 - x[1],
    ...           'jac': lambda x: np.array([3.*(x[0]**2.), -1.])},
    ...         {'type': 'ineq',
    ...           'fun': lambda x: x[1] - 1,
    ...           'jac': lambda x: np.array([0., 1.])}]
    >>> res = minimize(fun, x0, jac=dfun, method='slsqp', constraints=cons)
    >>> res.success
    False
    >>> res.message
    'Halted because roundoff errors limited progress. (In this case, the optimization still typically returns a useful result.)'
    >>> res.x.round(2)
    array([ 0.84,  0.6 ])

    >>> cons = [{'type': 'some bogus type',
    ...           'fun': lambda x: x[0]**3 - x[1],
    ...           'jac': lambda x: np.array([3.*(x[0]**2.), -1.])},
    ...         {'type': 'ineq',
    ...           'fun': lambda x: x[1] - 1,
    ...           'jac': lambda x: np.array([0., 1.])}]
    >>> res = minimize(fun, x0, jac=dfun, method='slsqp', constraints=cons, ftol_abs=1e-20)
    Traceback (most recent call last):
        ...
    ValueError: Constraint type not recognized
    """

    #if no method set, choose automatically
    if len(constraints) > 0:
        constraints_set = True
    else:
        constraints_set = None

    #if constraints and method set, check if chosen method 
    #supports them. If not switch to auglag

    if constraints_set and not method == 'auto':

        gradient_required = is_gradient_based(method)
        if gradient_required: 
            recommendation = 'SLSQP'
        else:
            recommendation = 'COBYLA'

        gradient_required = is_gradient_based(method)
        constraint_supporting_methods = ['slsqp', 'ccsaq', 'mma', 'cobyla',
            'ld_slsqp', 'ld_ccsaq', 'ld_mma', 'ln_cobyla']
        constraint_supporting_methods_upper = [name.upper() for name in constraint_supporting_methods]
        if method not in constraint_supporting_methods + constraint_supporting_methods_upper:
            warn("Method {} does not support constraints. "
            "Constraints will be handled by augmented lagrangian. "
            "In case of problems consider method='{}'.".format(method, recommendation), RuntimeWarning)

            return auglag(fun, x0, args, method, jac, bounds,
                constraints, True, ftol_rel, xtol_rel, ftol_abs, 
                xtol_abs, maxeval, maxtime, solver_options)
        
        #MMA and CCSAQ only support inequality constraints
        #Switch to auglag_eq if required
        equalities_required = equality_constraints_set(constraints)
        inequalities_supporting_methods = ['ccsaq', 'mma', 'ld_ccsaq', 'ld_mma']
        inequalities_supporting_methods_upper = [name.upper() for name in inequalities_supporting_methods]
        if equalities_required and method in inequalities_supporting_methods + inequalities_supporting_methods_upper:
            warn("Method {} does not support equality constraints. "
            "Equality constraints will be handled by augmented lagrangian. "
            "In case of problems consider method='{}'.".format(method, recommendation), RuntimeWarning)

            return auglag(fun, x0, args, method, jac, bounds,
                constraints, False, ftol_rel, xtol_rel, ftol_abs, 
                xtol_abs, maxeval, maxtime, solver_options)

    if method == 'auto':
        if jac:
            if constraints_set:
                method = 'slsqp'                         
            else:
                method = 'lbfgs'
        else:
            if constraints_set:
                method = 'cobyla'
            else:
                method = 'bobyqa'

    #setup optimizer
    dim = len(x0)
    opt = setup_optimizer(method, dim)

    # Initialize path
    path = []

    #check if method is gradient based
    gradient_required = is_gradient_based(method)

    # Create NLOpt objective function
    obj_fun = generate_nlopt_objective(fun, gradient_required, jac, args, bounds, path)
    opt.set_min_objective(obj_fun)

    # Normalize and set parameter bounds
    if bounds:
        lower, upper = zip(*normalize_bounds(bounds))
        opt.set_lower_bounds(lower)
        opt.set_upper_bounds(upper)

    # Equality and Inequality Constraints
    set_constraints(constraints, opt)

    #set tolerances
    opt.set_xtol_rel(xtol_rel)
    opt.set_ftol_rel(ftol_rel)
    opt.set_xtol_abs(xtol_abs)
    opt.set_ftol_abs(ftol_abs)

    if maxeval:
        opt.set_maxeval(maxeval)

    if maxtime:
        opt.set_maxtime(maxtime)

    # Set solver specific options
    for option, val in solver_options.items():
        try:
            set_option = getattr(opt, 'set_{option}'.format(option=option))
        except AttributeError:
            raise ValueError('Parameter {option} could not be '
                             'recognized.'.format(option=option))
        else:
            set_option(val)

    # Perform the optimization
    res = execute_optimization(opt, x0, path)

    return res

def auglag(fun, x0, args=(), method='auto', jac=None, bounds = None, 
    constraints = (), penalize_inequalities = True, ftol_rel = 1e-8, 
    xtol_rel = 1e-6, ftol_abs = 1e-14, xtol_abs = 1e-8, maxeval=None, 
    maxtime=None, solver_options={}):
    """
    Constrained local minimization via the augmented lagrangian method
    
    References:

    Andrew R. Conn, Nicholas I. M. Gould, and Philippe L. Toint, 
    "A globally convergent augmented Lagrangian algorithm for optimization 
    with general constraints and simple bounds," SIAM J. Numer. Anal. vol. 28, no. 2, p. 545-572 (1991)

    E. G. Birgin and J. M. Martínez, "Improving ultimate convergence of an augmented Lagrangian method," 
    Optimization Methods and Software vol. 23, no. 2, p. 177-195 (2008)

    Parameters
    ----------
    fun : callable 
        The objective function to be minimized. Must be in the form ``fun(x, *args)``, where ``x`` is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function
    x0 : ndarray
        Starting guess for the decision variable
    args : tuple, optional, default ()
        Further arguments to describe the objective function
    method : string or 'auto', optional, default 'auto'
        Optimization algorithm to use. If string, should be one of 

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
            - 'newuoa_bound'
            - 'newuoa'

        If 'auto', a suitable solver is chosen based on the availability
        of gradient information and if also inequalities should be penalized:

        jac != None and penalize_inequalities=True -> 'lbfgs'

        jac = None and penalize_inequalities=True -> 'bobyqa'

        jac != None and penalize_inequalities=False -> 'mma'

        jac = None and penalize_inequalities=False -> 'cobyla'
    bounds : tuple of array-like, optional, default None
        Bounds for variables. ``(min, max)`` pairs for each element in ``x``,
        defining the finite lower and upper bounds for the optimizing argument of ``fun``. 
        It is required to have ``len(bounds) == len(x)``.
    constraints: list, optional, default ()
        List of constraint functions. Constraints must be of the form ``f(x)`` for a constraint of the form f(x) <= 0.
    penalize_inequalities : bool, optional, default True
        If True, also penalizes violation of inequality constraints (NLopt code: AUGLAG).
        If False, only penalizes violation of equality constraints (NLopt code: AUGLAG_EQ). 
        In this case the chosen method must be able to handle inequality constraints.
    ftol_rel : float, optional, default 1e-8
        Relative function tolerance to signal convergence 
    xtol_rel : float, optional, default 1e-6
        Relative parameter vector tolerance to signal convergence
    ftol_abs : float, optional, default 1e-14
        Absolute function tolerance to signal convergence
    xtol_abs : float, optional, default 1e-8
        Absolute parameter vector tolerance to signal convergence
    maxeval : {int, 'auto'}, optional, default 'auto'
        Number of maximal function evaluations.
        If 'auto', set to 1.000 * dimensions
    maxtime : float, optional, default None
        maximum absolute time until the optimization is terminated.
    solver_options: dict, optional, default None
        Dictionary of additional options supplied to the solver.
    """

    #choose version of augmented lagrangian
    #if not set by user, choose local minimizer automatically
    # depending on gradient availability
    if method == 'auto':
        if jac:
            if penalize_inequalities:
                method = 'lbfgs'
            else:
                method = 'mma'
        else:
            if penalize_inequalities:
                method = 'bobyqa'
            else:
                method = 'cobyla'
    
    #check if local minimizer is gradient based
    gradient_required = is_gradient_based(method)    
    

    if penalize_inequalities:
        if gradient_required:
            auglag_version = 'LD_AUGLAG'
        else:
            auglag_version = 'LN_AUGLAG'

    else:
        if gradient_required:
            auglag_version = 'LD_AUGLAG_EQ'
        else:
            auglag_version = 'LN_AUGLAG_EQ'

    #extract dimension from starting point
    dim = len(x0)

    #set up local optimizer
    local_optimizer = setup_optimizer(method, dim)

    #set tolerances
    local_optimizer.set_xtol_rel(xtol_rel)
    local_optimizer.set_ftol_rel(ftol_rel)
    local_optimizer.set_xtol_abs(xtol_abs)
    local_optimizer.set_ftol_rel(ftol_abs)

    #set additional local optimizer options
    for option, val in solver_options.items():
        try:
            set_option = getattr(local_optimizer, 'set_{option}'.format(option=option))
        except AttributeError:
            raise ValueError('Parameter {option} could not be '
                             'recognized.'.format(option=option))
        else:
            set_option(val)

    #set up augmented lagrangian optimizer

    path = []
    auglag_optimizer = setup_optimizer(auglag_version, dim)
    obj_fun = generate_nlopt_objective(fun, gradient_required, jac, args, bounds, path)
    auglag_optimizer.set_min_objective(obj_fun)
    auglag_optimizer.set_local_optimizer(local_optimizer)

    if bounds:
        lower, upper = zip(*normalize_bounds(bounds))
        auglag_optimizer.set_lower_bounds(lower)
        auglag_optimizer.set_upper_bounds(upper)

    # Equality and Inequality Constraints
    set_constraints(constraints, auglag_optimizer)

    #set maximal number of function evaluations
    if maxeval:
        auglag_optimizer.set_maxeval(maxeval)

    #if given, set maxtime
    if maxtime:
        auglag_optimizer.set_maxtime(maxtime)

    result = execute_optimization(auglag_optimizer, x0, path)

    return result

'''
import matplotlib.pyplot as plt

def func(x, a, b, c, d, e, f):

    return a * np.exp(-b * x) + c + d * np.exp(-(e-x)**2/f**2)

def jac_func(x, a, b, c, d, e, f):

    da = np.exp(-b * x)
    db = -x * a * np.exp(-b * x)
    dc = np.full_like(x, 1.).ravel()
    dd = np.exp(-(e-x)**2/f**2)
    de = -2 * (e-x)/f**2 * d * np.exp(-(e-x)**2/f**2)
    df =  2 * (e-x)**2/f**3 * d * np.exp(-(e-x)**2/f**2)
    return np.array([da.ravel(), db.ravel(), dc, dd.ravel(), de.ravel(), df.ravel()])

#print(jac_func(np.array([2., 2.5]), 1.5, 1.5, 0.5))
#print(func(np.array([2., 2.5]), 1.5, 1.5, 2))

xdata = np.linspace(0, 4,50)
y = func(xdata, 2., 0.8, 2., 1.5, 3., 0.5)
#np.random.seed(1729)
y_noise = 0.1 * np.random.normal(size=xdata.size)

ydata = y + y_noise

p0 = np.array([1., 1., 1., 1., 1., 1.])
params, f, nfev = curve_fit(func, xdata, ydata, p0, method='neldermead', sigma = np.ones_like(xdata), loss = 'squared')#p0 = np.array([1., 2., 10.])
print("neldermead: p={}, nfev={}".format(params, nfev))


params, f, nfev = curve_fit(func, xdata, ydata, p0, method='mma', sigma = np.ones_like(xdata), jac = jac_func, loss = 'squared')#p0 = np.array([1., 2., 10.])
print("mma: p={}, nfev={}".format(params, nfev))
#params = curve_fit(func, xdata, ydata, method='neldermead', sigma = np.ones_like(xdata), jac=jac_func, loss = 'squared')#p0 = np.array([1., 2., 10.])
#print("Exact: ", params)
    
plt.scatter(xdata, ydata)
plt.plot(xdata, func(xdata, *params))
plt.show()
'''