from simplenlopt._Core import minimize, is_gradient_based, generate_nlopt_objective, setup_optimizer, normalize_bounds, execute_optimization
import numpy as np

def random_initial_point(lower, upper):
    '''
    Pick one random point from hyperrectangle with uniform probability
    '''
    lower_bounds = np.asarray(lower)
    upper_bounds = np.asarray(upper)

    x0 =  lower_bounds + (upper_bounds - lower_bounds) * np.random.rand(len(lower_bounds))

    return x0

def mlsl(fun, bounds, args=(), jac=None, x0='random', sobol_sampling = True, 
    population=4, local_minimizer='auto', ftol_rel=1e-8,
    xtol_rel=1e-6, ftol_abs = 1e-14, xtol_abs = 1e-8, 
    maxeval='auto', maxtime = None, local_minimizer_options={}):
    '''
    Global optimization via MultiLevel Single Linkage (MLSL)

    .. note::
        MLSL does not seem to respect the relative and absolute convergence criteria.
        By default, it will always run for the maximal number of iterations.

    Parameters
    --------
    fun : callable 
        The objective function to be minimized. Must be in the form ``fun(x, *args)``, where ``x`` is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function
    bounds : tuple of array-like
        Bounds for variables. ``(min, max)`` pairs for each element in ``x``, defining the finite lower and upper bounds for the optimizing argument of ``fun``. It is required to have ``len(bounds) == len(x)``. ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : list, optional, default ()
        Further arguments to describe the objective function
    jac : {callable,  '2-point', '3-point', 'NLOpt', bool}, optional, default None
        If callable, must be in the form ``jac(x, *args)``, where ``x`` is the argument. 
        in the form of a 1-D array and args is a tuple of any additional fixed parameters 
        needed to completely specify the function.\n
        If '2-point' will use forward difference to approximate the gradient.\n
        If '3-point' will use central difference to approximate the gradient.\n
        If 'NLOpt', must be in the form ``jac(x, grad, *args)``, where ``x`` is the argument 
        in the form of a 1-D array, ``grad`` a 1-D array containing the gradient 
        and args is a tuple of any additional fixed parameters needed to completely specify the function.
    x0 : {ndarray, 'random'}, optional, default 'random'
        Initial parameter vector guess.
        If ndarray, must be a 1-D array
        if 'random', picks a random initial guess in the feasible region.
    sobol_sampling : bool, optional, default True
        If True, starting points for local minimizations are sampled from a Sobol sequence.
    population : int, optional, default 4
        Number of local searches per iteration. 
    local_minimizer : string, optional, default 'auto'
        Local Optimization algorithm to use. If string, Should be one of 

            - 'lbfgs': Limited-memory Broyden-Fletcher Goldfarb Shanno algorithm
            - 'slsqp': Sequential least squares programming
            - 'mma': Method of moving asymptotes
            - 'ccsaq': conservative convex separable approximation
            - 'tnewton': truncated Newton
            - 'tnewton_restart': truncated Newton with restarting
            - 'tnewton_precond': truncated Newton with preconditioning
            - 'tnewton_precond_restart': truncated Newton with preconditioning and restarting
            - 'var1': Shifted limited-memory variable metric with rank 1-method
            - 'var2': Shifted limited-memory variable metric with rank 2-method
            - 'bobyqa': Bounded optimization by quadratic approximation
            - 'cobyla': Constrained optimization by linear approximation
            - 'neldermead': Nelder-Mead optimization
            - 'sbplx': Subplex algorithm
            - 'praxis': Principal Axis algorithm
            - 'auto'

        See `NLopt documentation <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_ 
        for a detailed description of these methods.\n
        If 'auto', will default to "lbfgs" if ``jac!= None`` and "bobyqa" if ``jac=None``
    ftol_rel : float, optional, default 1e-8
        Relative function tolerance to signal convergence 
    xtol_rel : float, optional, default 1e-6
        Relative parameter vector tolerance to signal convergence
    ftol_abs : float, optional, default 1e-14
        Absolute function tolerance to signal convergence
    xtol_abs : 1e-8, optional, default 1e-8
        Absolute parameter vector tolerance to signal convergence
    maxeval : {int, 'auto'}, optional, default 'auto'
        Number of maximal function evaluations.
        If 'auto', set to 1.000 * dimensions for gradient based local optimizer,
        and 10.000 * problem dimensional for gradient free local optimizer
    maxtime : float, optional, default None
        maximum absolute time until the optimization is terminated
    local_mimimizer_options : dict
        Further options supplied to the local minimizer

    Returns
    -------
    result : :py:class:`~OptimizeResult`
        The optimization result represented as a :py:class:`~OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination.
        See :py:class:`~OptimizeResult` for a description of other attributes.

    Notes
    -------
    References:\n
    A. H. G. Rinnooy Kan and G. T. Timmer, "Stochastic global optimization methods," 
    Mathematical Programming, vol. 39, p. 27-78 (1987)\n
    Sergei Kucherenko and Yury Sytsko, "Application of deterministic low-discrepancy sequences 
    in global optimization," Computational Optimization and Applications, vol. 30, p. 297-318 (2005)
    '''
    #if local_minimizer not given, choose automatically
    #depending on gradient availability

    if local_minimizer == 'auto':
        if jac:
            local_minimizer = 'lbfgs'
        else:
            local_minimizer = 'bobyqa'

    #check if local minimizer is gradient based
    gradient_required = is_gradient_based(local_minimizer)

    #choose the right version of MLSL based on gradient requirement
    if gradient_required:
        if sobol_sampling:
            mlsl_algorithm = 'GD_MLSL_LDS'
        else:
            mlsl_algorithm = 'GD_MLSL'
    else:
        if sobol_sampling:
            mlsl_algorithm = 'GN_MLSL_LDS'
        else:
            mlsl_algorithm = 'GN_MLSL'
    
    #extract dimension from bounds
    lower, upper = zip(*normalize_bounds(bounds))
    dim = len(lower)

    #set up local optimizer
    local_optimizer = setup_optimizer(local_minimizer, dim)

    #set tolerances
    local_optimizer.set_xtol_rel(xtol_rel)
    local_optimizer.set_ftol_rel(ftol_rel)
    local_optimizer.set_ftol_abs(ftol_abs)
    local_optimizer.set_xtol_abs(xtol_abs)

    #set additional local optimizer options
    for option, val in local_minimizer_options.items():
        try:
            set_option = getattr(local_optimizer, 'set_{option}'.format(option=option))
        except AttributeError:
            raise ValueError('Parameter {option} could not be '
                             'recognized.'.format(option=option))
        else:
            set_option(val)

    #set up global MLSL optimizer
    path = []
    mlsl_optimizer = setup_optimizer(mlsl_algorithm, dim)
    obj_fun = generate_nlopt_objective(fun, jac_required = gradient_required, 
        jac=jac, args = args, path = path)
    mlsl_optimizer.set_min_objective(obj_fun)

    #set local minimizer
    mlsl_optimizer.set_local_optimizer(local_optimizer)

    #set bounds
    mlsl_optimizer.set_lower_bounds(lower)
    mlsl_optimizer.set_upper_bounds(upper)

    #set maximal number of function evaluations
    if maxeval == 'auto':
        if gradient_required:
            maxeval=1000 * dim
        else:
            maxeval=10000 * dim

    mlsl_optimizer.set_maxeval(maxeval)
    
    #set population
    mlsl_optimizer.set_population(population)

    #if given, set maxtime
    if maxtime:
        mlsl_optimizer.set_maxtime(maxtime)

    #if no initial point given, pick randomly within bounds

    if x0 == 'random':
        x0 = random_initial_point(lower, upper)

    result = execute_optimization(mlsl_optimizer, x0, path)

    return result

def stogo(fun, bounds, args=(), jac=None, x0='random', randomize = False, 
    ftol_rel = 1e-8, xtol_rel = 1e-6, ftol_abs = 1e-14, xtol_abs = 1e-8, 
    maxeval='auto', maxtime = None, solver_options={}):
    '''
    Global optimization via STOchastic Global Optimization (STOGO)

    .. note::
        STOGO does not seem to respect the relative and absolute convergence criteria.
        By default, it will always run for the maximal number of iterations.

    Parameters
    --------
    fun : callable 
        The objective function to be minimized. Must be in the form ``fun(x, *args)``, where ``x`` is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function
    bounds : tuple of array-like
        Bounds for variables. ``(min, max)`` pairs for each element in ``x``, defining the finite lower and upper bounds for the optimizing argument of ``fun``. It is required to have ``len(bounds) == len(x)``. ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : list, optional, default ()
        Further arguments to describe the objective function
    jac : {callable,  '2-point', '3-point', 'NLOpt', bool}, optional, default None
        If callable, must be in the form ``jac(x, *args)``, where ``x`` is the argument 
        in the form of a 1-D array and args is a tuple of any additional fixed parameters 
        needed to completely specify the function. \n
        If '2-point' will use forward difference to approximate the gradient.\n
        If '3-point' will use central difference to approximate the gradient.\n
        If 'NLOpt', must be in the form ``jac(x, grad, *args)``, where ``x`` is the argument 
        in the form of a 1-D array, ``grad`` a 1-D array containing the gradient
        and args is a tuple of any additional fixed parameters needed to completely specify the function.
    x0 : {ndarray, 'random'}, optional, default 'random'
        Initial parameter vector guess.\n
        If ndarray, must be a 1-D array.\n
        If 'random', picks a random initial guess in the feasible region.
    randomize: bool, optional, default False
        If True, randomizes the branching process
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

    Returns
    -------
    result : :py:class:`~OptimizeResult`
        The optimization result represented as a :py:class:`~OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination.
        See :py:class:`~OptimizeResult` for a description of other attributes.

    Notes
    -------
    References:\n
    S. Zertchaninov and K. Madsen, "A C++ Programme for Global Optimization,"
    IMM-REP-1998-04, Department of Mathematical Modelling,
    Technical University of Denmark, DK-2800 Lyngby, Denmark, 1998\n
    S. Gudmundsson, "Parallel Global Optimization," M.Sc. Thesis, IMM,
    Technical University of Denmark, 1998
    '''
    if randomize:
        method='stogo_rand'

    else:
        method='stogo'

    if x0 == 'random':
        lower, upper = zip(*normalize_bounds(bounds))
        x0 = random_initial_point(lower, upper)

    if maxeval == 'auto':
        dim = len(x0)
        maxeval = 1000 * dim

    res = minimize(fun, x0, method=method, jac = jac, bounds=bounds,
             ftol_rel = ftol_rel, xtol_rel = xtol_rel, 
             ftol_abs = ftol_abs, xtol_abs = xtol_abs, maxeval=maxeval, 
            maxtime=maxtime, solver_options=solver_options)

    return res

def isres(fun, bounds, args=(), constraints = [], x0='random', population=None, 
    ftol_rel = 1e-8, xtol_rel = 1e-6, ftol_abs = 1e-14, xtol_abs = 1e-8, 
    maxeval='auto', maxtime = None, solver_options={}):    
    '''
    Global optimization via the Improved Stochastic Ranking Evolution Strategy

    .. note::
        ISRES does not seem to respect the relative and absolute convergence criteria.
        By default, it will always run for the maximal number of iterations.

    Parameters
    --------
    fun : callable 
        The objective function to be minimized. Must be in the form ``fun(x, *args)``, where ``x`` is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function
    bounds : tuple of array-like
        Bounds for variables. ``(min, max)`` pairs for each element in ``x``, defining the finite lower and upper bounds for the optimizing argument of ``fun``. It is required to have ``len(bounds) == len(x)``. ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : list, optional, default ()
        Further arguments to describe the objective function
    constraints: list, optional, default ()
        List of constraint functions. Constraints must be of the form ``f(x)`` for a constraint of the form f(x) <= 0.
    x0 : {ndarray, 'random'}, optional, default 'random'
        Initial parameter vector guess.
        If ndarray, must be a 1-D array
        if 'random', picks a random initial guess in the feasible region.
    population : int, optional, default None
        Population size.
        If None, will use NLopt's default population size.
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

    Returns
    -------
    result : :py:class:`~OptimizeResult`
        The optimization result represented as a :py:class:`~OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination.
        See :py:class:`~OptimizeResult` for a description of other attributes.

    Notes
    -------
    References:\n
    Thomas Philip Runarsson and Xin Yao, "Search biases in constrained evolutionary optimization,"
    IEEE Trans. on Systems, Man, and Cybernetics Part C: Applications and Reviews,
    vol. 35 (no. 2), pp. 233-243 (2005)\n
    Thomas P. Runarsson and Xin Yao, "Stochastic ranking for constrained evolutionary optimization," 
    IEEE Trans. Evolutionary Computation, vol. 4 (no. 3), pp. 284-294 (2000)
    '''
    if x0 == 'random':
        lower, upper = zip(*normalize_bounds(bounds))
        x0 = random_initial_point(lower, upper)

    if maxeval == 'auto':
        dim = len(x0)
        maxeval = 10000 * dim

    if population:
        solver_options['population'] = population

    res = minimize(fun, x0, method='isres', jac = None, bounds=bounds,
             constraints = constraints, ftol_rel = ftol_rel, xtol_rel = xtol_rel, 
             ftol_abs = ftol_abs, xtol_abs = xtol_abs, maxeval=maxeval, 
            maxtime=maxtime, solver_options=solver_options)

    return res

def esch(fun, bounds, args=(), x0='random', population=None, 
    ftol_rel = 1e-8, xtol_rel = 1e-6, ftol_abs = 1e-14, xtol_abs = 1e-8, 
    maxeval='auto', maxtime = None, solver_options={}):
    '''
    Global optimization via Differential Evolution variant

    .. note::
        ESCH does not seem to respect the relative and absolute convergence criteria.
        By default, it will always run for the maximal number of iterations.

    Parameters
    --------
    fun : callable 
        The objective function to be minimized. Must be in the form ``fun(x, *args)``, where ``x`` is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function
    bounds : tuple of array-like
        Bounds for variables. ``(min, max)`` pairs for each element in ``x``, defining the finite lower and upper bounds for the optimizing argument of ``fun``. It is required to have ``len(bounds) == len(x)``. ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : list, optional, default ()
        Further arguments to describe the objective function
    x0 : {ndarray, 'random'}, optional, default 'random'
        Initial parameter vector guess.\n
        If ndarray, must be a 1-D array.\n
        if 'random', picks a random initial guess in the feasible region.
    population : int, optional, default None
        Population size.
        If None, will use NLopt's default population size.
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

    Returns
    -------
    result : :py:class:`~OptimizeResult`
        The optimization result represented as a :py:class:`~OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination.
        See :py:class:`~OptimizeResult` for a description of other attributes.

    Notes
    -------
    Reference: C. H. da Silva Santos, "Parallel and Bio-Inspired Computing Applied 
    to Analyze Microwave and Photonic Metamaterial Strucutures," 
    Ph.D. thesis, University of Campinas, (2010)
    '''
    if x0 == 'random':
        lower, upper = zip(*normalize_bounds(bounds))
        x0 = random_initial_point(lower, upper)

    if maxeval == 'auto':
        dim = len(x0)
        maxeval = 10000 * dim

    if population:
        solver_options['population'] = population

    res = minimize(fun, x0, method='esch', jac = None, bounds=bounds,
             ftol_rel = ftol_rel, xtol_rel = xtol_rel, 
             ftol_abs = ftol_abs, xtol_abs = xtol_abs, maxeval=maxeval, 
            maxtime=maxtime, solver_options=solver_options)

    return res

def crs(fun, bounds, args=(), x0='random', population = None, 
    ftol_rel = 1e-8, xtol_rel = 1e-6, ftol_abs = 1e-14, xtol_abs = 1e-8, 
    maxeval=None, maxtime = None, solver_options={}):
    '''
    Global optimization via Controlled Random Search with local mutation

    Parameters
    --------
    fun : callable 
        The objective function to be minimized. Must be in the form ``fun(x, *args)``, where ``x`` is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function
    bounds : tuple of array-like
        Bounds for variables. ``(min, max)`` pairs for each element in ``x``, defining the finite lower and upper bounds for the optimizing argument of ``fun``. It is required to have ``len(bounds) == len(x)``. ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : list, optional, default ()
        Further arguments to describe the objective function
    x0 : {ndarray, 'random'}, optional, default 'random'
        Initial parameter vector guess.
        If ndarray, must be a 1-D array
        if 'random', picks a random initial guess in the feasible region.
    population : int, optional, default None
        Population size.
        If None, will use NLopt's default population size.
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

    Returns
    -------
    result : :py:class:`~OptimizeResult`
        The optimization result represented as a :py:class:`~OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination.
        See :py:class:`~OptimizeResult` for a description of other attributes.

    Notes
    -------
    References:\n
    P. Kaelo and M. M. Ali, "Some variants of the controlled random search algorithm
    for global optimization," J. Optim. Theory Appl. 130 (2), 253-264 (2006)\n
    W. L. Price, "Global optimization by controlled random search," 
    J. Optim. Theory Appl. 40 (3), p. 333-348 (1983)\n
    W. L. Price, "A controlled random search procedure for global optimization," 
    in Towards Global Optimization 2, p. 71-84 edited by L. C. W. Dixon and G. P. Szego 
    (North-Holland Press, Amsterdam, 1978)
    '''
    if x0 == 'random':
        lower, upper = zip(*normalize_bounds(bounds))
        x0 = random_initial_point(lower, upper)

    if population:
        solver_options['population'] = population

    res = minimize(fun, x0, method='crs2_lm', jac = None, bounds=bounds,
             ftol_rel = ftol_rel, xtol_rel = xtol_rel, 
             ftol_abs = ftol_abs, xtol_abs = xtol_abs, maxeval=maxeval, 
            maxtime=maxtime, solver_options=solver_options)

    return res

def direct(fun, bounds, args=(), locally_biased = True, scale = True, 
    randomize = False, original = False, ftol_rel = 1e-8, xtol_rel = 1e-6, 
    ftol_abs = 1e-14, xtol_abs = 1e-8, maxeval=None, maxtime=None, solver_options={}):
    '''
    Global optimization via variants of the DIviding RECTangles (DIRECT) algorithm

    Parameters
    --------
    fun : callable 
        The objective function to be minimized. Must be in the form ``fun(x, *args)``, where ``x`` is the argument in the form of a 1-D array and args is a tuple of any additional fixed parameters needed to completely specify the function
    bounds : tuple of array-like
        Bounds for variables. ``(min, max)`` pairs for each element in ``x``, defining the finite lower and upper bounds for the optimizing argument of ``fun``. It is required to have ``len(bounds) == len(x)``. ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : list, optional, default ()
        Further arguments to describe the objective function
    locally_biased : boolean, optional, default True
        If True, uses the locally biased variant of DIRECT known as DIRECT_L
    scale : boolean, optional, default True
        If True, scales the parameter space to a hypercube of length 1 in all dimensions
    randomize : boolean, optional, default False
        If True, randomize the algorithm by partly randomizing which side of the hyperrectangle is halved
    original : boolean, optional, default False
        If True, applies the original implementation of DIRECT by Jablonsky
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

    Returns
    -------
    result : :py:class:`~OptimizeResult`
        The optimization result represented as a :py:class:`~OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination.
        See :py:class:`~OptimizeResult` for a description of other attributes.

    Notes
    -------
    References:\n
    D. R. Jones, C. D. Perttunen, and B. E. Stuckmann, "Lipschitzian optimization without 
    the lipschitz constant," J. Optimization Theory and Applications, vol. 79, p. 157 (1993)\n
    J. M. Gablonsky and C. T. Kelley, "A locally-biased form of the DIRECT algorithm," 
    J. Global Optimization, vol. 21 (1), p. 27-37 (2001)\n\n
    By default will use the locally biased varian with NLopt code "DIRECT_L". For objective functions
    with many local minima, setting ``locally_biased=False`` which calls the DIRECT algorithm without 
    local bias is advisable.
    '''

    #pick the desired version of DIRECT

    if scale == True:
        if locally_biased == True:
            if randomize == True:
                direct_algorithm = 'GN_DIRECT_L_RAND'
            elif original:
                direct_algorithm = 'GN_ORIG_DIRECT_L'
            else:
                direct_algorithm = 'GN_DIRECT_L'
        else:
            if original:
                direct_algorithm = 'GN_ORIG_DIRECT'
            else:
                direct_algorithm = 'GN_DIRECT'

    else:
        if locally_biased:
            if randomize:
                direct_algorithm = 'GN_DIRECT_L_RAND_NOSCAL'
            else:
                direct_algorithm = 'GN_DIRECT_L_NOSCAL'
        else:
            direct_algorithm = 'GN_DIRECT_NOSCAL'   
    
    lower, upper = zip(*normalize_bounds(bounds))

    if maxeval == None:
        maxeval = 10000 * len(lower)
    
    result = minimize(fun, upper, args=args, method=direct_algorithm, jac = None, bounds=bounds,
             constraints=[], ftol_rel = ftol_rel, xtol_rel = xtol_rel, 
             ftol_abs = ftol_abs, xtol_abs = xtol_abs, maxeval=maxeval, 
            maxtime=maxtime, solver_options=solver_options)

    return result