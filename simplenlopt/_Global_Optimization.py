from simplenlopt._Core import minimize, is_gradient_based, generate_nlopt_objective, setup_optimizer, normalize_bounds, execute_optimization
import numpy as np

def random_initial_point(lower, upper):

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
    
    Parameters
    --------
    fun: callable function
    bounds: tuple of array-like
    args: list of function arguments
    jac: {callable,  '2-point', '3-point', 'NLOpt', bool}, optional
    x0: {ndarray, 'random'}
    sobol_sampling: bool
    population: int
    local_minimizer: str
    ftol_rel: float
    xtol_rel: float
    maxeval: {int, 'auto'}
    maxtime: float
    local_mimimizer_options: dict

    Returns
    --------
    OptimizeResult
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
    ftol_rel=1e-8, xtol_rel=1e-4, ftol_abs = 1e-12, xtol_abs = 1e-6, 
    maxeval='auto', maxtime = None):
    '''
    Global optimization via STOchastic Global Optimization (STOGO)
    
    Parameters
    --------
    fun: callable function
    bounds: tuple of array-like
    args: list of function arguments
    jac: {callable,  '2-point', '3-point', 'NLOpt', bool}, optional
    x0: {ndarray, 'random'}, optional
    randomize: bool, optional
    ftol_rel: float, optional
    xtol_rel: float, optional
    maxeval: {int, 'auto'}, optional
    maxtime: float, optional
    local_mimimizer_options: dict, optional

    Returns
    --------
    OptimizeResult
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
            maxtime=maxtime)

    return res

def isres(fun, bounds, args=(), constraints = [], x0='random', population=None, 
    ftol_rel=1e-8, xtol_rel=1e-4, ftol_abs = 1e-14, xtol_abs = 1e-7, 
    maxeval='auto', maxtime = None, solver_options={}):
    
    '''
    Global optimization via the Improved Stochastic Ranking Evolution Strategy

    Parameters
    --------
    fun: callable function
    bounds: tuple of array-like
    args: list of function arguments
    constraints: list, optional
    x0: {ndarray, 'random'}, optional
    population: int, optional
    ftol_rel: float, optional
    xtol_rel: float, optional
    ftol_abs: float, optional
    xtol_abs: float, optional
    maxeval: {int, 'auto'}, optional
    maxtime: float, optional
    solver_options: dict, optional

    Returns
    --------
    OptimizeResult
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
    ftol_rel=1e-10, xtol_rel=1e-6, ftol_abs = 1e-14, xtol_abs = 1e-7, 
    maxeval='auto', maxtime = None, solver_options={}):
    '''
    Global optimization via Differential Evolution variant

    Source: C. H. da Silva Santos, "Parallel and Bio-Inspired Computing Applied 
    to Analyze Microwave and Photonic Metamaterial Strucutures," 
    Ph.D. thesis, University of Campinas, (2010)
    Parameters
    --------
    fun: callable function
    bounds: tuple of array-like
    args: list of function arguments
    constraints: list, optional
    x0: {ndarray, 'random'}, optional
    population: int, optional
    ftol_rel: float, optional
    xtol_rel: float, optional
    ftol_abs: float, optional
    xtol_abs: float, optional
    maxeval: {int, 'auto'}, optional
    maxtime: float, optional
    solver_options: dict, optional

    Returns
    --------
    OptimizeResult
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
    ftol_rel=1e-10, xtol_rel=1e-6, ftol_abs = 1e-14, xtol_abs = 1e-7, 
    maxeval=None, maxtime = None, solver_options={}):

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

def direct(fun, bounds, args=(), constraints = (), locally_biased = True, scale = True, 
    randomize = False, original = False, ftol_rel = 1e-8, xtol_rel = 1e-4, 
    ftol_abs = 1e-14, xtol_abs = 1e-7, maxeval=None, maxtime=None, solver_options={}):
    '''
    Global optimization via variants of the 
    DIviding RECTangles (DIRECT) algorithm
    Default variant: DIRECT_L

    Parameters
    --------
    fun: callable function
    bounds: tuple of array-like
    args: list of function arguments
    locally_biased: bool
    scale: bool
    randomize: bool
    original: bool

    Returns
    --------
    OptimizeResult
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
            maxtime=maxtime, solver_options={})

    return result