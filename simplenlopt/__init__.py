from ._helpers import local_optimizers_info, global_optimizers_info
from ._Core import minimize, auglag, OptimizeResult
from ._Global_Optimization import direct, mlsl, crs, isres, esch, stogo
from ._Curve_Fit import curve_fit

__all__ = ["local_optimizers_info",
     "global_optimizers_info",
     "minimize",
     "OptimizeResult",
     "auglag",
     "direct",
     "mlsl",
     "crs",
     "isres",
     "esch",
     "stogo",
     "curve_fit"]
'''
_NLOPT_ALGORITHMS_KEYS = list(filter(partial(search, r'^[GL][ND]_'),
                                    dir(nlopt)))
_NLOPT_ALGORITHMS = {k: getattr(nlopt, k) for k in _NLOPT_ALGORITHMS_KEYS}
_NLOPT_ALG_NAMES = [alg_key.split('_',1)[1] for alg_key in _NLOPT_ALGORITHMS_KEYS]

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
#from .Global_Optimizers import *
'''