def local_optimizers_info():
    '''
    Prints a short summary of NLopt's local optimizers
    '''

    LOCAL_OPTIMIZERS=['LBFGS'.ljust(23) + ': Gradient based. Supports: Bounds.',
    'SLSQP'.ljust(23) + ': Gradient based. Supports: Bounds, Inequality Constraints, Equality Constraints.',
    'CCSAQ'.ljust(23) + ': Gradient based. Supports: Bounds, Inequality Constraints.',
    'MMA'.ljust(23) + ': Gradient based. Supports: Bounds, Inequality Constraints.',
    'TNEWTON_PRECOND_RESTART'.ljust(23) + ': Gradient based.',
    'VAR1'.ljust(23) + ': Gradient based. Supports: Bounds.',
    'VAR2'.ljust(23) + ': Gradient based. Supports: Bounds.',
    'TNEWTON'.ljust(23) + ': Gradient based. ',
    'TNEWTON_PRECOND'.ljust(23) + ': Gradient based.',
    'TNEWTON_RESTART'.ljust(23) + ': Gradient based.',

    "",
    'BOBYQA'.ljust(23) + ': Gradient free.'.ljust(18) + 'Supports: Bounds.',
    'COBYLA'.ljust(23) + ': Gradient free.'.ljust(18) + 'Supports: Bounds, Inequality Constraints, Equality Constraints.',
    'NELDERMEAD'.ljust(23) + ': Gradient free.'.ljust(18) + 'Supports: Bounds.',
    'SBPLX'.ljust(23) + ': Gradient free.'.ljust(18) + 'Supports: Bounds.',
    'PRAXIS'.ljust(23) + ': Gradient free.'.ljust(18) + 'Supports: Bounds.']

    print("NLOpt's local optimizers which can be passed as method to minimize and auglag")
    print("")
    print(*LOCAL_OPTIMIZERS, sep='\n')

def global_optimizers_info():
    '''
    Prints a short summary of NLopt's global optimizers
    '''
    
    GLOBAL_OPTIMIZERS=['DIRECT'.ljust(6) + ': Gradient free.'.ljust(18),
    'CRS'.ljust(6) + ': Gradient free.'.ljust(18),
    "",
    'MLSL'.ljust(6) + ': Gradient based depending on local minimizer. Warning: by default takes maxeval function evaluations.',
    'ISRES'.ljust(6) + ': Gradient free.'.ljust(18) + ' Supports: Inequality Constraints, Equality Constraints. '
    'Warning'.ljust(6) + ': by default takes maxeval function evaluations.',
    'ESCH'.ljust(6) + ': Gradient free.'.ljust(18) + 'Warning: by default takes maxeval function evaluations.',
    'STOGO'.ljust(6) + ': Gradient based.'.ljust(18) + 'Warning: by default takes maxeval function evaluations.']

    print("NLOpt's global optimizers to be called as functions.")
    print("")
    print(*GLOBAL_OPTIMIZERS, sep='\n')