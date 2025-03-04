def optimize(method, problem):
    if method == 'opro':
        from .methods.opro import OPRO
        return OPRO().optimize(problem)
    elif method == 'lmea':
        from .methods.lmea import LMEA
        return LMEA().optimize(problem)
    else:
        raise ValueError('Unknown method')