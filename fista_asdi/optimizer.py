__author__ = "Nicolas Mil-Homens Cavaco"
__all__ = ["optimize", 
           "power_method"]

import numpy as np
import torch 

def optimize(oracle, X, S, stepsize, MAX_ITER, tol=1e-4, verbose=False, output_list=None):
    """ Optimize with FISTA """
    i = 1
    stop_crit = tol+1
    X_new, S_new = torch.zeros_like(X), torch.zeros_like(S)
    y_X, y_S = X.clone(), S.clone()
    f_val = oracle.eval_obj(X, S)
    t = 1
    while stop_crit >= tol and i < MAX_ITER:
        X_new, S_new = oracle.update(y_X, y_S, *oracle.grad(y_X, y_S), mu=stepsize) 
        f_val_next = oracle.eval_obj(X_new, S_new)
        
        t_new = (1+np.sqrt(1+4*t**2)) / 2
        y_X, y_S = X_new + (t-1)/t_new * (X_new - X), S_new + (t-1)/t_new * (S_new - S)

        # Check stoping criterion
        stop_crit = oracle.stop_crit(S_new, S)

        if verbose:
            print(i, float(stop_crit), float(f_val-f_val_next), float(f_val_next), float(stepsize))
        if output_list is not None:
            output_list.append((i, float(stop_crit), 0.0, float(stepsize), 
                        float(oracle.eval_obj(X, S))))
        
        X[:], S[:] = X_new, S_new 
        t = t_new
        f_val = f_val_next
        i+=1
    
    return X, S, None, i, output_list


def power_method(shape_X, shape_S, grad, tol=5e-4, max_iter=500):
    # Estimate Lipschitz constant 
    X0 = torch.rand(*tuple(shape_X))
    S0 = torch.rand(*tuple(shape_S))
    A_norm = 1.0
    A_norm_prev = A_norm+1+tol
    n = 0
    while n <= max_iter:
        X0, S0 = grad(X0/A_norm, S0/A_norm, no_cube=True)
        A_norm = torch.sqrt(torch.sum(X0*X0)+torch.sum(S0*S0))
        delta = abs(A_norm - A_norm_prev)
        if delta <= tol:
            return float(A_norm)

        # print(n, ":", delta, float(A_norm))
        A_norm_prev = A_norm
        n+=1
    
    raise RuntimeError('max iter reached for Lipschitz constant estimation')
    # return float(A_norm)