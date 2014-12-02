import numpy as np
from numpy import log
from scipy import optimize
from scipy.optimize import fmin_slsqp


def fit(params, returns):
    var = garch_1_1(params, returns)

    # calculate our log likelihood - we negate as optimizers minimise
    log_like = 0.5 * (log(var) + (returns**2 / var))
    return log_like.sum()


def fit_hull(params, returns):
    var = garch_1_1(params, returns)

    # calculate our log likelihood - we negate as optimizers minimise
    log_like = -(-log(var) - (returns**2 / var))
    return log_like.sum()


def garch_1_1(params, returns):
    omega = params[0]
    alpha = params[1]
    beta = params[2]

    # initialise our variance to the variance of the returns
    var = np.ones(len(returns)) * returns.var()
    for i in range(1, len(var)):
        var[i] = omega + (alpha * returns[i-1]**2) + (beta * var[i-1])
    return var


def main():
    returns = np.array([0.945532630498276,
                0.614772790142383,
                0.834417758890680,
                0.862344782601800,
                0.555858715401929,
                0.641058419842652,
                0.720118656981704,
                0.643948007732270,
                0.138790608092353,
                0.279264178231250,
                0.993836948076485,
                0.531967023876420,
                0.964455754192395,
                0.873171802181126,
                0.937828816793698])

    optimise(returns)


def optimise(returns):

    # params = np.random.randn(3)
    params = np.array([0.1, 0.1, 0.1])

    results = optimize.fmin(fit, params, args=(returns,),
                           ftol=0.00001, xtol=0.00001)
    print('[Alexander] Omega: {}, alpha: {}, beta: {}'.format(
        results[0], results[1], results[2]))
    results = optimize.fmin(fit_hull, params, args=(returns,),
                           ftol=0.00001, xtol=0.00001)
    print('[Hull]      Omega: {}, alpha: {}, beta: {}'.format(
        results[0], results[1], results[2]))

    # alternative optimisers
    param_bounds = [(0., None), (0., None), (0., None)]
    opt_result = optimize.minimize(fit, params, args=(returns,),
                                   method='L-BFGS-B', bounds=param_bounds,
                                   tol=0.00001)
    if not opt_result.success:
        print(opt_result.message)
    results = opt_result.x
    print('[Alexander] Omega: {}, alpha: {}, beta: {}'.format(
        results[0], results[1], results[2]))

    opt_result = optimize.minimize(fit_hull, params, args=(returns,),
                                  method='L-BFGS-B', bounds=param_bounds,
                                  tol=0.00001)
    if not opt_result.success:
        print(opt_result.message)
    results = opt_result.x
    print('[Hull]      Omega: {}, alpha: {}, beta: {}'.format(
        results[0], results[1], results[2]))


if __name__ == '__main__':
    main()