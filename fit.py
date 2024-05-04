import numpy as np
from scipy.optimize import least_squares
from utils import print_parameters_with_model_name


def residuals(params, x, data, model):
    y_model = model.mag_phase(params, x)
    # y_model = model.all(params, x)
    res = y_model - data
    return res


def fit(params0, x, data, model):
    params = least_squares(
        residuals, params0,
        bounds=(0, np.inf), args=(x, data, model),
        method='trf', loss='linear',
        ftol=1e-8, gtol=1e-8, xtol=1e-8,
    )
    p, cost = params.x, params.cost
    return p, cost


def best_fit(x, data, model):
    convergence_threshold = 500
    sigma = 0.25
    np.random.seed(42)
    p0 = [1 for _ in range(model.params_num)]
    p_fit, cost_fit = fit(p0, x, data, model)
    best_cost = np.inf
    convergence_counter = 0
    while convergence_counter < convergence_threshold:
        # print_parameters_with_model_name(p_fit, model)
        p_fit, cost_fit = fit(p0, x, data, model)
        p0 = p_fit * \
            (1 + np.random.uniform(low=-sigma, high=sigma, size=model.params_num))
        if cost_fit < best_cost:
            best_cost = cost_fit
            best_p_fit = p_fit
            convergence_counter = 0
        else:
            convergence_counter += 1
    return best_p_fit
