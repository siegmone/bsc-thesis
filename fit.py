import numpy as np
from scipy.optimize import least_squares
from utils import print_parameters_with_model_name
from scipy.stats import chi2


def residuals(params, x, data, model):
    y_model = model.mag_phase(params, x)
    # y_model = model.all(params, x)
    res = y_model - data
    return res


def residuals_norm(params, x, data, model):
    y_model = model.mag_phase(params, x)
    # y_model = model.all(params, x)
    res = (y_model - data)
    return res


def fit(params0, x, data, model):
    params = least_squares(
        residuals_norm, params0,
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


def chi2_test(data, data_fit, sigma, n_params):
    sigma = sigma * 10
    chi2_value = np.sum(((data - data_fit)**2) / (sigma**2))
    dof = len(data_fit) - n_params
    p_value = chi2.sf(chi2_value, dof)
    return p_value

