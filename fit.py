import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2


def residuals(params, x, data, model) -> np.ndarray:
    y_model = model.func(params, x)
    res_mod = np.abs(y_model) - np.abs(data)
    res_phase = np.angle(y_model) - np.angle(data)
    return np.array([res_mod, res_phase]).flatten()


def loss(params, x, data, model) -> np.ndarray:
    return residuals(params, x, data, model)


def fit(params0, x, data, model):
    params_fit = least_squares(
        loss, params0, bounds=(0, np.inf), args=(x, data, model),
        method='trf', loss='linear', ftol=1e-8, gtol=1e-8, xtol=1e-8,
    )
    data_fit = model.func(params_fit.x, x)
    data_mod = np.abs(data_fit)
    data_phase = np.angle(data_fit)
    data_fit_mp = np.array([data_mod, data_phase]).flatten()
    return params_fit, data_fit_mp


def best_fit(x, data, model, convergence_threshold=100, sigma=0.1):
    np.random.seed(32)
    params0 = [1 for _ in range(model.params_num)]
    best_params_fit, best_data_fit = fit(params0, x, data, model)
    best_cost = np.inf
    convergence_counter = 0
    while convergence_counter < convergence_threshold:
        params_fit, data_fit = fit(params0, x, data, model)
        params0 = params_fit.x * \
            (1 + np.random.uniform(low=-sigma, high=sigma, size=model.params_num))
        if params_fit.cost < best_cost:
            best_cost = params_fit.cost
            best_params_fit = params_fit
            best_data_fit = data_fit
            convergence_counter = 0
        else:
            convergence_counter += 1
    return best_params_fit, best_data_fit


def chi2_test_pvalue(data, data_fit, model, params_fit) -> float:
    data_mod = np.abs(data)
    data_phase = np.angle(data)
    data_mp = np.array([data_mod, data_phase]).flatten()
    mod_err = np.abs(data_mp) * 0.001
    phase_err = 0.1
    err = np.array([mod_err, phase_err]).flatten()
    chi2_val = np.sum((data_mp - data_fit)**2 / err**2)
    dof = len(data_fit) - model.params_num
    p_value = chi2.sf(chi2_val, dof)
    return p_value
