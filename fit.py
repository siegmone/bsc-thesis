import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2
from iminuit import Minuit
from iminuit.cost import LeastSquares


def residuals(params, x, data, model):
    y_model = model.func(params, x)
    res_real = y_model.real - data.real
    res_imag = y_model.imag - data.imag
    return np.array([res_real, res_imag]).flatten()


def loss(params, x, data, model) -> np.ndarray:
    return residuals(params, x, data, model)


def fit_complex(params0, x, data, model):
    params_fit = least_squares(
        loss, params0, bounds=(0, np.inf), args=(x, data, model),
        method='trf', loss='linear', ftol=1e-8, gtol=1e-8, xtol=1e-8,
    )
    data_fit = model.func(params_fit.x, x)
    return params_fit, data_fit


def residuals_phase(params, x, data, model) -> np.ndarray:
    y_model = model.func(params, x)
    res_real = y_model.real - data.real
    res_imag = y_model.imag - data.imag
    res_phase = np.angle(y_model) - np.angle(data)
    return np.array([res_real, res_imag, res_phase]).flatten()


def loss_phase(params, x, data, model) -> np.ndarray:
    return residuals_phase(params, x, data, model)


def fit_complex_phase(params0, x, data, model):
    params_fit = least_squares(
        loss_phase, params0,
        bounds=(0, np.inf),
        args=(x, data, model),
        method='trf',
        loss='linear',
        ftol=1e-8,
        gtol=1e-8,
        xtol=1e-8,
    )
    data_fit = model.func(params_fit.x, x)
    return params_fit, data_fit


def best_fit_complex(x, data, model, err, convergence_threshold=100, sigma=0.1):
    np.random.seed(42)
    params0 = [1 for _ in range(model.params_num)]
    best_params_fit, best_data_fit = fit_complex_phase(params0, x, data, model)
    best_cost = np.inf
    convergence_counter = 0
    while convergence_counter < convergence_threshold:
        params_fit, data_fit = fit_complex(params0, x, data, model)
        params0 = params_fit.x * \
            (1 + np.random.uniform(low=-sigma, high=sigma, size=model.params_num))
        if params_fit.cost < best_cost:
            best_cost = params_fit.cost
            best_params_fit = params_fit
            best_data_fit = data_fit
            convergence_counter = 0
        else:
            convergence_counter += 1
    ls = LeastSquares(x, data, err, model.func_flat)
    m = Minuit(ls, params0, name=model.params_names)
    m.limits = [(0, None), (0, None), (0, None), (0, None), (0, None)]
    m.migrad()
    m.hesse()
    m.minos()
    best_params_fit = m.values
    best_data_fit = model.func(best_params_fit, x)
    return best_params_fit, best_data_fit


def chi2_test_pvalue(data, data_fit, model, params_fit):
    data, data_fit = data[3:], data_fit[3:]
    magnitude, magnitude_fit = np.abs(data), np.abs(data_fit)
    phase, phase_fit = np.angle(data), np.angle(data_fit)
    res = np.concatenate((magnitude - magnitude_fit, phase - phase_fit))
    mag_err = 0.001 * np.abs(data)
    phase_err = np.full_like(phase, 0.1)
    err = np.concatenate((mag_err, phase_err))
    chi2_val = np.sum(res**2 / err**2)
    dof = len(data) * 2 - model.params_num
    p_value = chi2.sf(chi2_val, dof)
    return p_value


def chi2_test_pvalue_phase(data, data_fit, model, params_fit):
    res = residuals_phase(params_fit.x, data, data_fit, model)
    chi2_val = np.sum(res**2)
    dof = len(data) * 3 - model.params_num
    p_value = 1 - chi2.cdf(chi2_val, dof)
    return p_value
