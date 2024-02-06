import numpy as np
from scipy.optimize import least_squares
import time


def residuals(params, x, data, model) -> np.ndarray:
    y_model = model.func(params, x)
    res_real = y_model.real - data.real
    res_imag = y_model.imag - data.imag
    return np.array([res_real, res_imag]).flatten()


def regularization_term(params, alpha=0.1) -> np.ndarray:
    return np.sqrt(alpha) * params


def loss(params, x, data, model) -> np.ndarray:
    # return np.concatenate([residuals(params, x, data, model), regularization_term(params)])
    return residuals(params, x, data, model)


def fit_complex(params0, x, data, model):
    params_fit = least_squares(
            loss, params0,
            bounds=(0, np.inf),
            args=(x, data, model),
            method='dogbox',
            loss='linear',
            ftol=1e-8,
            gtol=1e-8,
            xtol=1e-8,
            )
    data_fit = model.func(params_fit.x, x)
    return params_fit, data_fit


def best_fit_complex(x, data, model, convergence_threshold=30, sigma=0.1):
    np.random.seed(32)
    params0 = [1 for _ in range(model.params_num)]
    best_params_fit, best_data_fit = fit_complex(params0, x, data, model)
    best_cost = np.inf
    convergence_counter = 0
    last_params_fit = None
    while convergence_counter >= convergence_threshold:
        params_fit, data_fit = fit_complex(params0, x, data, model)
        # TEST #######################
        # print("\nCounter:", convergence_counter)
        # if last_params_fit is not None:
        #     for i, diff in enumerate(last_params_fit.x - params_fit.x):
        #         percent = diff / last_params_fit.x[i]
        #         print(f"Relative diff: {percent:.3e}")
        #     print("\nCosts:")
        #     print(f"\tLast: {last_params_fit.cost:.3e}")
        #     print(f"\tCurr: {params_fit.cost:.3e}")
        #     print(f"\tBest: {best_cost:.3e}\n\n")
        #     # time.sleep(0.1)
        ##############################
        params0 = params_fit.x * \
            (1 + np.random.uniform(low=-sigma, high=sigma, size=model.params_num))
        if params_fit.cost < best_cost:
            best_cost = params_fit.cost
            best_params_fit = params_fit
            best_data_fit = data_fit
            convergence_counter = 0
        else:
            convergence_counter += 1
        # last_params_fit = params_fit
    return best_params_fit, best_data_fit, best_cost
