import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2
from utils import get_impedance_data
from plot import plot_impedance_fit, plot_bodeplot
import logging
import glob

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/main.log', filemode='w'
)


def residuals(params, x, data, model) -> np.ndarray:
    y_model = model.func(params, x)
    res_real = y_model.real - data.real
    res_imag = y_model.imag - data.imag
    res_mod = np.abs(y_model) - np.abs(data)
    res_phase = np.angle(y_model) - np.angle(data)
    return np.array([res_real, res_imag, res_mod, res_phase]).flatten()


def loss(params, x, data, model) -> np.ndarray:
    return residuals(params, x, data, model)


def fit(params0, x, data, model):
    params_fit = least_squares(
        loss, params0, bounds=(0, np.inf), args=(x, data, model),
        method='trf', loss='linear', ftol=1e-8, gtol=1e-8, xtol=1e-8,
    )
    data_fit = model.func(params_fit.x, x)
    data_real = data_fit.real
    data_imag = data_fit.imag
    data_mod = np.abs(data_fit)
    data_phase = np.angle(data_fit)
    data_fit_mp = np.array([data_real, data_imag, data_mod, data_phase]).flatten()
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
    length = len(data)
    mod_err = 0.001 * np.ones(length // 2)
    phase_err = 0.1 * np.ones(length // 2)
    err = np.array([mod_err, phase_err]).flatten()
    chi2_val = np.sum((data - data_fit)**2 / err**2)
    dof = len(data_fit) - model.params_num
    p_value = chi2.sf(chi2_val, dof)
    return p_value


def fit_diode(diode, date, exp_type, models, sigma=0.1, convergence_threshold=30):
    csv_files = glob.glob(f"experiments/{diode}_{date}/{exp_type}/*.csv")
    stats = {}
    failures = {}
    for csv_file in csv_files:
        bias = csv_file.split('/')[-1].split('.')[0]
        freq, Z, theta = get_impedance_data(csv_file)
        if len(Z) < 5:
            logging.info(
                f"{diode} @ {bias} has insufficient data (< 5 points)"
            )
            continue
        for model in models:
            print(f"\n\nFitting {diode} @ {bias} with {model.name}\n")
            params, fit = best_fit(
                freq, Z,
                model, sigma=sigma, convergence_threshold=convergence_threshold
            )
            length = len(freq)
            fit_real = fit[:length]
            fit_imag = fit[length:2 * length]
            fit_mod = fit[2 * length:3 * length]
            fit_phase = fit[3 * length:]
            fit_mp = np.array([fit_mod, fit_phase]).flatten()
            fit_complex = fit_real + fit_imag * 1j
            data_mp = np.array([np.abs(Z), theta]).flatten()
            chi2_pvalue = chi2_test_pvalue(data_mp, fit_mp, model, params)
            logging.info(f"Fit failed: {params.message}")
            if chi2_pvalue < 0.05:
                logging.info(
                    f"Chi2 test failed for {diode} @ {bias} with {model.name} with p-value: {chi2_pvalue}"
                )
            plot_impedance_fit(
                freq, Z,
                fit_complex, params, model,
                title=f"{diode} @ {bias} - {model.name} fit",
            )
            plot_bodeplot(
                freq, Z, theta,
                fit_complex, params, model,
                title=f"{diode} @ {bias} - {model.name} fit",
            )
            if params.success:
                stats[(bias, model)] = [
                    (name, param) for name, param in zip(model.params_names, params.x)
                ]
                print(f"Fit successful: {params.message}")
            else:
                print(f"Fit failed: {params.message}")
                failures[(diode, bias, model.name)] = params.message
    return stats, failures
