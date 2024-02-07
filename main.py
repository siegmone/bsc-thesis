from fit import best_fit_complex
from models import R_RC, R_RC_RC, R_RC_RC_RC, R_RCW
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import time
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
import logging


def format_param_latex(p):
    coeff, exponent = f"{p:.3e}".split('e')
    exponent = "{" + str(int(exponent)) + "}"
    formatted_number = rf"{coeff}\cdot 10^{exponent}"
    return formatted_number


def get_impedance_data(filepath):
    df = pd.read_csv(filepath, skiprows=3, sep=', ', engine='python')
    df = df[df["Frequency (Hz)"].notna()]
    freq = np.array(df["Frequency (Hz)"])
    Z = np.array(df["Z' (Ohm)"]) + np.array(df["Z'' (Ohm)"]) * 1j
    freq, Z = preprocessing.ignoreBelowX(freq, Z)
    return freq, Z


def plot_impedance_fit(x, data, model, title="Impedance Fit", sigma=0.1, convergence_threshold=30):
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax = plt.subplots(figsize=(12, 9))

    params, fit, cost = best_fit_complex(x, data, model)

    scatter = ax.scatter(data.real, data.imag,
                         label="Impedance Data", c=x, cmap='rainbow_r', ec='k')
    ax.plot(fit.real, fit.imag, label="Best Fit", ls='--', c='red')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'$\text{Frequency (Hz)}$', rotation=0, labelpad=20)

    ax.legend(loc='lower right')

    text = ""
    for param_name, param_unit, param in zip(model.params_names, model.params_units, params.x):
        param = format_param_latex(param)
        text += f"${param_name}={param} {param_unit}$\n"
    text = text.strip()
    props = dict(boxstyle='round', fc='white',
                 ec='blue', lw=2, pad=1, alpha=0.5)
    ax.text(0.42, 0.90, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.set_title(title)
    ax.set_xlabel(r"$Z_\text{Re} (\Omega)$")
    ax.set_ylabel(r"$Z_\text{Im} (\Omega)$")
    ax.grid(True, alpha=0.5, linestyle='--')

    fig.savefig(f"plots/bias_scan/{title}.png")
    plt.close(fig)

    return params, fit, cost


def fit_diode(diode, date, exp_type, models, sigma=0.1, convergence_threshold=30):
    csv_files = glob.glob(f"experiments/{diode}_{date}/{exp_type}/*.csv")
    stats = {}  # []
    failures = {}
    for csv_file in csv_files:
        bias = csv_file.split('/')[-1].split('.')[0]
        freq, Z = get_impedance_data(csv_file)
        if len(Z) < 5:
            logging.info(
                f"{diode} @ {bias} has insufficient data (< 5 points)"
            )
            continue
        for model in models:
            print(f"\n\nFitting {diode} @ {bias} with {model.name}\n")
            params, fit, cost = plot_impedance_fit(
                freq,
                Z,
                model,
                title=f"{diode} @ {bias} - {model.name} fit",
                sigma=sigma,
                convergence_threshold=convergence_threshold
            )
            # stats.append([diode, bias, model.name, cost, *params.x])
            if params.success:
                stats[(bias, model)] = [
                    (name, param) for name, param in zip(model.params_names, params.x)
                ]
                print(f"Fit successful: {params.message}")
            else:
                print(f"Fit failed: {params.message}")
                failures[(diode, bias, model.name)] = params.message
    return stats, failures


def filter_stats(stats, fix_bias=None, fix_model=None):
    if fix_bias is not None and fix_model is not None:
        stats = {
            key: val for key, val in stats.items() if key[0] == fix_bias and key[1].name == fix_model
        }
    elif fix_bias is not None:
        stats = {
            key: val for key, val in stats.items() if key[0] == fix_bias
        }
    elif fix_model is not None:
        stats = {
            key: val for key, val in stats.items() if key[1].name == fix_model
        }
    biases = list(set([key[0].removesuffix("mV")
                  for key in stats.keys()])).sort()
    models = list(set([key[1].name for key in stats.keys()]))
    return stats


def calculate_taus(params, model):
    if model.name == "R_RC":
        Rs, Rp, Cp = params
        tau = Rs * Cp
        return tau,
    elif model.name == "R_RC_RC":
        Rs, Rp1, Cp1, Rp2, Cp2 = params
        tau1 = Rs * Cp1
        tau2 = Rs * Cp2
        return tau1, tau2
    elif model.name == "R_RC_RC_RC":
        Rs, Rp1, Cp1, Rp2, Cp2, Rp3, Cp3 = params
        tau1 = Rs * Cp1
        tau2 = Rs * Cp2
        tau3 = Rs * Cp3
        return tau1, tau2, tau3


def write_stats(stats, filename):
    dict_stats = {
        "Bias": [],
        "Model": [],
        "Rs": [],
        "Rp": [],
        "Cp": [],
        "Rp1": [],
        "Cp1": [],
        "Rp2": [],
        "Cp2": [],
        "Rp3": [],
        "Cp3": [],
        "tau1": [],
        "tau2": [],
        "tau3": [],
    }
    i = 0
    for (bias, model), n_params in stats.items():
        i += 1
        bias_v = int(bias.removesuffix("mV"))
        dict_stats["Bias"].append(bias_v)
        dict_stats["Model"].append(model.name)
        for name, param in n_params:
            if name not in dict_stats:
                dict_stats[name] = []
            dict_stats[name].append(param)
        params = [param for name, param in n_params]
        taus = calculate_taus(params, model)
        for k, tau in enumerate(taus):
            dict_stats[f"tau{k + 1}"].append(tau)
        for name in dict_stats.keys():
            if name not in params and name not in ["Bias", "Model"] and i > len(dict_stats[name]):
                dict_stats[name].append(None)
    df = pd.DataFrame(dict_stats)
    df.to_csv(filename, index=False)


def plot_total_capacitance(stats, diode, model_name=None, bias=None):
    possible_capacitance_names = ["Cp", "Cp1", "Cp2", "Cp3"]
    labels = {
        "R_RC": "Cp",
        "R_RC_RC": "Cp1 + Cp2",
        "R_RC_RC_RC": "Cp1 + Cp2 + Cp3",
    }
    filtered_stats = filter_stats(stats, fix_model=model_name, fix_bias=bias)
    data = []
    capacitances = {name: [] for name in possible_capacitance_names}
    for (bias, model), params in filtered_stats.items():
        C_tot = 0
        bias_v = int(bias.removesuffix("mV"))
        for name, param in params:
            if name in possible_capacitance_names:
                capacitances[name].append(param)
                C_tot += param
        data.append([bias_v, C_tot])
    data = np.array(data)
    sorter = data[:, 0].argsort()
    data = data[sorter]
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(data[:, 0], data[:, 1], label=labels[model_name], marker='x')
    for name, values in capacitances.items():
        if len(values) > 0:
            values_sort = np.array(values)[sorter]
            # ax.plot(data[:, 0], values_sort, label=name)
    ax.set_title(
        f"Total Capacitance vs Bias for {diode} - {model_name} fit")
    ax.set_xlabel("Bias (mV)")
    ax.set_ylabel("Capacitance (F)")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.legend(loc='upper left')
    fig.savefig(f"plots/properties/{diode}_{model_name}_total_capacitance.png")
    plt.close(fig)


def plot_series_resistance(stats, diode, model_name=None, bias=None):
    filtered_stats = filter_stats(stats, fix_model=model_name, fix_bias=bias)
    data = []
    for (bias, model), params in filtered_stats.items():
        bias_v = int(bias.removesuffix("mV"))
        for name, param in params:
            if name == "Rs":
                data.append([bias_v, param])
    data = np.array(data)
    sorter = data[:, 0].argsort()
    data = data[sorter]
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(data[:, 0], data[:, 1], label="Rs", marker='x')
    ax.set_title(
        f"Series Resistance vs Bias for {diode} - {model_name} fit")
    ax.set_xlabel("Bias (mV)")
    ax.set_ylabel(r"${Series Resistance (\Omega)}$")
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.legend(loc='upper left')
    fig.savefig(f"plots/properties/{diode}_{model_name}_series_resistance.png")
    plt.close(fig)


def plot_parallel_resistances(stats, diode, model_name=None, bias=None):
    possible_resistances_names = ["Rp", "Rp1", "Rp2", "Rp3"]
    labels = {
        "R_RC": "Rp",
        "R_RC_RC": "Rp1 + Rp2",
        "R_RC_RC_RC": "Rp1 + Rp2 + Rp3",
    }
    filtered_stats = filter_stats(stats, fix_model=model_name, fix_bias=bias)
    data = []
    resistances = {name: [] for name in possible_resistances_names}
    for (bias, model), params in filtered_stats.items():
        bias_v = int(bias.removesuffix("mV"))
        Rp_tot = 0
        for name, param in params:
            if name in possible_resistances_names:
                resistances[name].append(param)
                Rp_tot += param
        data.append([bias_v, Rp_tot])
    data = np.array(data)
    sorter = data[:, 0].argsort()
    data = data[sorter]
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(data[:, 0], data[:, 1], label=labels[model_name], marker='x')
    for name, values in resistances.items():
        if len(values) > 0:
            values_sort = np.array(values)[sorter]
            ax.plot(data[:, 0], values_sort, label=name)
    ax.set_title(
        f"Resistance vs Bias for {diode} - {model_name} fit")
    ax.set_xlabel("Bias (mV)")
    ax.set_ylabel(r"$Resistance (\Omega)$")
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.legend(loc='upper left')
    fig.savefig(f"plots/properties/{diode}_{model_name}_resistances.png")
    plt.close(fig)


def main():
    sigma = 0.3
    convergence_threshold = 200

    # models = [R_RC(), R_RC_RC(), R_RC_RC_RC(), R_RCW()]
    models = [R_RC(), R_RC_RC(), R_RC_RC_RC()]
    # models = [R_RC_RC()]

    model_names = [model.name for model in models]

    exp_type = "BIAS_SCAN"
    date = "2024-01-15"
    diodes = ["1N4001", "1N4002", "1N4003", "1N4007"]

    # stats: [diode, bias, model, cost, *params]

    for diode in diodes:
        stats, failures = fit_diode(
            diode,
            date,
            exp_type,
            models,
            sigma=sigma,
            convergence_threshold=convergence_threshold
        )
        for key, val in failures.items():
            logging.error(f"Fit failed for: {key} with message: {val}")
            time.sleep(0.1)
        for model in model_names:
            plot_total_capacitance(stats, diode, model_name=model)
            plot_series_resistance(stats, diode, model_name=model)
            plot_parallel_resistances(stats, diode, model_name=model)
        write_stats(stats, f"stats/{diode}_{date}_{exp_type}.csv")
    print("Done!")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        filename='logs/main.log', filemode='w'
    )
    main()
