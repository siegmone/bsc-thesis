import numpy as np
import pandas as pd
from impedance import preprocessing
import logging


logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/main.log', filemode='w'
)


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
    theta = np.abs(np.angle(Z, deg=True))
    return freq, Z, theta


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
        "Bias": [], "Model": [],
        "Rs": [],
        "Rp": [], "Cp": [],
        "Rp1": [], "Cp1": [],
        "Rp2": [], "Cp2": [],
        "Rp3": [], "Cp3": [],
        "tau1": [], "tau2": [], "tau3": [],
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
