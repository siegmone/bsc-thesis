from fit import best_fit_complex
from models import R_RC, R_RC_RC, R_RC_RC_RC, R_RCW
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

from impedance import preprocessing
from impedance.models.circuits import CustomCircuit


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


def plot_impedance_fit(x, data, model, title="Impedance Fit"):
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

    fig.savefig(f"plots/{title}.png")

    return params, fit, cost


def fit_diode(diode, date, exp_type, models):
    csv_files = glob.glob(f"experiments/{diode}_{date}/{exp_type}/*.csv")
    stats = []
    for csv_file in csv_files:
        bias = csv_file.split('/')[-1].split('.')[0]
        print(f"\n\nFitting {diode} @ {bias}\n")
        freq, Z = get_impedance_data(csv_file)
        for model in models:
            params, fit, cost = plot_impedance_fit(
                freq, Z, model, title=f"{diode} @ {bias} - {model.name} fit")
            stats.append([diode, bias, model.name, cost, *params.x])
    return stats


def main():
    models = [R_RC(), R_RC_RC(), R_RC_RC_RC(), R_RCW()]

    exp_type = "BIAS_SCAN"
    date = "2024-01-15"


    # diode = "1N4001"
    # fit_diode(diode, date, exp_type, models)

    # diode = "1N4002"
    # fit_diode(diode, date, exp_type, models)

    # diode = "1N4003"
    # fit_diode(diode, date, exp_type, models)

    # diode = "1N4007"
    # fit_diode(diode, date, exp_type, models)


if __name__ == '__main__':
    main()
