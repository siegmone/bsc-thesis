from fit import best_fit_complex, chi2_test_pvalue, chi2_test_pvalue_phase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
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


def plot_impedance_fit(x, data, fit, params, model, title="Impedance Fit"):
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(
        data.real, -data.imag,
        label="Impedance Data", c=x, cmap='rainbow_r', ec='k',
        vmin=1, vmax=1e6, zorder=2
    )
    xerr = 0.01 * np.abs(data.real)
    yerr = 0.01 * np.abs(data.imag)
    ax.errorbar(
        data.real, -data.imag,
        xerr=xerr, yerr=yerr,
        ecolor='k', elinewidth=0.5, capsize=2, fmt='none', zorder=1
    )
    ax.plot(fit.real, -fit.imag, label="Best Fit", ls='--', c='red')
    cbar = plt.colorbar(scatter, ax=ax, extend='both')
    cbar.set_label(r'$\text{Frequency (Hz)}$', rotation=0, labelpad=20)
    text = ""
    for param_name, param_unit, param in zip(model.params_names, model.params_units, params.x):
        param = format_param_latex(param)
        text += f"${param_name}={param} {param_unit}$\n"
    text = text.strip()
    props = dict(boxstyle='round', fc='white',
                 ec='blue', lw=2, pad=1, alpha=0.5)
    ax.text(0.42, 0.30, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    ax.set_title(title)
    ax.set_xlabel(r"$Z_\text{Re} (\Omega)$")
    ax.set_ylabel(r"$-Z_\text{Im} (\Omega)$")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.legend(loc='upper left', fontsize=12)
    fig.savefig(f"plots/bias_scan/{title}.png")
    plt.close(fig)


def plot_bodeplot(x, Z, theta, fit, params, model, title="Bodeplot Fit"):
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(
        x, np.abs(Z),
        label=r"$|Z|$", c='blue', ec='k', zorder=2
    )
    ax.plot(x, np.abs(fit), label=r"$|Z|$ fit", ls='--', c='blue')
    ax.set_title(title)
    ax.set_xlabel(r"$\text{Frequency (Hz)}$")
    ax.set_ylabel(r"$|Z| (\Omega)$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.25), fontsize=12)

    theta_fit = np.abs(np.angle(fit, deg=True))
    ax2 = ax.twinx()
    ax2.set_ylabel('Phase (°)', color='red')
    ax2.scatter(
        x, theta,
        label=r"$\theta$", c='green', ec='k', zorder=2
    )
    ax2.plot(x, theta_fit, label=r"$\theta$ fit", ls='--', c='red')
    yerr = 0.1
    ax2.errorbar(
        x, theta,
        xerr=0, yerr=yerr,
        ecolor='k', elinewidth=0.5, capsize=2, fmt='none', zorder=1
    )
    ax2.set_ylim(0, 90)
    ax2.set_yticks(
        np.linspace(
            ax2.get_yticks()[0],
            ax2.get_yticks()[-1],
            10
        )
    )
    ax.grid(True, alpha=0.5, linestyle='--')

    ax2.legend(loc='lower left', bbox_to_anchor=(0.2, 0.26), fontsize=12)

    fig.savefig(f"plots/bias_scan/{title}_bode.png")
    # fig.savefig(f"plots/bodeplot/{title}.svg")
    plt.close(fig)


def fit_diode(diode, date, exp_type, models, sigma=0.1, convergence_threshold=30):
    csv_files = glob.glob(f"experiments/{diode}_{date}/{exp_type}/*.csv")
    stats = {}  # []
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
            params, fit = best_fit_complex(
                freq, Z,
                model, sigma=sigma, convergence_threshold=convergence_threshold
            )
            chi2_pvalue = chi2_test_pvalue_phase(Z, fit, model, params)
            logging.info(f"Fit failed: {params.message}")
            logging.info(
                f"{diode} @ {bias} with {model.name}: p-value = {chi2_pvalue:.6e} < 0.05"
            )
            plot_impedance_fit(
                freq, Z,
                fit, params, model,
                title=f"{diode} @ {bias} - {model.name} fit",
            )
            plot_bodeplot(
                freq, Z, theta,
                fit, params, model,
                title=f"{diode} @ {bias} - {model.name} fit",
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
    # biases = list(set([key[0].removesuffix("mV")
    #               for key in stats.keys()])).sort()
    # models = list(set([key[1].name for key in stats.keys()]))
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
