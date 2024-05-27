import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
from subprocess import call
from time import sleep
from icecream import ic
import json

FONTSIZE = 16


# define clear function
def clear():
    # check and make call for specific operating system
    _ = call('clear' if os.name == 'posix' else 'cls')


def print_parameters_with_model_name(params, model):
    output = ""
    for name, p in zip(model.params_names, params):
        output += f"{name} = {p:.3e}\n"
    output.strip()
    print(output)
    sleep(0.01)
    clear()


def get_valid_fits(filepath):
    colnames = ["DIODE", "BIAS", "MODEL", "FIT_VALIDATION"]
    df = pd.read_csv(filepath, sep=" ", header=None, names=colnames)
    valid_df = df[df["FIT_VALIDATION"] == "VALID"].sort_values(by=["DIODE", "BIAS"])
    invalid_df = df[df["FIT_VALIDATION"] == "INVALID"].sort_values(by=["DIODE", "BIAS"])
    valid = np.array(valid_df[["DIODE", "BIAS"]])
    invalid = np.array(invalid_df[["DIODE", "BIAS"]])
    ic(df)
    ic(valid)
    ic(invalid)
    return valid, invalid


def format_param_latex(p):
    coeff, exponent = f"{float(p):.3e}".split('e')
    exponent = "{" + str(int(exponent)) + "}"
    formatted_number = rf"{coeff}\cdot 10^{exponent}"
    return formatted_number


def get_impedance_data(filepath, drop=0):
    flag = False
    df = pd.read_csv(filepath, skiprows=3, sep=', ', engine='python')
    ic(filepath)
    df = df[df["Frequency (Hz)"].notna() & (df["Z'' (Ohm)"] < 0)]
    df.drop(df.tail(drop).index, inplace=True)
    if df.empty:
        print("No valid points!")
        flag = True
    freq = np.array(df["Frequency (Hz)"])
    Z = np.array(df["Z' (Ohm)"]) + np.array(df["Z'' (Ohm)"]) * 1j
    Z_mag = np.array(df["| Z | (Ohm)"])
    theta = np.array(df["Phase (Deg)"])
    return freq, Z, Z_mag, theta, flag


def print_raw_csv(src, dest):
    freq, Z, Z_mag, phase, _ = get_impedance_data(src)
    df = pd.DataFrame({
        "frequency": freq,
        "Z": Z_mag,
        "phase": phase,
        "Z_re": Z.real,
        "Z_im": Z.imag,
        })
    df.to_csv(dest + ".csv", index=False)


def params_to_json(filepath, diode, bias, model, p, sigma_p):
    p_l = list(p)
    sigma_p_l = list(sigma_p)
    exists = False
    if os.path.exists(filepath):
        # Load JSON data from file
        try:
            with open(filepath, "r") as json_file:
                data = json.load(json_file)
            exists = True
        except Exception as e:
            print(e)
            print("Error decoding existing json file... creating new one")
            exists = False
    if not exists:
        # Initialize an empty dictionary if the file doesn't exist
        data = {
                diode: {
                    bias: {
                        model.name: {
                            "params": model.params_names,
                            "values": p_l,
                            "errors": sigma_p_l
                            }
                        }
                    }
                }
    if diode not in data:
        data[diode] = {}
    if bias not in data[diode]:
        data[diode][bias] = {}
    if model.name not in data[diode][bias]:
        data[diode][bias][model.name] = {}
    data[diode][bias][model.name]["params"] = model.params_names
    data[diode][bias][model.name]["values"] = p_l
    data[diode][bias][model.name]["errors"] = sigma_p_l
    with open(filepath, "w") as json_file:
        json.dump(data, json_file, indent=4)


def get_diode_capacitance(filepath, valids=None):
    valids = list(valids)
    with open(filepath, "r") as json_file:
        data = json.load(json_file)
    d = {}
    for diode in data:
        biases = []
        cp1s = []
        cp2s = []
        diode_data = data[diode]
        for bias in diode_data:
            bias = bias.removesuffix("mV")
            if int(bias) not in valids:
                continue
            bias_data = diode_data[bias]
            biases.append(int(bias))
            for model in bias_data:
                model_data = bias_data[model]
                names = model_data["params"]
                values = model_data["values"]
                errors = model_data["errors"]
                idx1, idx2 = names.index("Cp1"), names.index("Cp2")
                cp1, cp2 = values[idx1], values[idx2]
                condition = cp1 > cp2
                cp1s.append(cp1 * condition + (1 - condition) * cp2)
                condition = not condition
                cp2s.append(cp1 * condition + (1 - condition) * cp2)
        d[diode] = (biases, cp1s, cp2s)
    return d


def get_diode_res(filepath, valids=None):
    valids = list(valids)
    with open(filepath, "r") as json_file:
        data = json.load(json_file)
    d = {}
    for diode in data:
        biases = []
        rss = []
        rp1s = []
        rp2s = []
        diode_data = data[diode]
        for bias in diode_data:
            bias = bias.removesuffix("mV")
            if int(bias) not in valids:
                continue
            bias_data = diode_data[bias]
            biases.append(int(bias))
            for model in bias_data:
                model_data = bias_data[model]
                names = model_data["params"]
                values = model_data["values"]
                errors = model_data["errors"]
                idx1, idx2, idx3 = names.index("Rs"), names.index("Rp1"), names.index("Rp2")
                rs, rp1, rp2 = values[idx1], values[idx2], values[idx3]
                rss.append(rs)
                condition = rp1 > rp2
                rp1s.append(rp1 * condition + (1 - condition) * rp2)
                condition = not condition
                rp2s.append(rp1 * condition + (1 - condition) * rp2)
        d[diode] = (biases, rss, rp1s, rp2s)
    return d




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




def plot_impedance_fit(x, data, fit, params, model, title="Impedance Fit"):
    # plt.style.use('seaborn-v0_8-colorblind')
    plt.style.use(["science", "ieee"])
    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(
        data.real,
        -data.imag,
        label="Impedance Data",
        c=x,
        cmap='rainbow_r',
        ec='k',
        vmin=1, vmax=1e6,
        zorder=2
    )
    xerr = 0.01 * np.abs(data.real)
    yerr = 0.01 * np.abs(data.imag)
    ax.errorbar(
        data.real, -data.imag, xerr=xerr, yerr=yerr,
        ecolor='k', elinewidth=0.5, capsize=2, fmt='none', zorder=1
    )
    ax.plot(fit.real, -fit.imag, label="Best Fit", ls='--', c='red')
    cbar = plt.colorbar(scatter, ax=ax, extend='both')
    cbar.set_label(r'$\text{Frequency (Hz)}$',
                   rotation=90, labelpad=20, size=FONTSIZE)
    cbar.ax.tick_params(labelsize=FONTSIZE)
    text = ""
    for param_name, param_unit, param in zip(model.params_names, model.params_units, params.x):
        param = format_param_latex(param)
        text += f"${param_name}={param} {param_unit}$\n"
    text = text.strip()
    props = dict(boxstyle='round', fc='white',
                 ec='blue', lw=2, pad=1, alpha=0.5)
    ax.text(0.42, 0.30, text, transform=ax.transAxes, fontsize=FONTSIZE,
            verticalalignment='top', bbox=props)
    ax.set_title(title, fontsize=FONTSIZE)
    ax.set_xlabel(r"$Z_\text{Re} (\Omega)$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$-Z_\text{Im} (\Omega)$", fontsize=FONTSIZE)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.legend(loc='upper left', fontsize=FONTSIZE)
    fig.savefig(f"plots/bias_scan/{title}.png")
    plt.close(fig)


def plot_diff_bode(x, Z, fit, title="Magnitude diff"):
    plt.style.use(["science", "ieee"])
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, np.zeros_like(x), c='k', ls='--')
    ax.scatter(
        x, np.abs(Z) - np.abs(fit),
        label=r"$|Z - Z_{fit}|$", c='blue', ec='k', zorder=2
    )
    ax.errorbar(
        x, np.abs(Z) - np.abs(fit), xerr=0, yerr=0.001 * np.abs(Z),
        ecolor='k', elinewidth=0.5, capsize=2, fmt='none', zorder=1
    )
    ax.set_title(title, fontsize=FONTSIZE)
    ax.set_xlabel(r"$\text{Frequency (Hz)}$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$|Z - Z_{fit}| (\Omega)$", color='blue', fontsize=FONTSIZE)
    ax.set_xscale('log')
    ax.set_ylim(bottom=-2, top=2)
    ax.legend(loc='lower left', fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.5, linestyle='--')
    fig.savefig(f"plots/bias_scan/{title}_diff_m.png")
    plt.close(fig)

    fig1, ax1 = plt.subplots(figsize=(12, 9))
    bins = np.linspace(-2, 2, 40, endpoint=True)
    ax1.hist(np.abs(Z) - np.abs(fit), bins=bins, color='blue', alpha=0.9, ec="k")
    ax1.set_xlim(left=-2, right=2)

    ax1.set_title(f"{title} Histogram", fontsize=FONTSIZE)
    ax1.set_xlabel(r"$|Z - Z_{fit}| (\Omega)$", fontsize=FONTSIZE)
    ax1.set_ylabel("Frequency", fontsize=FONTSIZE)
    ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax1.grid(True, alpha=0.5, linestyle='--')
    fig1.savefig(f"plots/bias_scan/{title}_diff_m_hist.png")
    plt.close(fig1)


def plot_diff_phase(x, theta, fit, title="Phase diff"):
    theta_fit = np.abs(np.angle(fit, deg=True))
    plt.style.use(["science", "ieee"])
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, np.zeros_like(x), c='k', ls='--')
    ax.scatter(
        x, theta - theta_fit,
        label=r"$|\theta - \theta_{fit}|$", c='green', ec='k', zorder=2
    )
    ax.errorbar(
        x, theta - theta_fit, xerr=0, yerr=0.1,
        ecolor='k', elinewidth=0.5, capsize=2, fmt='none', zorder=1
    )
    ax.set_title(title, fontsize=FONTSIZE)
    ax.set_xlabel(r"$\text{Frequency (Hz)}$", fontsize=FONTSIZE)
    ax.set_ylabel(
        r"$|\theta - \theta_{fit}| (\Omega)$", color='green', fontsize=FONTSIZE)
    ax.set_xscale('log')
    ax.set_ylim(bottom=-2, top=2)
    ax.legend(loc='lower left', fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.5, linestyle='--')
    fig.savefig(f"plots/bias_scan/{title}_diff_p.png")
    plt.close(fig)

    fig1, ax1 = plt.subplots(figsize=(12, 9))
    bins = np.linspace(-2, 2, 40, endpoint=True)
    ax1.hist(theta - theta_fit, bins=bins, color='green', alpha=0.9, ec="k")
    ax1.set_xlim(left=-2, right=2)
    ax1.set_title(f"{title} Histogram", fontsize=FONTSIZE)
    ax1.set_xlabel(r"$|\theta - \theta_{fit}| (\Omega)$", fontsize=FONTSIZE)
    ax1.set_ylabel("Frequency", fontsize=FONTSIZE)
    ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax1.grid(True, alpha=0.5, linestyle='--')
    fig1.savefig(f"plots/bias_scan/{title}_diff_p_hist.png")
    plt.close(fig1)


def plot_bodeplot(x, Z, theta, fit, params, model, title="Bodeplot Fit"):
    plt.style.use(["science", "ieee"])
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(
        x, np.abs(Z),
        label=r"$|Z|$", c='blue', ec='k', zorder=2
    )
    ax.plot(x, np.abs(fit), label=r"$|Z|$ fit", ls='--', c='blue')
    ax.set_title(title, fontsize=FONTSIZE)
    ax.set_xlabel(r"$\text{Frequency (Hz)}$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$|Z| (\Omega)$", color='blue', fontsize=FONTSIZE)
    ax.set_xscale('log')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.25), fontsize=FONTSIZE)

    theta_fit = np.abs(np.angle(fit, deg=True))
    ax2 = ax.twinx()
    ax2.set_ylabel('Phase (°)', color='red', fontsize=FONTSIZE)
    ax2.scatter(
        x, theta,
        label=r"$\theta$", c='green', ec='k', zorder=2
    )
    ax2.plot(x, theta_fit, label=r"$\theta$ fit", ls='--', c='red')
    yerr = 0.1
    ax2.errorbar(
        x, theta, xerr=0, yerr=yerr,
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

    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax2.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax2.legend(loc='lower left', bbox_to_anchor=(0.2, 0.26), fontsize=FONTSIZE)

    fig.savefig(f"plots/bias_scan/{title}_bode.png")

    plt.close(fig)


def fit_diode(diode, date, exp_type, models, biases, sigma=0.1, convergence_threshold=30):
    csv_files = glob.glob(f"experiments/{diode}_{date}/{exp_type}/*.csv")
    stats = {}
    failures = {}
    to_ignore = 5
    for csv_file in csv_files:
        bias = csv_file.split('/')[-1].split('.')[0]
        bias_int = int(bias.removesuffix("mV"))
        if bias_int < biases[0] or bias_int > biases[1]:
            continue
        freq, Z, theta = get_impedance_data(csv_file)
        freq, Z, theta = freq[:-to_ignore], Z[:-to_ignore], theta[:-to_ignore]
        for model in models:
            print(f"\n\nFitting {diode} @ {bias} with {model.name}\n")
            params, fit = best_fit_complex(
                freq,
                Z,
                model,
                err=0.001 * np.abs(Z),
                sigma=sigma,
                convergence_threshold=convergence_threshold
            )
            chi2_pvalue = chi2_test_pvalue(Z, fit, model, params)
            plot_impedance_fit(
                freq, Z, fit, params, model,
                title=f"{diode} @ {bias} - {model.name} fit",
            )
            plot_bodeplot(
                freq, Z, theta, fit, params, model,
                title=f"{diode} @ {bias} - {model.name} fit",
            )
            plot_diff_bode(
                freq, Z, fit,
                title=f"{diode} @ {bias} - {model.name} fit",
            )
            plot_diff_phase(
                freq, theta, fit,
                title=f"{diode} @ {bias} - {model.name} fit",
            )

            if params.success:
                stats[(bias, model)] = [
                    (name, param) for name, param in zip(model.params_names, params.values)
                ]
                print(f"Fit successful: {params.message}")
            else:
                print(f"Fit failed: {params.message}")
                failures[(diode, bias, model.name)] = params.message
    return stats, failures


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
    ax.plot(data[:len(data) - 4, 0], data[:len(data) - 4, 1],
            label=labels[model_name], marker='x')
    for name, values in capacitances.items():
        if len(values) > 0:
            values_sort = np.array(values)[sorter]
            ax.plot(data[:len(data) - 4, 0],
                    values_sort[:len(data) - 4], label=name)
    ax.set_title(
        f"Total Capacitance vs Bias for {diode} - {model_name} fit",
        fontsize=FONTSIZE
    )
    ax.set_xlabel("Bias (mV)", fontsize=FONTSIZE)
    ax.set_ylabel("Capacitance (F)", fontsize=FONTSIZE)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.legend(loc='upper left', fontsize=FONTSIZE)
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
    plt.style.use(["science", "ieee"])
    ax.plot(data[:, 0], data[:, 1], label="Rs", marker='x')
    ax.set_title(
        f"Series Resistance vs Bias for {diode} - {model_name} fit",
        fontsize=FONTSIZE
    )
    ax.set_xlabel("Bias (mV)", fontsize=FONTSIZE)
    ax.set_ylabel(r"${Series Resistance (\Omega)}$", fontsize=FONTSIZE)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.legend(loc='upper left', fontsize=FONTSIZE)
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
    plt.style.use(["science", "ieee"])
    ax.plot(data[:, 0], data[:, 1], label=labels[model_name], marker='x')
    for name, values in resistances.items():
        if len(values) > 0:
            values_sort = np.array(values)[sorter]
            ax.plot(data[:, 0], values_sort, label=name)
    ax.set_title(
        f"Resistance vs Bias for {diode} - {model_name} fit",
        fontsize=FONTSIZE
    )
    ax.set_xlabel("Bias (mV)", fontsize=FONTSIZE)
    ax.set_ylabel(r"$Resistance (\Omega)$", fontsize=FONTSIZE)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.legend(loc='upper left', fontsize=FONTSIZE)
    fig.savefig(f"plots/properties/{diode}_{model_name}_resistances.png")
    plt.close(fig)


def get_vi_data(filepath):
    df = pd.read_csv(filepath, skiprows=3, sep=', ', engine='python')
    voltage = np.array(df["Voltage (V)"])
    current = np.array(df["Current (A)"])
    range_conv = {
        1: 3e-1, 2: 3e-2, 3: 3e-3, 4: 3e-4,
        5: 3e-5, 6: 3e-6, 7: 3e-7, 8: 3e-8,
    }
    voltage_range = np.array(df["V Range ()"])
    current_range = np.array(df["I Range ()"])
    voltage_err = np.zeros_like(voltage_range)
    current_err = np.zeros_like(current_range)
    current_offset = 30e-15
    for i, cr in enumerate(current_range):
        reading = current[i]
        c_range = range_conv[cr]
        # 0.1% + 0.05% + 30 fA
        current_err[i] = (reading * 1e-3) + (c_range * 5e-4) + current_offset
    voltage_err = np.abs(voltage_err)
    current_err = np.abs(current_err)
    return voltage, current, voltage_err, current_err


def plot_characteristic(diode, date, exp_type):
    csv_files = glob.glob(f"experiments/{diode}_{date}/{exp_type}/*.csv")
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.style.use(["science", "ieee"])
    for csv_file in csv_files:
        v, i, v_err, i_err = get_vi_data(csv_file)
        ax.scatter(v, np.abs(i), marker=".", c="k", ec="k",
                   label=f"{diode}", alpha=0.7, zorder=2)
        ax.errorbar(v, np.abs(i), xerr=v_err, yerr=i_err, fmt="none")
        ax.set_yscale('log')
        ax.set_title(f"Characteristic for {diode}", fontsize=FONTSIZE)
        ax.legend(loc='upper left', fontsize=FONTSIZE)
        ax.set_xlabel("Voltage (V)", fontsize=FONTSIZE)
        ax.set_ylabel("Current (A)", fontsize=FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        ax.set_xlim(left=-0.4, right=0.4)
        ax.grid(True, alpha=0.5, linestyle='--')
    fig.savefig(f"plots/characteristics/{diode}_{date}.png")


def plot_all_char(diodes, date, exp_type):
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.style.use(["science", "ieee"])
    ax.set_title("Characteristics", fontsize=FONTSIZE)
    for diode in diodes:
        csv_files = glob.glob(f"experiments/{diode}_{date}/{exp_type}/*.csv")
        for csv_file in csv_files:
            v, i, v_r, i_r = get_vi_data(csv_file)
            ax.plot(
                v, np.abs(i), label=f"{diode}",
                alpha=0.7, lw=2
            )
            ax.set_yscale('log')
            ax.legend(loc='upper left', fontsize=FONTSIZE)
            ax.set_xlabel("Voltage (V)", fontsize=FONTSIZE)
            ax.set_ylabel("Current (A)", fontsize=FONTSIZE)
            ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
            ax.set_xlim(left=-0.4, right=0.4)
            ax.grid(True, alpha=0.5, linestyle='--')
    fig.savefig(f"plots/characteristics/{date}.png")
