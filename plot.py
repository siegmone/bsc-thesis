import numpy as np
import matplotlib.pyplot as plt
from utils import filter_stats


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
    ax.set_yscale('log')
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
    ax.set_yscale('log')
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.legend(loc='upper left')
    fig.savefig(f"plots/properties/{diode}_{model_name}_resistances.png")
    plt.close(fig)
