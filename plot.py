import numpy as np
import matplotlib.pyplot as plt
from utils import filter_stats, format_param_latex


def plot_impedance_fit(x, data, fit, params, model, title="Impedance Fit"):
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(
        data.real, -data.imag,
        label="Impedance Data", c=x, cmap='rainbow_r', ec='k',
        vmin=1, vmax=1e6, zorder=2
    )
    xerr = yerr = 0.01 * np.abs(data)
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
    plt.close(fig)


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
