import numpy as np
import matplotlib.pyplot as plt
from utils import filter_stats, format_param_latex
from matplotlib.ticker import ScalarFormatter, StrMethodFormatter
from matplotlib.colors import LogNorm
from icecream import ic

FONTSIZE = 18

def plot_nyquist(x, data, x_fit, data_fit, x_err, y_err, p, sigma_p, model, title="Nyquist-Plot"):
    # Plot settings
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(title, fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    # Data points
    scatter = ax.scatter(
        data.real, -data.imag,
        label="Data", c=x, cmap='rainbow_r', ec='k',
        zorder=2, marker='o', s=50,
        norm=LogNorm(vmin=1, vmax=1e+6)
    )
    ax.errorbar(
        data.real, -data.imag,
        xerr=x_err, yerr=y_err,
        ecolor='k', elinewidth=0.5, capsize=2, fmt='none', zorder=1
    )

    # Fit
    ax.plot(data_fit.real, -data_fit.imag,
            label="Best Fit", ls='--', c='k', alpha=0.7)
    # Colorbar
    cbar = plt.colorbar(
        scatter, ax=ax, orientation='horizontal', aspect=30)
    cbar.ax.invert_xaxis()
    # Colorbar settings
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family("'DejaVu Sans'")
    cbar.set_label(r'Frequency (Hz)', rotation=0, labelpad=20,
                   fontfamily='DejaVu Sans', fontsize=FONTSIZE)
    cbar.ax.tick_params(labelsize=FONTSIZE)
    # Text box
    # text = ""
    # for pr, v, e, u in zip(model.params_names, p, sigma_p, model.params_units):
    #     # p = format_param_latex(p)
    #     text += f"${pr}=({v:.3e} \\pm {e:.3e}) {u}$\n"
    # text = text.strip()
    # props = dict(boxstyle='round', fc='white',
    #              ec='blue', lw=2, pad=1, alpha=0.5)
    # ax.text(0.42, 0.5, text, transform=ax.transAxes, fontsize=FONTSIZE,
    #         verticalalignment='top', bbox=props)
    # Labels and ticks settings
    plt.xlabel(r"$Z'\ (\Omega)$",
               fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    plt.ylabel(r"$-Z''\ (\Omega)$",
               fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', shadow=True, prop={
              'family': 'DejaVu Sans', 'size': 16})
    ax.set_aspect('equal', adjustable='box')
    plt.ticklabel_format(style='plain')
    # plt.savefig(f"plots/bias_scan/{title}-nyquist.svg", dpi=300)
    plt.savefig(f"plots/bias_scan/{title}-nyquist.png", dpi=300)
    plt.close(fig)

    data_fit = model.impedance(p, x)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(
        x, (data.real - data_fit.real) / data_fit.real,
        label="Residual Real", marker="o"
    )
    ax.plot(
        x, (data.imag - data_fit.imag) / data_fit.imag,
        label="Residual Imag", marker="o"
    )
    ax.set_title(f"{title}-residuals", fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    ax.set_ylim(-5, 5)
    plt.xlabel(r"Frequency (Hz)",
               fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    plt.ylabel(r"Residual (%)",
               fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.legend(loc='upper left', shadow=True, prop={
              'family': 'DejaVu Sans', 'size': 16})
    # plt.savefig(f"plots/bias_scan/{title}-nyquist-residual.svg", dpi=300)
    plt.savefig(f"plots/bias_scan/{title}-nyquist-residual.png", dpi=300)
    plt.close(fig)


def plot_bode(x, data, x_fit, data_fit, Z_mag_err, theta_err, p, sigma_p, model, title="Bode-Plot"):
    # Collect all data
    ll = len(x)
    ll_f = len(x_fit)
    Z_mag, theta = data[:ll], data[ll:]
    Z_mag_fit, theta_fit = data_fit[:ll_f], data_fit[ll_f:]
    # Plot settings
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xscale('log')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_title(title, fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.set_xlabel("Frequency (Hz)", fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONTSIZE,
    #                    fontfamily='DejaVu Sans')
    # Mag data
    ax.scatter(x, Z_mag,
               label=r"|Z|", c='blue',
               ec='k', zorder=2, s=30)
    ax.errorbar(
        x, Z_mag,
        xerr=0, yerr=Z_mag_err,
        ecolor='k', elinewidth=0.5, capsize=2, fmt='none', zorder=1
    )

    # Mag fit
    ax.plot(x_fit, Z_mag_fit, label=r"|Z| fit", ls='--', c='blue', alpha=0.5)
    # Impedence settings
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONTSIZE,
    #                    fontfamily='DejaVu Sans')
    ax.legend(loc='lower left', prop={'family': 'DejaVu Sans', 'size': 16})
    ax.set_ylim(0)
    ax.set_ylabel(r"$|Z|\ (\Omega)$", color='blue',
                  fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    # Theta plot
    ax2 = ax.twinx()
    # Theta data
    ax2.scatter(x, theta,
                label=r"$\theta$", c='red', ec='k', zorder=2,
                marker='*', s=30)
    # Theta fit
    ax2.plot(x_fit, theta_fit, label=r"θ fit", ls='-.', c='red', alpha=0.5)
    ax2.errorbar(
        x, theta,
        xerr=0, yerr=theta_err,
        ecolor='k', elinewidth=0.5, capsize=2, fmt='none', zorder=1
    )
    # Theta settings
    ax2.set_ylim(0, -90)
    ax2.set_ylabel('Phase (°)', color='red',
                   fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax2.set_yticks(ticks=np.linspace(
        ax2.get_yticks()[0], ax2.get_yticks()[-1], 10))
    ax2.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax2.legend(loc='upper right', prop={'family': 'DejaVu Sans', 'size': 16})

    # plt.savefig(f"plots/bias_scan/{title}-bode.svg", dpi=300)
    plt.savefig(f"plots/bias_scan/{title}-bode.png", dpi=300)
    plt.close(fig)

    data_fit = model.mag_phase(p, x)
    Z_mag_fit, theta_fit = data_fit[:ll], data_fit[ll:]
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(
        x, (Z_mag - Z_mag_fit) / Z_mag_fit,
        label="Residual $|Z|$", marker="o"
    )
    ax.plot(
        x, (theta - theta_fit) / theta_fit,
        label="Residual Phase", marker="o"
    )
    ax.set_ylim(-5, 5)
    ax.set_title(f"{title}-residuals", fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    plt.xlabel(r"Frequency (Hz)",
               fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    plt.ylabel(r"Residual (%)",
               fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.legend(loc='upper left', shadow=True, prop={
              'family': 'DejaVu Sans', 'size': 16})
    # plt.savefig(f"plots/bias_scan/{title}-bode-residual.svg", dpi=300)
    plt.savefig(f"plots/bias_scan/{title}-bode-residual.png", dpi=300)
    plt.close(fig)


def plot_caps(biases, caps, diode, title="Capacitance Plot", label="C", opt=None, opt_label=None):
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_yscale('log')
    ax.grid(True, alpha=0.4, linestyle='--')
    # ax.set_title(title, fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.set_xlabel(r"Bias (mV)", fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.set_ylabel(r"C (F)", fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.plot(biases, caps, label=label, c='blue', marker="o",
            markeredgecolor='black', markersize=8)
    if (opt is not None) and (opt_label is not None):
        print("Apposto")
        ax.plot(biases, opt, label=opt_label, c='red', marker="*",
                markeredgecolor='black', markersize=8)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.legend(prop={'family': 'DejaVu Sans', 'size': 16})
    # plt.savefig(f"plots/{diode}-cap_v_bias.svg", dpi=300)
    plt.savefig(f"plots/properties/{diode}-cap_v_bias.png", dpi=300)
    plt.close(fig)


def plot_res(biases, res1, res2, res3, diode, title="Resistance Plot", labels=[]):
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_yscale('log')
    ax.grid(True, alpha=0.4, linestyle='--')
    # ax.set_title(title, fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.set_xlabel("Bias (mV)", fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.set_ylabel(r"$R\ (\Omega)$", fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.plot(biases, res1, label=labels[0], c='blue', marker="o",
            markeredgecolor='black', markersize=8)
    ax.plot(biases, res2, label=labels[1], c='red', marker="v",
            markeredgecolor='black', markersize=8)
    ax.plot(biases, res3, label=labels[2], c='green', marker="s",
            markeredgecolor='black', markersize=8)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.legend(prop={'family': 'DejaVu Sans', 'size': 16})
    # plt.savefig(f"plots/{diode}-{label}_v_bias.svg", dpi=300)
    plt.savefig(f"plots/properties/{diode}-{labels}_v_bias.png", dpi=300)
    plt.close(fig)


def plot_vi(v, i, v_err, i_err, title):
    v = v*1000
    plt.style.use('seaborn-v0_8-colorblind')
    i = np.abs(i)
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_yscale('log')
    ax.grid(True, alpha=0.4, linestyle='--')
    # ax.set_title(title, fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.set_xlabel("Bias (mV)", fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.set_ylabel("I (A)", fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.scatter(v, i, label="Diode characteristic", c='blue', ec='k', s=25)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    # ax.legend(prop={'family': 'DejaVu Sans', 'size': 16})
    # plt.savefig(f"plots/{title}.svg", dpi=300)
    plt.savefig(f"plots/characteristics/{title}.png", dpi=300)
    plt.close(fig)
