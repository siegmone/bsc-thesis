from models import R_RC, R_RC_RC, R_RCW
import numpy as np
import matplotlib.pyplot as plt

FONTSIZE = 16


def plot_nyquist(x, data, title="", path="examples"):
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(
        data.real, -data.imag,
        label=r"$Z = Z_\text{Re} + j Z_\text{Im}$",
        c='red', ls='-.'
    )
    ax.set_xlabel(r"$Z' (\Omega)$", fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.set_ylabel(r"$-Z'' (\Omega)$", fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.set_title(title, fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.5, linestyle='--')
    plt.xticks(fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    plt.yticks(fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.set_aspect('equal', adjustable='box')
    fig.savefig(f"{path}/{title}.png")
    plt.close(fig)


def plot_bode(x, data, phase, title="", path="examples"):
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(x, data, label=r"$Modulus\ |Z|$", ls='-.', c='blue')
    ax.set_xlabel(r"$\text{Frequency (Hz)}$", fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.set_ylabel(r"$|Z| (\Omega)$", fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.set_title(title, fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax2 = ax.twinx()
    ax2.set_ylabel('Phase (°)', color='red', fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    ax2.plot(x, phase, label=r"$Phase\ \theta$", ls='-.', c='red')
    ax2.set_ylim(0, -90)
    ax2.set_yticks(
        np.linspace(
            ax2.get_yticks()[0],
            ax2.get_yticks()[-1],
            10
        )
    )
    ax.legend(loc="center left", bbox_to_anchor=[0.0, 0.5], fontsize=12)
    ax2.legend(loc="center left", bbox_to_anchor=[0.0, 0.4], fontsize=12)
    ax.grid(True, alpha=0.5, linestyle='--')
    plt.xticks(fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    plt.yticks(fontsize=FONTSIZE, fontfamily='DejaVu Sans')
    fig.savefig(f"{path}/{title}.png")
    plt.close(fig)


f = np.logspace(-2, 10, 10000)

model1 = R_RC()
R = 100
Rp1 = 1000
Cp1 = 1e-6

data1 = model1.impedance([R, Rp1, Cp1], f)
data1_mag_phase = model1.mag_phase([R, Rp1, Cp1], f)
data1_mag, data1_phase = data1_mag_phase[:len(f)], data1_mag_phase[len(f):]
plot_nyquist(f, data1, title="EXAMPLE-R_RC-NYQUIST")
plot_bode(f, data1_mag, data1_phase, title="EXAMPLE-R_RC-BODE")


model2 = R_RC_RC()
Rp2 = 1000
Cp2 = 1e-12

data2 = model2.impedance([R, Rp1, Cp1, Rp2, Cp2], f)
data2_mag_phase = model2.mag_phase([R, Rp1, Cp1, Rp2, Cp2], f)
data2_mag, data2_phase = data2_mag_phase[:len(f)], data2_mag_phase[len(f):]
plot_nyquist(f, data2, title="EXAMPLE-R_RC_RC-NYQUIST")
plot_bode(f, data2_mag, data2_phase, title="EXAMPLE-R_RC_RC-BODE")


model3 = R_RCW()
W = 10
data3 = model3.impedance([R, Rp1, Cp1, W], f)
data3_mag_phase = model3.mag_phase([R, Rp1, Cp1, W], f)
data3_mag, data3_phase = data3_mag_phase[:len(f)], data3_mag_phase[len(f):]
plot_nyquist(f, data3, title="EXAMPLE-RANDLES-NYQUIST")
plot_bode(f, data3_mag, data3_phase, title="EXAMPLE-RANDLES-BODE")
