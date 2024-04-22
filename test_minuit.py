from utils import get_impedance_data
from iminuit import Minuit
from iminuit.cost import LeastSquares
from models import R_RC_RC
import glob
import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from fit import best_fit_complex
from plot import plot_impedance_fit



model = R_RC_RC()
model_name = model.name
model_func = model.func_flat
diode = "1N4007"
exp_type = "BIAS_SCAN"
date = "2024-01-15"

csv_files = glob.glob(f"experiments/{diode}_{date}/{exp_type}/*.csv")
csv_file = "experiments/1N4007_2024-01-15/BIAS_SCAN/500.0mV.csv"

bias = csv_file.split('/')[-1].split('.')[0]
freq, Z, theta = get_impedance_data(csv_file)

Z_mag = np.abs(Z)
Z_mag_err = 0.001 * Z_mag

Z_real, Z_imag = Z.real, Z.imag
Z_flat = np.concatenate((Z_real, Z_imag))

Z_real_err, Z_imag_err = np.abs(Z_mag_err * cos(theta)), np.abs(Z_mag_err * sin(theta))
Z_err = np.concatenate((Z_real_err, Z_imag_err))

theta_err = 0.1 * np.ones(len(theta))
freq_flat = np.concatenate((freq, freq))

plt.errorbar(
    Z_real, Z_imag,
    xerr=Z_real_err, yerr=Z_imag_err,
    ecolor='k', elinewidth=0.5, capsize=2, fmt='none', zorder=1,
    label="Data"
)
print(Z.shape)
print(Z_err.shape)

params_fit, data_fit = best_fit_complex(freq, Z, model)
params0 = tuple(params_fit.x)
print(params0)

print(new_params)
data_fit_min = model.func(new_params, freq)

plot_impedance_fit(freq, Z, data_fit_min, new_params, model)
