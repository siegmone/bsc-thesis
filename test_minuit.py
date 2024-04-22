from utils import get_impedance_data
from iminuit import Minuit
from iminuit.cost import LeastSquares
from models import R_RC_RC
import glob
import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from fit import best_fit_complex
from plot import plot_impedance_fit, plot_bodeplot
from scipy.stats import chi2, chisquare






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
Z_mag_err = 0.01 * Z_mag

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

params_fit, data_fit = best_fit_complex(freq, Z, model)

param_dict = {name: val for name, val in zip(model.params_names, params_fit)}
ls = LeastSquares(freq_flat, Z_flat, Z_err, model_func)
m = Minuit(ls, params_fit, name=model.params_names)
m.limits = [(0, None) for _ in range(model.params_num)]
m.migrad()

params_fit = m.values

x_fit = np.logspace(-1, 8, 10000)
data_fit = model.func(params_fit, x_fit)

plot_impedance_fit(freq, Z, x_fit, data_fit, params_fit, model, minuit=m, title=f"{diode}_{bias}_{model_name}")
plot_bodeplot(freq, Z, theta, x_fit, data_fit, params_fit, model, minuit=m, title=f"{diode}_{bias}_{model_name}")

plt.show()

fit_flat = np.concatenate((data_fit.real, data_fit.imag))

alpha = 0.05
dof = len(Z) - model.params_num
chi2_val = np.sum((Z_flat - fit_flat)**2 / Z_err**2)
chi2_crit = chi2.ppf(1 - alpha, dof)
p_value = 1 - chi2.cdf(chi2_val, dof)
print(f"Chi2 value: {chi2_val}")
print(f"Chi2 critical: {chi2_crit}")
print(f"P-value: {p_value}")

