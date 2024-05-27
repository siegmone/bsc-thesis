import glob
from models import R_RC_RC
from utils import get_impedance_data, get_valid_fits, print_raw_csv, params_to_json, get_diode_capacitance, get_vi_data, get_diode_res
from plot2 import plot_nyquist, plot_bode, plot_caps, plot_vi, plot_res
import numpy as np
from numpy import cos, sin
from icecream import ic
import matplotlib.pyplot as plt
import os
from fit import chi2_test

def sort_arrays(arr1, arr2):
    # Create pairs of elements from both arrays
    pairs = list(zip(arr1, arr2))

    # Sort pairs based on the values from the first array
    sorted_pairs = sorted(pairs, key=lambda x: x[0])

    # Extract sorted elements from both arrays
    sorted_arr1 = [pair[0] for pair in sorted_pairs]
    sorted_arr2 = [pair[1] for pair in sorted_pairs]

    return sorted_arr1, sorted_arr2


def main():
    models = [R_RC_RC()]
    date = "2024-01-15"
    diodes = ["1N4007"]
    # diodes = ["1N4001", "1N4002", "1N4003"]

    f_fit = np.logspace(-1, 8, 10000)

    p_values = open("p_values.txt", "w")

    with open("validation.txt", "w") as file:
        file.write("")

    for diode in diodes:
        char_files = glob.glob(f"experiments/{diode}_{date}/CHARACTERISTIC/*.csv")
        for idx, char_file in enumerate(char_files):
            v, i, v_err, i_err = get_vi_data(char_file)
            plot_vi(v, i, v_err, i_err, title=f"{diode}_Characteristic-Curve")
        csv_files = glob.glob(f"experiments/{diode}_{date}/BIAS_SCAN/*.csv")
        for csv_file in csv_files:
            bias = str(csv_file.split('/')[-1].split('.')[0])
            f, Z, Z_mag, theta, flag = get_impedance_data(csv_file)
            bias = bias.removesuffix("mV")
            print_raw_csv(f"{csv_file.removesuffix('mV')}", f"clean_raw/{diode}-{bias}")
            Z_real, Z_imag = Z.real, Z.imag
            Z_mag_err = 0.001 * Z_mag
            Z_real_err = np.abs(Z_real * 0.001)
            Z_imag_err = np.abs(Z_imag * 0.001)
            theta_err = 0.1 * np.ones_like(theta, float)
            data = np.concatenate([Z_mag, theta])
            sigma = np.concatenate([Z_mag_err, theta_err])
            data_bode = np.concatenate([Z_mag, theta])
            for model in models:
                n_params = model.params_num
                title = f"{diode} Diode @ {bias}mV"
                print(f"\n\nFitting {diode} @ {bias} with {model.name}\n")
                p, sigma_p, valid = model.fit(f, data, sigma)
                data_fit = model.mag_phase(p, f)
                with open("validation.txt", "a") as file:
                    if valid:
                        file.write(f"{diode} {bias} {model.name} VALID\n")
                    else:
                        file.write(f"{diode} {bias} {model.name} INVALID\n")
                params_to_json("params.json", diode, bias, model, p, sigma_p)
                print("Plotting Nyquist Plot")
                data_nyquist_fit = model.impedance(p, f_fit)
                plot_nyquist(
                    f, Z,
                    f_fit, data_nyquist_fit,
                    Z_real_err, Z_imag_err,
                    p, sigma_p, model,
                    title=title
                )
                print("Plotting Bode Plot")
                data_bode_fit = model.mag_phase(p, f_fit)
                plot_bode(
                    f, data_bode,
                    f_fit, data_bode_fit,
                    Z_mag_err, theta_err,
                    p, sigma_p, model,
                    title=title
                )
                p_value = chi2_test(data, data_fit, sigma, n_params)
                p_values.write(f"{diode}-{bias}-{model.name} -- {p_value:.3e}\n")
        valids, invalids = get_valid_fits("validation.txt")
        valids = valids[:, 1]
        cbias, cp1, cp2 = get_diode_capacitance("params.json", valids)[diode]
        rbias, rs, rp1, rp2 = get_diode_res("params.json", valids)[diode]

        rbias_s, rp1 = sort_arrays(rbias, rp1)
        _, rp2 = sort_arrays(rbias, rp2)
        _, rs = sort_arrays(rbias, rs)
        plot_res(rbias_s[:-1], rs[:-1], rp1[:-1], rp2[:-1], diode, title="Diode Resistances", labels=["$R_s$", "$R_1$", "$R_2$"])

        cbias_s, cp1 = sort_arrays(cbias, cp1)
        _, cp2 = sort_arrays(cbias, cp2)
        np.savetxt(f"{diode}-capacitances.txt", np.array([cbias_s, cp1, cp2]).T)
        plot_caps(cbias_s[:-1], cp1[:-1], diode + "tutto", title="Diode Capacities", label=r"$C_{\text{diff}}+C_{\text{depl}}$", opt=cp2[:-1], opt_label=r"$C_{\text{stray}}$")
    print("Done!")
    p_values.close()
    exit(0)


if __name__ == '__main__':
    main()
