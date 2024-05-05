import glob
from models import R_RC_RC
from utils import get_impedance_data, get_valid_fits, print_raw_csv, params_to_json
from plot import plot_nyquist, plot_bode
import numpy as np
from numpy import cos, sin
from icecream import ic


def main():
    models = [R_RC_RC()]
    BIAS_SCAN = "BIAS_SCAN"
    CHARACTERISTIC = "CHARACTERISTIC"
    date = "2024-01-15"
    diodes = ["1N4001", "1N4002", "1N4003", "1N4007"]

    f_fit = np.logspace(-1, 8, 10000)

    with open("validation.txt", "w") as f:
        f.write("")

    for diode in diodes:
        csv_files = glob.glob(f"experiments/{diode}_{date}/BIAS_SCAN/*.csv")
        for csv_file in csv_files:
            bias = str(csv_file.split('/')[-1].split('.')[0]).removesuffix("mV")
            f, Z, Z_mag, theta, flag = get_impedance_data(csv_file)
            print_raw_csv(csv_file, f"clean_raw/{diode}-{bias}")
            Z_real, Z_imag = Z.real, Z.imag
            if flag:
                continue
            Z_mag_err = 0.001 * Z_mag
            Z_real_err = np.abs(Z_real * 0.001)
            Z_imag_err = np.abs(Z_imag * 0.001)
            theta_err = 0.1 * np.ones_like(theta, float)
            data = np.concatenate([Z_mag, theta])
            sigma = np.concatenate([Z_mag_err, theta_err])
            data_bode = np.concatenate([Z_mag, theta])
            for model in models:
                title = f"{diode}@{bias}-{model.name}"
                print(f"\n\nFitting {diode} @ {bias} with {model.name}\n")
                p, sigma_p, valid = model.fit(f, data, sigma)
                with open("validation.txt", "a") as file:
                    if valid:
                        file.write(f"{diode} {bias} {model.name} VALID\n")
                    else:
                        file.write(f"{diode} {bias} {model.name} INVALID\n")
                if not valid:
                    continue
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
    get_valid_fits("validation.txt")
    print("Done!")
    exit(0)


if __name__ == '__main__':
    main()
