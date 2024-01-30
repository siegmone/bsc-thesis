import unittest
import matplotlib.pyplot as plt
import numpy as np
from fit import best_fit_complex
from models import R_RC, R_RC_RC, R_RC_RC_RC, R_RCW, R_R
from matplotlib.colors import LogNorm
import logging
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
import time


logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    filename='tests/test.log', filemode='w'
)

flag_plot = True
flag_circ = True
flag_imp = False


def format_param_latex(p):
    coeff, exponent = f"{p:.3e}".split('e')
    exponent = "{" + str(int(exponent)) + "}"
    formatted_number = rf"{coeff}\cdot 10^{exponent}"
    return formatted_number


def generate_data(model, params, f):
    return model.func(params, f)


class TestFit(unittest.TestCase):
    def setUp(self):
        self.rtol = 0.1
        self.delay = 1

    @unittest.skipIf(flag_plot, "Skipping test_R_RC_fit()")
    def test_R_RC_fit(self):
        print("Testing R_RC_fit()...")
        time.sleep(self.delay)
        # Test RC
        f = np.logspace(-1, 7, 1000)
        Rs = 100
        Rp = 50
        Cp = 1e-3
        noise = 0.01
        model = R_RC()
        data = model.func([Rs, Rp, Cp], f)
        data.real += noise * np.random.rand(f.shape[0])
        data.imag += noise * np.random.rand(f.shape[0])

        params_fit, data_fit, cost = best_fit_complex(f, data, model)

        fig, ax = plt.subplots(figsize=(12, 9))
        scatter = ax.scatter(data.real, data.imag, label="data",
                             c=f, cmap='rainbow_r', norm=LogNorm())
        ax.plot(data_fit.real, data_fit.imag, label="fit")
        ax.set_title(model.name)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'$\text{Frequency (Hz)}$', rotation=0, labelpad=40)

        text = ""
        for param_name, param_unit, param in zip(model.params_names, model.params_units, params_fit.x):
            param = format_param_latex(param)
            text += f"${param_name}={param} {param_unit}$\n"
        text = text.strip()
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.42, 0.90, text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        ax.legend()
        fig.savefig("tests/test_R_RC.png")
        print(np.isclose(params_fit.x, [Rs, Rp, Cp], rtol=self.rtol))
        assert params_fit.success

    @unittest.skipIf(flag_plot, "Skipping test_R_RC_RC_fit()")
    def test_R_RC_RC_fit(self):
        print("Testing R_RC_RC_fit()...")
        time.sleep(self.delay)
        # Test RC_RC
        f = np.logspace(-1, 7, 1000)
        Rs = 100
        Rp1 = 50
        Cp1 = 1e-3
        Rp2 = 25
        Cp2 = 1e-6
        noise = 0.01
        model = R_RC_RC()
        data = model.func([Rs, Rp1, Cp1, Rp2, Cp2], f)
        data.real += noise * np.random.rand(f.shape[0])
        data.imag += noise * np.random.rand(f.shape[0])

        params_fit, data_fit, cost = best_fit_complex(f, data, model)

        fig, ax = plt.subplots(figsize=(12, 9))
        scatter = ax.scatter(data.real, data.imag, label="data",
                             c=f, cmap='rainbow_r', norm=LogNorm())
        ax.plot(data_fit.real, data_fit.imag, label="fit")
        ax.set_title(model.name)

        text = ""
        for param_name, param_unit, param in zip(model.params_names, model.params_units, params_fit.x):
            param = format_param_latex(param)
            text += f"${param_name}={param} {param_unit}$\n"
        text = text.strip()
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.42, 0.90, text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'$\text{Frequency (Hz)}$', rotation=0, labelpad=40)

        ax.legend()
        fig.savefig("tests/test_R_RC_RC.png")
        print(np.isclose(params_fit.x, [
              Rs, Rp1, Cp1, Rp2, Cp2], rtol=self.rtol))
        assert params_fit.success

    @unittest.skipIf(flag_plot, "Skipping test_R_RC_RC_RC_fit()")
    def test_R_RC_RC_RC_fit(self):
        print("Testing R_RC_RC_RC_fit()...")
        time.sleep(self.delay)
        # Test RC_RC
        f = np.logspace(-1, 7, 1000)
        Rs = 100
        Rp1 = 50
        Cp1 = 1e-3
        Rp2 = 25
        Cp2 = 1e-6
        Rp3 = 10
        Cp3 = 1e-12
        noise = 0.01
        model = R_RC_RC_RC()
        data = model.func([Rs, Rp1, Cp1, Rp2, Cp2, Rp3, Cp3], f)
        data.real += noise * np.random.rand(f.shape[0])
        data.imag += noise * np.random.rand(f.shape[0])

        params_fit, data_fit, cost = best_fit_complex(f, data, model)

        fig, ax = plt.subplots(figsize=(12, 9))
        scatter = ax.scatter(data.real, data.imag, label="data",
                             c=f, cmap='rainbow_r', norm=LogNorm())
        ax.plot(data_fit.real, data_fit.imag, label="fit")
        ax.set_title(model.name)

        text = ""
        for param_name, param_unit, param in zip(model.params_names, model.params_units, params_fit.x):
            param = format_param_latex(param)
            text += f"${param_name}={param} {param_unit}$\n"
        text = text.strip()
        props = dict(boxstyle='round', fc='white',
                     ec='blue', lw=2, pad=1, alpha=0.5)
        ax.text(0.42, 0.90, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'$\text{Frequency (Hz)}$', rotation=0, labelpad=40)

        ax.legend()
        fig.savefig("tests/test_R_RC_RC_RC.png")
        print(np.isclose(params_fit.x, [
              Rs, Rp1, Cp1, Rp2, Cp2, Rp3, Cp3], rtol=self.rtol))
        assert params_fit.success

    @unittest.skipIf(flag_plot, "Skipping test_R_RCW_fit()")
    def test_R_RCW_fit(self):
        print("Testing R_RCW_fit()...")
        time.sleep(self.delay)
        # Test RC_RCW
        f = np.logspace(-1, 7, 1000)
        Rs = 100
        Rp = 50
        Cp = 1e-3
        W = 1
        noise = 0.01
        model = R_RCW()
        data = model.func([Rs, Rp, Cp, W], f)
        data.real += noise * np.random.rand(f.shape[0])
        data.imag += noise * np.random.rand(f.shape[0])

        params_fit, data_fit, cost = best_fit_complex(f, data, model)

        fig, ax = plt.subplots(figsize=(12, 9))
        scatter = ax.scatter(data.real, data.imag, label="data",
                             c=f, cmap='rainbow_r', norm=LogNorm())
        ax.plot(data_fit.real, data_fit.imag, label="fit")
        ax.set_title(model.name)

        text = ""
        for param_name, param_unit, param in zip(model.params_names, model.params_units, params_fit.x):
            param = format_param_latex(param)
            text += f"${param_name}={param} {param_unit}$\n"
        text = text.strip()
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.42, 0.90, text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'$\text{Frequency (Hz)}$', rotation=0, labelpad=40)

        ax.legend()
        fig.savefig("tests/test_R_RCW.png")
        print(np.isclose(params_fit.x, [Rs, Rp, Cp, W], rtol=self.rtol))
        assert params_fit.success

    @unittest.skipIf(flag_circ, "Skipping test_R_R_fit()")
    def test_RC_circ(self):
        print("Testing R_RC_circ()...")
        time.sleep(self.delay)
        model = R_RC()
        f = np.logspace(-1, 7, 1000)
        Rs = 100
        Rp = 50
        Cp = 1e-3
        params = [Rs, Rp, Cp]
        data = model.func(params, f)
        params_fit, data_fit, cost = best_fit_complex(f, data, model)
        # LOGGING #######################
        logging.info(f"MODEL: {model.name}")
        logging.info(f"params_names: {model.params_names}")
        logging.info(f"params: {*params,}")
        logging.info(f"params_fit: {[f'{i:.3e}' for i in params_fit.x]}")
        logging.info(
            f"close: {np.isclose(params_fit.x, params, rtol=self.rtol)}\n")
        #################################
        assert params_fit.success
        assert np.allclose(params_fit.x, params, rtol=self.rtol)

    @unittest.skipIf(flag_circ, "Skipping test_RC_RC_fit()")
    def test_RC_RC_circ(self):
        print("Testing R_RC_RC_circ()...")
        time.sleep(self.delay)
        model = R_RC_RC()
        f = np.logspace(-1, 7, 1000)
        Rs = 100
        Rp1 = 50
        Cp1 = 1e-3
        Rp2 = 25
        Cp2 = 1e-6
        params = [Rs, Rp1, Cp1, Rp2, Cp2]
        data = model.func(params, f)
        params_fit, data_fit, cost = best_fit_complex(f, data, model)
        # LOGGING #######################
        logging.info(f"MODEL: {model.name}")
        logging.info(f"params_names: {model.params_names}")
        logging.info(f"params: {*params,}")
        logging.info(f"params_fit: {[f'{i:.3e}' for i in params_fit.x]}")
        logging.info(
            f"close: {np.isclose(params_fit.x, params, rtol=self.rtol)}\n")
        #################################
        assert params_fit.success
        assert np.allclose(params_fit.x, params, rtol=self.rtol)

    @unittest.skipIf(flag_circ, "Skipping test_RC_RC_RC_fit()")
    def test_RC_RC_RC_circ(self):
        print("Testing R_RC_RC_RC_circ()...")
        time.sleep(self.delay)
        model = R_RC_RC_RC()
        f = np.logspace(-1, 7, 1000)
        Rs = 100
        Rp1 = 50
        Cp1 = 1e-3
        Rp2 = 25
        Cp2 = 1e-6
        Rp3 = 10
        Cp3 = 1e-12
        params = [Rs, Rp1, Cp1, Rp2, Cp2, Rp3, Cp3]
        data = model.func(params, f)
        params_fit, data_fit, cost = best_fit_complex(f, data, model)
        # LOGGING #######################
        logging.info(f"MODEL: {model.name}")
        logging.info(f"params_names: {model.params_names}")
        logging.info(f"params: {*params,}")
        logging.info(f"params_fit: {[f'{i:.3e}' for i in params_fit.x]}")
        logging.info(
            f"close: {np.isclose(params_fit.x, params, rtol=self.rtol)}\n")
        #################################
        assert params_fit.success
        assert np.allclose(params_fit.x, params, rtol=self.rtol)

    @unittest.skipIf(flag_circ, "Skipping test_RCW_fit()")
    def test_RCW_circ(self):
        print("Testing R_RCW_circ()...")
        time.sleep(self.delay)
        model = R_RCW()
        f = np.logspace(-1, 7, 1000)
        Rs = 100
        Rp = 50
        Cp = 1e-3
        W = 1
        params = [Rs, Rp, Cp, W]
        data = model.func(params, f)
        params_fit, data_fit, cost = best_fit_complex(f, data, model)
        # LOGGING #######################
        logging.info(f"MODEL: {model.name}")
        logging.info(f"params_names: {model.params_names}")
        logging.info(f"params: {*params,}")
        logging.info(f"params_fit: {[f'{i:.3e}' for i in params_fit.x]}")
        logging.info(
            f"close: {np.isclose(params_fit.x, params, rtol=self.rtol)}\n")
        #################################
        assert params_fit.success
        assert np.allclose(params_fit.x, params, rtol=self.rtol)

    @unittest.skipIf(flag_circ, "Skipping test_R_R_fit()")
    def test_R_R_circ(self):
        print("Testing R_R_circ()...")
        time.sleep(self.delay)
        model = R_R()
        f = np.logspace(-1, 7, 1000)
        Rs = 100
        Rp = 50
        params = [Rs, Rp]
        data = model.func(params, f)
        params_fit, data_fit, cost = best_fit_complex(f, data, model)
        # LOGGING #######################
        logging.info(f"MODEL: {model.name}")
        logging.info(f"params_names: {model.params_names}")
        logging.info(f"params: {*params,}")
        logging.info(f"params_fit: {[f'{i:.3e}' for i in params_fit.x]}")
        logging.info(
            f"close: {np.isclose(params_fit.x, params, rtol=self.rtol)}\n")
        #################################
        assert params_fit.success
        assert np.allclose(params_fit.x, params, rtol=self.rtol)

    @unittest.skipIf(flag_imp, "Skipping test_impedancepy_RCRCRC_fit()")
    def test_impedancepy_RCRCRC_fit(self):
        print("Testing impedancepy_RCRCRC_fit()...")
        time.sleep(self.delay)
        f = np.logspace(-1, 7, 1000)
        params = [100, 50, 1e-3, 25, 1e-6, 10, 1e-12]
        model = R_RC_RC_RC()
        Z = generate_data(model, params, f)
        circuit = 'R0-p(R1,C1)-p(R2,C2)-p(R3,C3)'
        # initial_guess = [1, 1, 1, 1, 1, 1, 1]
        initial_guess = params

        circuit = CustomCircuit(circuit, initial_guess=initial_guess)
        circuit.fit(f, Z)

        Z_fit = circuit.predict(f)
        fig, ax = plt.subplots()
        plot_nyquist(Z, fmt='o', scale=10, ax=ax)
        plot_nyquist(Z_fit, fmt='-', scale=10, ax=ax)
        ax.legend(['Data', 'Fit'])
        fig.savefig("tests/test_impedancepy_RCRCRC_fit.png")

        logging.info(f"MODEL: {model.name}")
        logging.info(f"params_names: {model.params_names}")
        logging.info(f"params: {*params,}")
        logging.info(
            f"params_fit: {[f'{i:.3e}' for i in circuit.parameters_]}")
        logging.info(
            f"close: {np.isclose(params, list(circuit.parameters_), rtol=self.rtol)}\n")
        assert np.isclose(list(circuit.parameters_),
                          params, rtol=self.rtol).all()

    @unittest.skipIf(flag_imp, "Skipping test_impedancepy_RCRC_fit()")
    def test_impedancepy_RCRC_fit(self):
        print("Testing impedancepy_RCRC_fit()...")
        time.sleep(self.delay)
        f = np.logspace(-1, 7, 1000)
        params = [100, 50, 1e-3, 25, 1e-6]
        model = R_RC_RC()
        Z = generate_data(model, params, f)
        circuit = 'R0-p(R1,C1)-p(R2,C2)'
        # initial_guess = [1, 1, 1, 1, 1, 1, 1]
        initial_guess = params

        circuit = CustomCircuit(circuit, initial_guess=initial_guess)
        circuit.fit(f, Z)

        Z_fit = circuit.predict(f)
        fig, ax = plt.subplots()
        plot_nyquist(Z, fmt='o', scale=10, ax=ax)
        plot_nyquist(Z_fit, fmt='-', scale=10, ax=ax)
        ax.legend(['Data', 'Fit'])
        fig.savefig("tests/test_impedancepy_RCRC_fit.png")

        logging.info(f"MODEL: {model.name}")
        logging.info(f"params_names: {model.params_names}")
        logging.info(f"params: {*params,}")
        logging.info(
            f"params_fit: {[f'{i:.3e}' for i in circuit.parameters_]}")
        logging.info(
            f"close: {np.isclose(params, list(circuit.parameters_), rtol=self.rtol)}\n")
        assert np.isclose(list(circuit.parameters_),
                          params, rtol=self.rtol).all()
