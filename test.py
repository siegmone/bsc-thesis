import unittest
import matplotlib.pyplot as plt
import numpy as np
from fit import best_fit_complex, fit_complex
from models import R_RC, R_RC_RC, R_RC_RC_RC, R_RCW, R_R, parallel, RC, imp_C, imp_W, series
from matplotlib.colors import LogNorm
import logging
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist


logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    filename='tests/test.log', filemode='w'
)

skip_plot = False
skip_circ = False
flag_imp = True
flag_rcw = True
skip_seed = True


def format_param_latex(p):
    coeff, exponent = f"{p:.3e}".split('e')
    exponent = "{" + str(int(exponent)) + "}"
    formatted_number = rf"{coeff}\cdot 10^{exponent}"
    return formatted_number


def generate_data(model, params, f):
    return model.func(params, f)


def plot_impedance_fit(params, f, model, type):
    data = model.func(params, f)
    noise = 0.01
    np.random.seed(32)
    try:
        data.real += noise * np.random.rand(f.shape[0])
        data.imag += noise * np.random.rand(f.shape[0])
    except AttributeError:
        data += noise * np.random.rand(f.shape[0])

    params_fit, data_fit, cost = best_fit_complex(
        f, data, model, sigma=0.4, convergence_threshold=100)

    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(data.real, -data.imag, label="data",
                         c=f, cmap='rainbow_r', norm=LogNorm())
    ax.plot(data_fit.real, -data_fit.imag, label="fit", ls='--', c='red', lw=2)
    ax.set_title(f"{model.name} - {type}")

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
    fig.savefig(f"tests/test_{model.name}_{type}.png")
    return params_fit, data_fit, cost


def log_test(f, model, params, rtol, type):
    params_fit, data_fit, cost = plot_impedance_fit(params, f, model, type)
    print(np.isclose(params_fit.x, params, rtol=rtol))
    # LOGGING #######################
    logging.info(f"TEST: test_{model.name}_{type}()")
    logging.info(f"MODEL: {model.name}")
    logging.info(f"params_names: {model.params_names}")
    logging.info(f"params: {*params,}")
    logging.info(f"params_fit: {[f'{i:.3e}' for i in params_fit.x]}")
    logging.info(
        f"close: {np.isclose(params_fit.x, params, rtol=rtol)}\n")
    #################################
    return params_fit, data_fit, cost


class TestFit(unittest.TestCase):
    def setUp(self):
        self.rtol = 0.1
        self.delay = 1
        self.f = np.logspace(-1, 7, 1000)
        self.Rs = 100
        self.Rp1 = 1000
        self.Cp1 = 1e-9
        self.Rp2 = 10000
        self.Cp2 = 1e-7
        self.Rp3 = 100000
        self.Cp3 = 1e-6

    @unittest.skipIf(skip_seed, "Skipping test_find_seed()")
    def test_find_seed(self):
        print("\nTesting find_seed()...")
        f = self.f
        models = [R_RC(), R_RC_RC(), R_RC_RC_RC()]
        params1 = [self.Rs, self.Rp1, self.Cp1]
        params2 = [self.Rs, self.Rp1, self.Cp1, self.Rp2, self.Cp2]
        params3 = [self.Rs, self.Rp1, self.Cp1,
                   self.Rp2, self.Cp2, self.Rp3, self.Cp3]
        parameters = [params1, params2, params3]
        datas = [model.func(params, f)
                 for model, params in zip(models, parameters)]
        i = 0
        logging.info("TEST: test_find_seed()")
        while i < 1000:
            logging.info(f"Seed: {i}")
            print(f"Seed: {i}")
            close_counter = 0
            for model, data, params in zip(models, datas, parameters):
                print(f"Testing {model.name}")
                np.random.seed(i)
                params_fit, data_fit, cost = best_fit_complex(
                    f, data, model, sigma=0.1, convergence_threshold=30)
                # assert not np.isclose(params_fit.x, params, rtol=self.rtol)
                # LOGGING #######################
                logging.info(f"TEST: test_{model.name}_{type}()")
                logging.info(f"MODEL: {model.name}")
                logging.info(f"params_names: {model.params_names}")
                logging.info(f"params: {*params,}")
                logging.info(
                    f"params_fit: {[f'{i:.3e}' for i in params_fit.x]}")
                logging.info(
                    f"close: {np.isclose(params_fit.x, params, rtol=self.rtol)}\n")
                #################################
                if np.allclose(params_fit.x, params, rtol=self.rtol):
                    close_counter += 1
                print(np.isclose(params_fit.x, params, rtol=self.rtol))
                print(f"close_counter: {close_counter}")
                fig, ax = plt.subplots(figsize=(12, 9))
                scatter = ax.scatter(data.real, -data.imag, label="data",
                                     c=f, cmap='rainbow_r', norm=LogNorm())
                ax.plot(data_fit.real, -data_fit.imag,
                        label="fit", ls='--', c='red', lw=2)
                ax.set_title(f"{model.name} - {type}")

                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(r'$\text{Frequency (Hz)}$',
                               rotation=0, labelpad=40)

                text = ""
                for param_name, param_unit, param in zip(model.params_names, model.params_units, params_fit.x):
                    param = format_param_latex(param)
                    text += f"${param_name}={param} {param_unit}$\n"
                text = text.strip()
                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                ax.text(0.42, 0.90, text, transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)

                ax.legend()
                fig.savefig(f"tests/test_{model.name}_{i}.png")
                plt.close(fig)
            if close_counter == len(models):
                print(f"Found the one and only True Seed: {i}")
                break
            i += 1

    @unittest.skipIf(skip_plot, "Skipping test_R_RC_fit()")
    def test_R_RC_fit(self):
        print("\nTesting R_RC_fit()...")

        # Test RC
        f = self.f
        Rs = self.Rs
        Rp = self.Rp1
        Cp = self.Cp1
        params = [Rs, Rp, Cp]
        model = R_RC()
        params_fit, data_fit, cost = log_test(
            f, model, params, rtol=self.rtol, type="fit")
        tau1 = Rp * Cp
        tau1_fit = params_fit.x[1] * params_fit.x[2]
        taus = [tau1]
        taus_fit = [tau1_fit]
        print(np.sum(taus))
        print(np.sum(taus_fit))
        assert np.isclose(np.sum(taus), np.sum(taus_fit), rtol=self.rtol)
        assert params_fit.success
        # assert np.allclose(params_fit.x, params, rtol=self.rtol)

    @unittest.skipIf(skip_plot, "Skipping test_R_RC_RC_fit()")
    def test_R_RC_RC_fit(self):
        print("\nTesting R_RC_RC_fit()...")

        # Test RC_RC
        f = self.f
        Rs = self.Rs
        Rp1 = self.Rp1
        Cp1 = self.Cp1
        Rp2 = self.Rp2
        Cp2 = self.Cp2
        params = [Rs, Rp1, Cp1, Rp2, Cp2]
        model = R_RC_RC()
        params_fit, data_fit, cost = log_test(
            f, model, params, rtol=self.rtol, type="fit")

        tau1 = Rp1 * Cp1
        tau2 = Rp2 * Cp2
        tau1_fit = params_fit.x[1] * params_fit.x[2]
        tau2_fit = params_fit.x[3] * params_fit.x[4]
        taus = [tau1, tau2]
        taus_fit = [tau1_fit, tau2_fit]
        print(np.sum(taus))
        print(np.sum(taus_fit))
        assert np.isclose(np.sum(taus), np.sum(taus_fit), rtol=self.rtol)
        assert params_fit.success
        # assert np.abs(tau3_fit - tau3) < 1e-6
        # assert np.allclose(params_fit.x, params, rtol=self.rtol)

    @unittest.skipIf(skip_plot, "Skipping test_R_RC_RC_RC_fit()")
    def test_R_RC_RC_RC_fit(self):
        print("\nTesting R_RC_RC_RC_fit()...")

        f = self.f
        Rs = self.Rs
        Rp1 = self.Rp1
        Cp1 = self.Cp1
        Rp2 = self.Rp2
        Cp2 = self.Cp2
        Rp3 = self.Rp3
        Cp3 = self.Cp3
        params = [Rs, Rp1, Cp1, Rp2, Cp2, Rp3, Cp3]
        model = R_RC_RC_RC()
        params_fit, data_fit, cost = log_test(
            f, model, params, rtol=self.rtol, type="fit")

        tau1 = Rp1 * Cp1
        tau2 = Rp2 * Cp2
        tau3 = Rp3 * Cp3
        tau1_fit = params_fit.x[1] * params_fit.x[2]
        tau2_fit = params_fit.x[3] * params_fit.x[4]
        tau3_fit = params_fit.x[5] * params_fit.x[6]

        taus = [tau1, tau2, tau3]
        taus_fit = [tau1_fit, tau2_fit, tau3_fit]

        print(np.sum(taus))
        print(np.sum(taus_fit))
        assert np.isclose(np.sum(taus), np.sum(taus_fit), rtol=self.rtol)
        assert params_fit.success
        # assert np.allclose(params_fit.x, params, rtol=self.rtol)

    @unittest.skipIf(flag_rcw, "Skipping test_R_RCW_fit()")
    def test_R_RCW_fit(self):
        print("\nTesting R_RCW_fit()...")

        # Test RC_RCW
        f = self.f
        Rs = self.Rs
        Rp = self.Rp1
        Cp = self.Cp1
        W = 1
        params = [Rs, Rp, Cp, W]
        model = R_RCW()
        params_fit, data_fit, cost = log_test(
            f, model, params, rtol=self.rtol, type="fit")
        print(params[3])
        print(params_fit.x[3])
        print(np.isclose(params_fit.x, params, rtol=self.rtol))
        assert params_fit.success
        assert np.allclose(params_fit.x, params, rtol=self.rtol)

    @unittest.skipIf(skip_circ, "Skipping test_RC_circ()")
    def test_RC_circ(self):
        print("\nTesting R_RC_circ()...")

        model = R_RC()
        f = self.f
        Rs = self.Rs
        Rp = self.Rp1
        Cp = self.Cp1
        params = [Rs, Rp, Cp]
        params_fit, data_fit, cost = log_test(
            f, model, params, rtol=self.rtol, type="circ")
        tau1 = Rp * Cp
        tau1_fit = params_fit.x[1] * params_fit.x[2]
        print(f"tau1: {tau1:.3e} -> {tau1_fit:.3e}")
        assert params_fit.success
        assert np.allclose(params_fit.x, params, rtol=self.rtol)

    @unittest.skipIf(skip_circ, "Skipping test_RC_RC_fit()")
    def test_RC_RC_circ(self):
        print("\nTesting R_RC_RC_circ()...")

        model = R_RC_RC()
        f = self.f
        Rs = self.Rs
        Rp1 = self.Rp1
        Cp1 = self.Cp1
        Rp2 = self.Rp2
        Cp2 = self.Cp2
        params = [Rs, Rp1, Cp1, Rp2, Cp2]
        params_fit, data_fit, cost = log_test(
            f, model, params, rtol=self.rtol, type="circ")
        tau1 = Rp1 * Cp1
        tau2 = Rp2 * Cp2
        tau1_fit = params_fit.x[1] * params_fit.x[2]
        tau2_fit = params_fit.x[2] * params_fit.x[3]
        print(f"tau1: {tau1:.3e} -> {tau1_fit:.3e}")
        print(f"tau2: {tau2:.3e} -> {tau2_fit:.3e}")
        assert params_fit.success
        assert np.allclose(params_fit.x, params, rtol=self.rtol)

    @unittest.skipIf(skip_circ, "Skipping test_RC_RC_RC_fit()")
    def test_RC_RC_RC_circ(self):
        print("\nTesting R_RC_RC_RC_circ()...")

        model = R_RC_RC_RC()
        f = self.f
        Rs = self.Rs
        Rp1 = self.Rp1
        Cp1 = self.Cp1
        Rp2 = self.Rp2
        Cp2 = self.Cp2
        Rp3 = self.Rp3
        Cp3 = self.Cp3
        params = [Rs, Rp1, Cp1, Rp2, Cp2, Rp3, Cp3]
        params_fit, data_fit, cost = log_test(
            f, model, params, rtol=self.rtol, type="circ")
        tau1 = Rp1 * Cp1
        tau2 = Rp2 * Cp2
        tau3 = Rp3 * Cp3
        tau1_fit = params_fit.x[1] * params_fit.x[2]
        tau2_fit = params_fit.x[2] * params_fit.x[3]
        tau3_fit = params_fit.x[4] * params_fit.x[5]
        print(f"tau1: {tau1:.3e} -> {tau1_fit:.3e}")
        print(f"tau2: {tau2:.3e} -> {tau2_fit:.3e}")
        print(f"tau3: {tau3:.3e} -> {tau3_fit:.3e}")
        assert params_fit.success
        assert np.allclose(params_fit.x, params, rtol=self.rtol)

    @unittest.skipIf(flag_rcw, "Skipping test_RCW_fit()")
    def test_RCW_circ(self):
        print("\nTesting R_RCW_circ()...")

        model = R_RCW()
        f = self.f
        Rs = self.Rs
        Rp = self.Rp1
        Cp = self.Cp1
        W = 1
        params = [Rs, Rp, Cp, W]
        params_fit, data_fit, cost = log_test(
            f, model, params, rtol=self.rtol, type="circ")
        print(params[3])
        print(params_fit.x[3])
        print(np.isclose(params_fit.x, params, rtol=self.rtol))
        assert params_fit.success
        assert np.allclose(params_fit.x, params, rtol=self.rtol)

    @unittest.skipIf(skip_circ, "Skipping test_R_R_fit()")
    def test_R_R_circ(self):
        print("\nTesting R_R_circ()...")

        model = R_R()
        f = np.logspace(-1, 7, 1000)
        Rs = 100
        Rp = 50
        params = [Rs, Rp]
        params_fit, data_fit, cost = log_test(
            f, model, params, rtol=self.rtol, type="circ")
        assert params_fit.success
        assert np.allclose(params_fit.x, params, rtol=self.rtol)

    def test_parallel(self):
        print("\nTesting parallel()...")
        f = self.f
        R = 100
        C = 1e-3
        Z_R = R
        Z_C = imp_C(f, C)
        Z = parallel(Z_R, Z_C)
        Z_test = 1 / (1 / Z_R + 1 / Z_C)
        assert np.allclose(Z, Z_test)
        print("Test passed")

    def test_parallel_CW(self):
        print("\nTesting parallel_CW()...")
        f = np.logspace(2, 3, 1000)
        Rs = 1
        R = 1
        C = 1
        W = 2
        params = [Rs, R, C, W]
        model = R_RCW()
        params_fit, data_fit, cost = log_test(
            f, model, params, rtol=self.rtol, type="circ")
        print(params)
        print(sum(params))
        print(params_fit.x)
        print(sum(params_fit.x))
        assert params_fit.success
        assert np.isclose(sum(params_fit.x), sum(params), rtol=self.rtol)
        print("Test passed")

    def test_complex_parallel(self):
        print("\nTesting complex_parallel()...")
        f = np.full(1000, 1/(2*np.pi), dtype=float)
        R = 1
        C = 1
        W = 1
        Z_R = R
        Z_C = imp_C(f, C)
        Z_W = imp_W(f, W)
        Z_pred = (0.75-0.25j) / 2
        Z = parallel(
            parallel(Z_R, series(Z_W, Z_C)),
            parallel(Z_R, series(Z_W, Z_C))
        )
        Z_test = np.full(1000, Z_pred, dtype=complex)
        assert np.allclose(Z, Z_test)
        print("Test passed")

#
#     @unittest.skipIf(flag_imp, "Skipping test_impedancepy_RCRCRC_fit()")
#     def test_impedancepy_RCRCRC_fit(self):
#         print("\nTesting impedancepy_RCRCRC_fit()...")
#
#         f = np.logspace(-1, 7, 1000)
#         params = [100, 50, 1e-3, 25, 1e-6, 10, 1e-12]
#         model = R_RC_RC_RC()
#         Z = generate_data(model, params, f)
#         circuit = 'R0-p(R1,C1)-p(R2,C2)-p(R3,C3)'
#         # initial_guess = [1, 1, 1, 1, 1, 1, 1]
#         initial_guess = params
#
#         circuit = CustomCircuit(circuit, initial_guess=initial_guess)
#         circuit.fit(f, Z)
#
#         Z_fit = circuit.predict(f)
#         fig, ax = plt.subplots()
#         plot_nyquist(Z, fmt='o', scale=10, ax=ax)
#         plot_nyquist(Z_fit, fmt='-', scale=10, ax=ax)
#         ax.legend(['Data', 'Fit'])
#         fig.savefig("tests/test_impedancepy_RCRCRC_fit.png")
#
#         logging.info(f"MODEL: {model.name}")
#         logging.info(f"params_names: {model.params_names}")
#         logging.info(f"params: {*params,}")
#         logging.info(
#             f"params_fit: {[f'{i:.3e}' for i in circuit.parameters_]}")
#         logging.info(
#             f"close: {np.isclose(params, list(circuit.parameters_), rtol=self.rtol)}\n")
#         assert np.isclose(list(circuit.parameters_),
#                           params, rtol=self.rtol).all()
#
#     @unittest.skipIf(flag_imp, "Skipping test_impedancepy_RCRC_fit()")
#     def test_impedancepy_RCRC_fit(self):
#         print("\nTesting impedancepy_RCRC_fit()...")
#
#         f = np.logspace(-1, 7, 1000)
#         params = [100, 1000, 1e-9, 10000, 1e-7]
#         model = R_RC_RC()
#         Z = generate_data(model, params, f)
#         circuit = 'R0-p(R1,C1)-p(R2,C2)'
#         initial_guess = [1, 1, 1, 1, 1]
#         # initial_guess = params
#
#         circuit = CustomCircuit(circuit, initial_guess=initial_guess)
#         circuit.fit(f, Z)
#
#         Z_fit = circuit.predict(f)
#         fig, ax = plt.subplots()
#         plot_nyquist(Z, fmt='o', scale=10, ax=ax)
#         plot_nyquist(Z_fit, fmt='-', scale=10, ax=ax)
#         ax.legend(['Data', 'Fit'])
#         fig.savefig("tests/test_impedancepy_RCRC_fit.png")
#
#         logging.info(f"MODEL: {model.name}")
#         logging.info(f"params_names: {model.params_names}")
#         logging.info(f"params: {*params,}")
#         logging.info(
#             f"params_fit: {[f'{i:.3e}' for i in circuit.parameters_]}")
#         logging.info(
#             f"close: {np.isclose(params, list(circuit.parameters_), rtol=self.rtol)}\n")
#         assert np.isclose(list(circuit.parameters_),
#                           params, rtol=self.rtol).all()
