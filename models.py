import numpy as np
from fit import best_fit
from iminuit import Minuit
from iminuit.cost import LeastSquares
from icecream import ic


def imp_C(f, C):
    return 1 / (1j * 2 * np.pi * f * C)


def imp_L(f, L):
    return 1j * 2 * np.pi * f * L


def imp_Q(f, Q, a):
    return Q / (1j * 2 * np.pi * f)**a


def imp_W(f, W):
    return (W / np.sqrt(2 * np.pi * f)) + (W / (1j * np.sqrt(2 * np.pi * f)))


def series(*args):
    return sum(args)


def parallel(*args):
    return 1 / sum(1 / arg for arg in args)


class Model:
    def __init__(self, name):
        self.name = name
        self.params_names = []
        self.params_units = []
        self.params_num = 0

    def __repr__(self):
        return self.name

    def set_params_num(self):
        self.params_num = len(self.params_names)

    def impedance(self, params, f):
        raise NotImplementedError

    def mag_phase(self, params, f):
        Z = self.impedance(params, f)
        return np.concatenate([np.abs(Z), np.angle(Z, deg=True)])

    def all(self, params, f):
        Z = self.impedance(params, f)
        Z_real, Z_imag = Z.real, Z.imag
        return np.concatenate([self.mag_phase(params, f), Z_real, Z_imag])

    def mag_phase_minuit(self, ff, params):
        f = ff[:len(ff) // 2]
        Z = self.impedance(params, f)
        return np.concatenate([np.abs(Z), np.angle(Z, deg=True)])

    def real_imag_minuit(self, ff, params):
        f = ff[:len(ff) // 2]
        Z = self.impedance(params, f)
        return np.concatenate([Z.real, Z.imag])

    def all_minuit(self, f_4, params):
        f = f_4[:len(f_4) // 4]
        Z = self.impedance(params, f)
        Z_real, Z_imag = Z.real, Z.imag
        return np.concatenate([self.mag_phase(params, f), Z_real, Z_imag])

    def fit(self, f, data, sigma):
        print("Using custom minimizer")
        p = best_fit(f, data, self)
        ff = np.concatenate([f, f])
        print("Using Minuit")
        ls = LeastSquares(ff, data, sigma, self.mag_phase_minuit)
        minuit = Minuit(ls, p, name=self.params_names)
        minuit.limits = [(0, None) for _ in range(self.params_num)]
        minuit.migrad()
        minuit.hesse()
        print(minuit)
        return minuit.values, minuit.errors, minuit.valid


class RC(Model):
    def __init__(self):
        super().__init__('RC')
        self.params_names = ['R', 'C']
        self.params_units = [r'\Omega', 'F']
        super().set_params_num()

    def impedance(self, params, f):
        R, C = params
        Z_real = R / (1 + (2 * np.pi * f * R * C)**2)
        Z_imag = -(2 * np.pi * f * C * (R**2)) / \
            (1 + (2 * np.pi * f * R * C) ** 2)
        Z = Z_real + 1j * Z_imag
        return Z


class R_RC(Model):
    def __init__(self):
        super().__init__('R_RC')
        self.params_names = ['Rs', 'Rp', 'Cp']
        self.params_units = [r'\Omega', r'\Omega', 'F']
        super().set_params_num()

    def impedance(self, params, f):
        Rs, Rp, Cp = params
        Z = series(Rs, RC().impedance([Rp, Cp], f))
        return Z


class R_RC_RC(Model):
    def __init__(self):
        super().__init__('R_RC_RC')
        self.params_names = ['Rs', 'Rp1', 'Cp1', 'Rp2', 'Cp2']
        self.params_units = [r'\Omega', r'\Omega', 'F', r'\Omega', 'F']
        super().set_params_num()

    def impedance(self, params, f):
        Rs, Rp1, Cp1, Rp2, Cp2 = params
        p1 = RC().impedance([Rp1, Cp1], f)
        p2 = RC().impedance([Rp2, Cp2], f)
        Z = series(Rs, p1, p2)
        return Z


class R_RC_RC_RC(Model):
    def __init__(self):
        super().__init__('R_RC_RC_RC')
        self.params_names = ['Rs', 'Rp1', 'Cp1', 'Rp2', 'Cp2', 'Rp3', 'Cp3']
        self.params_units = [r'\Omega', r'\Omega', 'F', r'\Omega', 'F',
                             r'\Omega', 'F']
        super().set_params_num()

    def impedance(self, params, f):
        Rs, Rp1, Cp1, Rp2, Cp2, Rp3, Cp3 = params
        p1 = RC().impedance([Rp1, Cp1], f)
        p2 = RC().impedance([Rp2, Cp2], f)
        p3 = RC().impedance([Rp3, Cp3], f)
        Z = series(Rs, p1, p2, p3)
        return Z


class R_RCW(Model):
    def __init__(self):
        super().__init__('R_RCW')
        self.params_names = ['Rs', 'Rp', 'Cp', 'W']
        self.params_units = [r'\Omega', r'\Omega',
                             'F', r'\frac{\Omega}{s^{1/2}}']
        super().set_params_num()

    def impedance(self, params, f):
        Rs, Rp, Cp, W = params
        Z_R = Rs
        Z_W = imp_W(f, W)
        Z_C = imp_C(f, Cp)
        Z = series(
            Rs,
            parallel(
                Z_C,
                series(Z_R, Z_W)
            )
        )
        return Z
