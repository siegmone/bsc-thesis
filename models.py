import numpy as np


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

    def func(self, params, f):
        raise NotImplementedError

    def set_params_num(self):
        self.params_num = len(self.params_names)


class RC(Model):
    def __init__(self):
        super().__init__('RC')
        self.params_names = ['R', 'C']
        self.params_units = [r'\Omega', 'F']
        super().set_params_num()

    def func(self, params, f):
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

    def func(self, params, f):
        Rs, Rp, Cp = params
        Z = series(Rs, RC().func([Rp, Cp], f))
        return Z


class R_RC_RC(Model):
    def __init__(self):
        super().__init__('R_RC_RC')
        self.params_names = ['Rs', 'Rp1', 'Cp1', 'Rp2', 'Cp2']
        self.params_units = [r'\Omega', r'\Omega', 'F', r'\Omega', 'F']
        super().set_params_num()

    def func(self, params, f):
        Rs, Rp1, Cp1, Rp2, Cp2 = params
        p1 = RC().func([Rp1, Cp1], f)
        p2 = RC().func([Rp2, Cp2], f)
        Z = series(Rs, p1, p2)
        return Z


class R_RC_RC_RC(Model):
    def __init__(self):
        super().__init__('R_RC_RC_RC')
        self.params_names = ['Rs', 'Rp1', 'Cp1', 'Rp2', 'Cp2', 'Rp3', 'Cp3']
        self.params_units = [r'\Omega', r'\Omega', 'F', r'\Omega', 'F',
                             r'\Omega', 'F']
        super().set_params_num()

    def func(self, params, f):
        Rs, Rp1, Cp1, Rp2, Cp2, Rp3, Cp3 = params
        p1 = RC().func([Rp1, Cp1], f)
        p2 = RC().func([Rp2, Cp2], f)
        p3 = RC().func([Rp3, Cp3], f)
        Z = series(Rs, p1, p2, p3)
        return Z


class R_RCW(Model):
    def __init__(self):
        super().__init__('R_RCW')
        self.params_names = ['Rs', 'Rp', 'Cp', 'W']
        self.params_units = [r'\Omega', r'\Omega',
                             'F', r'\frac{\Omega}{s^{1/2}}']
        super().set_params_num()

    def func(self, params, f):
        Rs, Rp, Cp, W = params
        Z_R = Rs
        Z_W = imp_W(f, W)
        Z_C = imp_C(f, Cp)
        Z = series(
            Rs,
            parallel(
                Z_R,
                series(Z_W, Z_C)
            )
        )
        return Z


class R_R(Model):
    def __init__(self):
        super().__init__('R_R')
        self.params_names = ['Rs', 'Rp']
        self.params_units = [r'\Omega', r'\Omega']
        super().set_params_num()

    def func(self, params, f):
        Rs, Rp = params
        Z = series(Rs, Rp)
        return Z
