import numpy as np


def imp_C(f, C):
    return 1 / (1j * 2 * np.pi * f * C)


def imp_L(f, L):
    return 1j * 2 * np.pi * f * L


def imp_Q(f, Q, a):
    return Q / (1j * 2 * np.pi * f)**a


def imp_W(f, W):
    return (W / (2 * np.pi * f)) + (W / (1j * 2 * np.pi * f))


def parallel(*args):
    return 1 / sum([1 / arg for arg in args])


def series(*args):
    return sum(args)


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


class R_RC(Model):
    def __init__(self):
        super().__init__('R_RC')
        self.params_names = ['Rs', 'Rp', 'Cp']
        self.params_units = [r'\Omega', r'\Omega', 'F']
        super().set_params_num()

    def func(self, params, f):
        Rs, Rp, Cp = params
        Z = series(Rs, parallel(Rp, imp_C(f, Cp)))
        return Z


class R_RC_RC(Model):
    def __init__(self):
        super().__init__('R_RC_RC')
        self.params_names = ['Rs', 'Rp1', 'Cp1', 'Rp2', 'Cp2']
        self.params_units = [r'\Omega', r'\Omega', 'F', r'\Omega', 'F']
        super().set_params_num()

    def func(self, params, f):
        Rs, Rp1, Cp1, Rp2, Cp2 = params
        Z = series(Rs, parallel(Rp1, imp_C(f, Cp1),
                   parallel(Rp2, imp_C(f, Cp2))))
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
        Z = series(Rs, parallel(Rp1, imp_C(f, Cp1),
                   parallel(Rp2, imp_C(f, Cp2),
                   parallel(Rp3, imp_C(f, Cp3)))))
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
        Z = series(Rs, parallel(Rp, series(imp_C(f, Cp), imp_W(f, W))))
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
