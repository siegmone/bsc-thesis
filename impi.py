from impedance import preprocessing
from utils import get_impedance_data
from impedance.models.circuits import CustomCircuit
import matplotlib.pyplot as plt
from impedance.visualization import plot_nyquist


f, Z, Z_mag, theta, flag = get_impedance_data("./experiments/1N4007_2024-01-15/BIAS_SCAN/500.0mV.csv")
f, Z = preprocessing.ignoreBelowX(f, Z)


circuit = 'R0-p(R1,C1)-p(R2-Wo1,C2)'
initial_guess = [.01, .01, 100, .01, .05, 100, 1]

circuit = CustomCircuit(circuit, initial_guess=initial_guess)
circuit.fit(f, Z)
Z_fit = circuit.predict(f)

fig, ax = plt.subplots()
plot_nyquist(Z, fmt='o', scale=10, ax=ax)
plot_nyquist(Z_fit, fmt='-', scale=10, ax=ax)

plt.legend(['Data', 'Fit'])
plt.show()

