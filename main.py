from utils import fit_diode, write_stats, plot_total_capacitance, plot_series_resistance, plot_parallel_resistances, plot_characteristic, plot_all_char
from models import R_RC, R_RC_RC, R_RC_RC_RC
import time
import logging

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/main.log', filemode='w'
)


def main():
    sigma = 0.2
    convergence_threshold = 50

    # models = [R_RC(), R_RC_RC(), R_RC_RC_RC(), R_RCW()]
    # models = [R_RC(), R_RC_RC(), R_RC_RC_RC()]
    models = [R_RC_RC()]

    model_names = [model.name for model in models]

    BIAS_SCAN = "BIAS_SCAN"
    CHARACTERISTIC = "CHARACTERISTIC"
    date = "2024-01-15"
    # diodes = ["1N4001", "1N4002", "1N4003", "1N4007"]
    diodes = ["1N4007"]

    b_low, b_high = 500, 500
    biases = [b_low, b_high]

    plot_all_char(diodes, date, CHARACTERISTIC)
    # stats: [diode, bias, model, cost, *params]
    for diode in diodes:
        plot_characteristic(diode, date, CHARACTERISTIC)
        stats, failures = fit_diode(
            diode,
            date,
            BIAS_SCAN,
            models,
            biases,
            sigma=sigma,
            convergence_threshold=convergence_threshold
        )
        for key, val in failures.items():
            logging.error(f"Fit failed for: {key} with message: {val}")
        for model in model_names:
            plot_total_capacitance(stats, diode, model_name=model)
            plot_series_resistance(stats, diode, model_name=model)
            plot_parallel_resistances(stats, diode, model_name=model)
        write_stats(stats, f"stats/{diode}_{date}_{BIAS_SCAN}.csv")
    print("Done!")
    exit(0)


if __name__ == '__main__':
    main()
