from utils import fit_diode, write_stats, plot_total_capacitance, plot_series_resistance, plot_parallel_resistances
from models import R_RC, R_RC_RC, R_RC_RC_RC
import time
import logging

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/main.log', filemode='w'
)


def main():
    sigma = 0.1
    convergence_threshold = 200

    # models = [R_RC(), R_RC_RC(), R_RC_RC_RC(), R_RCW()]
    models = [R_RC(), R_RC_RC(), R_RC_RC_RC()]
    # models = [R_RC_RC()]

    model_names = [model.name for model in models]

    exp_type = "BIAS_SCAN"
    date = "2024-01-15"
    diodes = ["1N4001", "1N4002", "1N4003", "1N4007"]
    # diodes = ["1N4007"]

    # stats: [diode, bias, model, cost, *params]
    for diode in diodes:
        stats, failures = fit_diode(
            diode,
            date,
            exp_type,
            models,
            sigma=sigma,
            convergence_threshold=convergence_threshold
        )
        for key, val in failures.items():
            logging.error(f"Fit failed for: {key} with message: {val}")
        for model in model_names:
            plot_total_capacitance(stats, diode, model_name=model)
            plot_series_resistance(stats, diode, model_name=model)
            plot_parallel_resistances(stats, diode, model_name=model)
        write_stats(stats, f"stats/{diode}_{date}_{exp_type}.csv")
    print("Done!")


if __name__ == '__main__':
    main()
