"""Assignment 03: summarize a telemetry ward's systolic readings.

Run from the assignment directory with the project environment active:

    python3 analysis.py
"""

import numpy as np


def load_readings(filename):
    """Return (patient_ids, monitors, hour_columns, readings) from the supplied CSV.

    readings is a 2D array of integers: one row per patient, one column per
    monitored hour, in the order the header lists them.
    """
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    header = lines[0].strip().split(",")
    rows = [line.strip().split(",") for line in lines[1:] if line.strip()]

    patient_ids = np.array([row[0] for row in rows])
    monitors = np.array([row[1] for row in rows])
    hour_columns = np.array(header[2:])
    readings = np.array([row[2:] for row in rows]).astype(int)
    return patient_ids, monitors, hour_columns, readings


def main():
    patient_ids, monitors, hour_columns, readings = load_readings("data/bp_readings.csv")
    print(f"Loaded {readings.shape[0]} patients x {readings.shape[1]} hours")

    # answer each question in the README's summary table with NumPy.
    patients = readings.shape[0]
    total_readings = readings.size
    mean_sbp = readings.mean()
    sd_sbp = readings.std()
    min_sbp = readings.min()
    max_sbp = readings.max()

    # write one "key: value" line per answer to output/vitals_summary.txt.
    
    patient_means = readings.mean(axis = 1)
    stage2_patients = (patient_means >= 140).sum()
    top = patient_means.argmax()
    highest_patient = patient_ids[top]
    highest_patient_mean = patient_means[top]

    #
    hour_means = readings.mean(axis = 0)
    peak = hour_means.argmax()
    peak_hour_column = hour_columns[peak]
    peak_hour_mean = hour_means[peak]

    #
    monitor_names = sorted(set(monitors))           # ['M01', ..., 'M06']
    monitor_averages = []
    for name in monitor_names:
        mask = monitors == name                     # True for that monitor's patients
        monitor_averages.append(patient_means[mask].mean())
    monitor_averages = np.array(monitor_averages)
    high_monitor = monitor_names[monitor_averages.argmax()]

    on_high = monitors == high_monitor
    other_means = patient_means[~on_high]           # ~ flips True/False: everyone else
    monitor_offset = patient_means[on_high].mean() - other_means.mean()
    stage2_other_monitors = (other_means >= 140).sum()

    with open("output/vitals_summary.txt", "w") as file:
        file.write(f"patients: {patients}\n")
        file.write(f"readings: {total_readings}\n")
        file.write(f"mean_sbp: {mean_sbp:.2f}\n")
        file.write(f"sd_sbp: {sd_sbp:.2f}\n")
        file.write(f"min_sbp: {min_sbp}\n")
        file.write(f"max_sbp: {max_sbp}\n")
        file.write(f"stage2_patients: {stage2_patients}\n")
        file.write(f"highest_patient: {highest_patient}\n")
        file.write(f"highest_patient_mean: {highest_patient_mean:.2f}\n")
        file.write(f"peak_hour_column: {peak_hour_column}\n")
        file.write(f"peak_hour_mean: {peak_hour_mean:.2f}\n")
        file.write(f"high_monitor: {high_monitor}\n")
        file.write(f"monitor_offset: {monitor_offset:.2f}\n")
        file.write(f"stage2_other_monitors: {stage2_other_monitors}\n")

if __name__ == "__main__":
    main()
