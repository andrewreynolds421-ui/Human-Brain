#!/usr/bin/env python3
"""Build a small fake aisynphys-style sqlite database (same table and column
names as the real release) so the extraction script can be tested without
the real data.  Values are random and meaningless.

    python tests/make_mock_synphys.py /tmp/mock_synphys.sqlite
"""
import random
import sqlite3
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mock_synphys.sqlite"
random.seed(0)
con = sqlite3.connect(path)
c = con.cursor()
c.executescript("""
DROP TABLE IF EXISTS slice; DROP TABLE IF EXISTS experiment; DROP TABLE IF EXISTS cell;
DROP TABLE IF EXISTS pair; DROP TABLE IF EXISTS synapse; DROP TABLE IF EXISTS dynamics;
DROP TABLE IF EXISTS synapse_model; DROP TABLE IF EXISTS intrinsic; DROP TABLE IF EXISTS cortical_cell_location;
CREATE TABLE slice (id INTEGER PRIMARY KEY, species TEXT);
CREATE TABLE experiment (id INTEGER PRIMARY KEY, slice_id INTEGER, project_name TEXT);
CREATE TABLE cell (id INTEGER PRIMARY KEY, experiment_id INTEGER, cell_class TEXT, cell_class_nonsynaptic TEXT, target_layer TEXT, cre_type TEXT);
CREATE TABLE cortical_cell_location (id INTEGER PRIMARY KEY, cell_id INTEGER, cortical_layer TEXT, distance_to_pia REAL, fractional_depth REAL);
CREATE TABLE pair (id INTEGER PRIMARY KEY, experiment_id INTEGER, pre_cell_id INTEGER, post_cell_id INTEGER, has_synapse INTEGER, has_electrical INTEGER,
                   n_ex_test_spikes INTEGER, n_in_test_spikes INTEGER, distance REAL, lateral_distance REAL, vertical_distance REAL);
CREATE TABLE synapse (id INTEGER PRIMARY KEY, pair_id INTEGER, synapse_type TEXT, latency REAL, psp_amplitude REAL, psp_rise_time REAL, psp_decay_tau REAL, psc_amplitude REAL);
CREATE TABLE dynamics (id INTEGER PRIMARY KEY, pair_id INTEGER, qc_pass INTEGER, n_source_events INTEGER, paired_pulse_ratio_50hz REAL, stp_initial_50hz REAL,
                       stp_induction_50hz REAL, stp_recovery_250ms REAL, variability_resting_state REAL);
CREATE TABLE synapse_model (id INTEGER PRIMARY KEY, pair_id INTEGER, n_source_events INTEGER, ml_n_release_sites REAL, ml_base_release_probability REAL,
                            ml_mini_amplitude REAL, ml_depression_amount REAL, ml_depression_tau REAL, ml_facilitation_amount REAL, ml_facilitation_tau REAL, ml_strength REAL);
CREATE TABLE intrinsic (id INTEGER PRIMARY KEY, cell_id INTEGER, tau REAL, input_resistance REAL, input_resistance_ss REAL, rheobase REAL, ap_threshold_v REAL,
                        sag REAL, adaptation_index REAL, ap_width REAL);
""")
c.execute("INSERT INTO slice VALUES (1, 'human')")
c.execute("INSERT INTO slice VALUES (2, 'mouse')")
cell_id = 0
pair_id = 0
for exp in range(1, 41):
    species_slice = 1 if exp <= 30 else 2
    c.execute("INSERT INTO experiment VALUES (?, ?, ?)", (exp, species_slice, "human coarse matrix" if species_slice == 1 else "mouse V1 coarse matrix"))
    layer = random.choice(["2", "3", "4", "5", "6"])
    ids = []
    for k in range(8):
        cell_id += 1
        cls = random.choice(["ex", "ex", "ex", "in"])
        c.execute("INSERT INTO cell VALUES (?, ?, ?, ?, ?, ?)", (cell_id, exp, cls, cls, layer, ""))
        c.execute("INSERT INTO cortical_cell_location VALUES (?, ?, ?, ?, ?)", (cell_id, cell_id, layer, random.uniform(200e-6, 2000e-6), random.random()))
        c.execute("INSERT INTO intrinsic VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (cell_id, cell_id, random.uniform(10e-3, 30e-3), random.uniform(80e6, 300e6), random.uniform(60e6, 250e6),
                   random.uniform(50e-12, 300e-12), random.uniform(-45e-3, -35e-3), random.uniform(0.8, 1.0), random.uniform(0.5, 1.0), random.uniform(0.5e-3, 1.5e-3)))
        ids.append((cell_id, cls))
    for a, cls_a in ids:
        for b, cls_b in ids:
            if a == b:
                continue
            pair_id += 1
            connected = random.random() < 0.1
            c.execute("INSERT INTO pair VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (pair_id, exp, a, b, int(connected), int(random.random() < 0.02), random.randint(0, 60), random.randint(0, 60),
                       random.uniform(20e-6, 250e-6), random.uniform(10e-6, 200e-6), random.uniform(0, 100e-6)))
            if connected:
                c.execute("INSERT INTO synapse VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                          (pair_id, pair_id, cls_a, random.uniform(0.8e-3, 2.5e-3), random.uniform(0.1e-3, 2e-3) * (1 if cls_a == "ex" else -1),
                           random.uniform(1e-3, 5e-3), random.uniform(5e-3, 40e-3), random.uniform(10e-12, 200e-12) * (1 if cls_a == "ex" else -1)))
                c.execute("INSERT INTO dynamics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (pair_id, pair_id, 1, random.randint(20, 200), random.uniform(0.4, 1.6), random.uniform(-0.5, 0.5),
                           random.uniform(-0.7, 0.5), random.uniform(-0.3, 0.3), random.uniform(0.2, 1.0)))
                c.execute("INSERT INTO synapse_model VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (pair_id, pair_id, random.randint(20, 200), random.uniform(1, 20), random.uniform(0.05, 0.6), random.uniform(0.1e-3, 1e-3),
                           random.uniform(0, 1), random.uniform(0.05, 1.0), random.uniform(0, 0.5), random.uniform(0.05, 0.5), random.uniform(0.1e-3, 3e-3)))
con.commit()
print(f"mock database written to {path}: {cell_id} cells, {pair_id} pairs")
