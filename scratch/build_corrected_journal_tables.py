"""
scratch/build_corrected_journal_tables.py — Generate Publication-Ready Corrected Tables
"""
import json
import numpy as np

with open("scratch/session16_results.json", "r") as f:
    data = json.load(f)

baselines = data["baseline_comparison"]
ablations = data["ablation_study"]

methods = ["DCP", "CAP", "CLAHE", "Corrected Gamma", "Original CEEF", "Modified CEEF", "Proposed Method"]

print("Data loaded successfully.")
print(f"Total baseline rows: {len(baselines)}, Ablation rows: {len(ablations)}")
