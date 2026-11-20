"""Standalone numerical model evaluation: TC11."""
from evaluation_common import load_evaluation, display, metrics, MODELS
import numpy as np
import pandas as pd

def main():
    actual, predictions, sites, timestamps, latency, M, base = load_evaluation()
    display(11,'LSTM Error Distribution',{k:M['LSTM'][k] for k in ['mean_error','median_abs_error','p95_abs_error','max_abs_error']})

if __name__ == "__main__":
    main()
