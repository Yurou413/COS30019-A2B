"""Standalone numerical model evaluation: TC10."""
from evaluation_common import load_evaluation, display, metrics, MODELS
import numpy as np
import pandas as pd

def main():
    actual, predictions, sites, timestamps, latency, M, base = load_evaluation()
    display(10,'GRU Error Distribution',{k:M['GRU'][k] for k in ['mean_error','median_abs_error','p95_abs_error','max_abs_error']})

if __name__ == "__main__":
    main()
