"""Standalone numerical model evaluation: TC12."""
from evaluation_common import load_evaluation, display, metrics, MODELS
import numpy as np
import pandas as pd

def main():
    actual, predictions, sites, timestamps, latency, M, base = load_evaluation()
    display(12,'GRU Prediction Bias',{k:M['GRU'][k] for k in ['mean_error','over_pct','under_pct','negative_n','min_pred']})

if __name__ == "__main__":
    main()
