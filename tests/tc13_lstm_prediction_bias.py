"""Standalone numerical model evaluation: TC13."""
from evaluation_common import load_evaluation, display, metrics, MODELS
import numpy as np
import pandas as pd

def main():
    actual, predictions, sites, timestamps, latency, M, base = load_evaluation()
    display(13,'LSTM Prediction Bias',{k:M['LSTM'][k] for k in ['mean_error','over_pct','under_pct','negative_n','min_pred']})

if __name__ == "__main__":
    main()
