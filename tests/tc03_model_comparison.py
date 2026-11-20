"""Standalone numerical model evaluation: TC03."""
from evaluation_common import load_evaluation, display, metrics, MODELS
import numpy as np
import pandas as pd

def main():
    actual, predictions, sites, timestamps, latency, M, base = load_evaluation()
    display(3, 'GRU vs LSTM Accuracy', pd.DataFrame([{'Model':n, **{k:M[n][k] for k in ['MAE','RMSE','R2']}} for n in MODELS]))

if __name__ == "__main__":
    main()
