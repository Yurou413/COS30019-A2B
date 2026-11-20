"""Standalone numerical model evaluation: TC01."""
from evaluation_common import load_evaluation, display, metrics, MODELS
import numpy as np
import pandas as pd

def main():
    actual, predictions, sites, timestamps, latency, M, base = load_evaluation()
    display(1, 'GRU Prediction Accuracy', {k:M['GRU'][k] for k in ['n','MAE','MSE','RMSE','R2']})

if __name__ == "__main__":
    main()
