"""Standalone numerical model evaluation: TC09."""
from evaluation_common import load_evaluation, display, metrics, MODELS
import numpy as np
import pandas as pd

def main():
    actual, predictions, sites, timestamps, latency, M, base = load_evaluation()
    if timestamps is None or pd.isna(timestamps).any():
        display(9,'LSTM Off-Peak Performance',None)
    else:
        hours=np.asarray(timestamps.hour)+np.asarray(timestamps.minute)/60
        peak=((hours>=7)&(hours<9))|((hours>=16)&(hours<18))
        mask=~peak
        result=metrics(actual[mask],predictions['LSTM'][mask]) if mask.any() else None
        display(9,'LSTM Off-Peak Performance',None if result is None else {k:result[k] for k in ['n','MAE','RMSE']})

if __name__ == "__main__":
    main()
