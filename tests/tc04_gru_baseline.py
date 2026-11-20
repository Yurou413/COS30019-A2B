"""Standalone numerical model evaluation: TC04."""
from evaluation_common import load_evaluation, display, metrics, MODELS
import numpy as np
import pandas as pd

def main():
    actual, predictions, sites, timestamps, latency, M, base = load_evaluation()
    name='GRU'
    improvement=100*(base['MAE']-M[name]['MAE'])/base['MAE'] if base['MAE'] else np.nan
    display(4,'GRU vs Persistence Baseline',{'model_MAE':M[name]['MAE'],'baseline_MAE':base['MAE'],'improvement_pct':improvement,'model_RMSE':M[name]['RMSE'],'baseline_RMSE':base['RMSE']})

if __name__ == "__main__":
    main()
