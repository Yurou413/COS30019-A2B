"""Standalone numerical model evaluation: TC15."""
from evaluation_common import load_evaluation, display, metrics, MODELS
import numpy as np
import pandas as pd

def main():
    actual, predictions, sites, timestamps, latency, M, base = load_evaluation()
    rows=[]
    for site in np.unique(sites):
        mask=sites==site
        for name in MODELS:
            m=metrics(actual[mask],predictions[name][mask])
            rows.append({'Site':int(site),'Model':name,'Samples':m['n'],'MAE':m['MAE'],'RMSE':m['RMSE']})
    display(15,'Prediction Accuracy Across SCATS Sites',pd.DataFrame(rows))

if __name__ == "__main__":
    main()
