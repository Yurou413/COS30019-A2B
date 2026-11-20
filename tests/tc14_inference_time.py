"""Standalone numerical model evaluation: TC14."""
from evaluation_common import load_evaluation, display, metrics, MODELS
import numpy as np
import pandas as pd

def main():
    actual, predictions, sites, timestamps, latency, M, base = load_evaluation()
    display(14,'GRU vs LSTM Inference Time (ms/sample, batch=32)',pd.DataFrame([{'Model':name,'mean_ms':v.mean(),'median_ms':np.median(v),'p95_ms':np.percentile(v,95)} for name,v in latency.items()]))

if __name__ == "__main__":
    main()
