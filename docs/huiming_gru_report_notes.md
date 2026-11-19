# Huiming - GRU Report Notes

These notes are for the GRU contribution to the team report. The comparison
and overall conclusion must be updated after the LSTM and third model are
evaluated on the same shared test set.

## GRU Method

The GRU model predicts the next 15-minute traffic-volume observation from the
four immediately preceding 15-minute readings. The shared preprocessing
pipeline covers 40 SCATS sites. It scales traffic readings with a separate
MinMaxScaler for each site and adds a one-hot site identifier to the input
features. The train/test split is chronological, with the training cutoff at
2006-10-25 19:00.

The model selection compared 32-unit and 64-unit GRU layers using the same
chronological validation subset. Both candidates used a maximum of 30 epochs,
batch size 32, Adam, mean squared error, and early stopping with patience 5.
The 64-unit candidate had the lower validation MSE (0.003689 versus 0.004028),
so it was selected. The selected network uses `tanh`, 20% dropout, a 32-unit
ReLU dense layer, and a one-unit output layer. The final test set was not used
to choose the GRU size.

## Current GRU Results

The model was trained on 93,692 sequences and evaluated on 22,280 held-out
sequences. The reported inference latency is the average time per test sample
after one warm-up prediction.

| Measure | Normalized target scale | Original traffic-volume scale |
|---|---:|---:|
| MSE | 0.003723 | 3221.937435 |
| RMSE | 0.061020 | 56.762113 |
| MAE | 0.046922 | 41.175203 |

- Total model-selection training time: 84.58 seconds across both candidates
- Average batch-prediction time: 0.0302 milliseconds per sample (measured over the held-out set after one warm-up prediction)

| Candidate GRU units | Epochs before early stopping | Best validation MSE |
|---:|---:|---:|
| 32 | 12 | 0.004028 |
| 64 (selected) | 11 | 0.003689 |

Use the original-scale metrics when explaining errors in the dataset's traffic
volume units. Use normalized metrics only when comparing models that used the
same preprocessing and test examples. Do not compare these figures directly
with the earlier site-2000-only run; that run used a different input shape and
dataset split.

## Conclusion Draft

The updated GRU pipeline trained successfully on the shared chronological
dataset for 40 SCATS sites and produced one-step-ahead traffic-volume
predictions. On the held-out sequences, the selected 64-unit model achieved an
RMSE of 56.76 and an MAE of 41.18 in the original traffic-volume scale. These figures
describe the GRU alone; they do not establish that it is better than the LSTM
or the third model. The final report should make that comparison only after all
models use the same site coverage, split, and scaling method. The data covers
October 2006, so performance on other periods has not been established.

## Resources and Acknowledgements

- The course-provided SCATS October 2006 workbook supplies the historical
  traffic readings used to train the GRU.
- The course-provided VicRoads SCATS site listing and traffic-count location
  file provide site identifiers and location context for the wider system.
- The course-provided traffic-flow-to-travel-time guide informs the group's
  downstream route-cost conversion. The GRU itself predicts traffic volume;
  the route integration must apply the conversion after prediction.
- The course-provided LSTM/GRU example repository and regression-tree example
  notebooks are reference material. They are not part of the team's submitted
  implementation.

## References to Include

1. Swinburne University of Technology. *COS30019 Assignment 2B: Machine
   Learning and Software Integration*, Semester 2, 2026.
2. Swinburne University of Technology. *Scats Data October 2006* [course
   dataset].
3. Swinburne University of Technology. *SCATS Site Listing Spreadsheet -
   VicRoads* [course dataset].
4. Swinburne University of Technology. *Traffic Flow to Travel Time Conversion
   v1.0* [course resource].
5. hyswin. *Example LSTM and GRU for COS30019 A2B* [course-linked code
   example]. GitHub: https://github.com/hyswin/Example-LSTM-and-GRU-for-A2B-COS30019-

