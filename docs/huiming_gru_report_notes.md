# Huiming - GRU Report Contribution Draft

## GRU Architecture

The GRU predicts one traffic-volume reading for the next 15-minute interval.
Each input uses the four preceding consecutive 15-minute readings, representing
one hour of recent history. The shared dataset contains 40 SCATS sites. The
traffic value is scaled with a separate MinMaxScaler for each site, and a
one-hot site identifier is appended to every time step. The resulting model
input shape is `(4, 41)`: four time steps, each with one scaled flow feature
and 40 site-identity features. The model produces one scaled traffic-volume
value.

The train/test split is chronological. The training cutoff is
`2006-10-25 19:00`; the fitted scalers are based only on observations before
that cutoff. The shared arrays contain 93,692 training sequences and 22,280
held-out test sequences.

The selected architecture has a 64-unit GRU layer with `tanh` activation, a
20% dropout layer, a 32-unit ReLU dense layer, and a one-value output layer.
Training uses Adam, mean squared error, a batch size of 32, and a chronological
20% validation split. To select the GRU size, 32-unit and 64-unit candidates
were trained with early stopping (patience 5, maximum 30 epochs). The
64-unit candidate was selected because its best validation MSE was lower:
0.003689 versus 0.004028 for 32 units. The held-out test set was not used for
this selection.

**Table 1. GRU hidden-unit candidates and validation results.**

| GRU units | Epochs before early stopping | Best validation MSE |
|---:|---:|---:|
| 32 | 12 | 0.004028 |
| 64 (selected) | 11 | 0.003689 |

## GRU Evaluation Results

The test metrics are shown in both normalized space and the original
traffic-volume scale. Original-scale MAE is useful for interpreting the
typical absolute error in the dataset's flow units. RMSE gives greater weight
to larger errors. MSE is in squared flow units.

**Table 2. GRU performance on 22,280 chronological held-out sequences.**

| Metric | Normalized scale | Original traffic-volume scale |
|---|---:|---:|
| MSE | 0.003723 | 3221.937435 |
| RMSE | 0.061020 | 56.762113 |
| MAE | 0.046922 | 41.175203 |

The two candidate fits took 84.58 seconds in total. Batched prediction over
the held-out set took approximately 0.0302 milliseconds per sample after a
warm-up prediction. This is a batch-derived average, not the latency of an
individual GUI request.

The callable interface is `predict_flow(site_id, timestamp) -> float`. It
selects the site's four consecutive readings before the requested timestamp,
applies the site's scaler and site-ID encoding, runs the saved GRU, and
converts the result back to the original flow scale. A manual interface check
for site 2000 at `2006-10-01 01:00:00` returned `160.23`. This check confirms
that the saved model and prediction function can produce an output; it is not
an additional accuracy result, and the GUI has not yet been verified to call
the function.

## Conclusion - GRU Contribution

The GRU provides one-step-ahead traffic-volume predictions from recent
temporal observations while sharing one model across 40 SCATS sites. On the
held-out set, the selected model achieved an RMSE of 56.76 and an MAE of 41.18
in the original traffic-volume scale. These results describe the GRU alone;
they do not establish that it is more accurate than the LSTM or the third
model. The final report should compare models only when they use the same
sites, chronological split, target values, and evaluation scale.

The available data covers October 2006, and the GRU input contains only four
recent readings and site identity. Performance on other periods or with
additional features has not been established. For the team-level conclusion,
also discuss whether the predictor is connected to the Part A route-search
costs and how the integrated system could be improved.

## Acknowledgements and Resources

- **SCATS Data October 2006 workbook:** supplied the historical traffic-volume
  observations used to prepare the training and test sequences.
- **SCATS Site Listing Spreadsheet - VicRoads:** supplied SCATS site identifiers
  and site information used by the wider project to identify intersections.
- **Traffic Count Locations with Longitude/Latitude CSV:** supplied geographic
  coordinates used for mapping and route-network context; coordinates are not
  input features to this GRU model.
- **IBM data_Bor.pdf:** provided a Boroondara map showing the road area and
  SCATS-site context, which helps explain the geographic scope of the system.
- **Traffic Flow to Travel Time Conversion v1.0:** describes the simplified
  flow-speed relationship and the conversion from predicted flow to segment
  travel time. The GRU predicts flow; this conversion belongs to the route-cost
  calculation after prediction.
- **LSTM/GRU example repository:** served as a learning reference for recurrent
  model structure and sequence-based prediction. The example is reference
  material; the team's model and data pipeline are implemented in this repo.
- **Regression-tree notebook and call example with `regress_traffic.py`:**
  demonstrate a third-model prediction workflow and how model error measures
  such as RMSE/NRMSE can be interpreted in a route-prediction call. These are
  examples, not files to submit as the team's implementation.

## References

Use the team's agreed citation style and retain the exact resource titles.

1. Swinburne University of Technology. *COS30019 Assignment 2 Part B: Machine
   Learning and Software Integration* [assignment brief, 2026].
2. Swinburne University of Technology. *Scats Data October 2006* [course
   dataset].
3. VicRoads. *SCATS Site Listing Spreadsheet - VicRoads* [course dataset].
4. Swinburne University of Technology. *Traffic Count Locations with Long/Lat*
   [course dataset].
5. Y. Ma. *IBM data_Bor.pdf* [Boroondara SCATS map; map dated July 2009,
   course resource].
6. Swinburne University of Technology. *Traffic Flow to Travel Time Conversion
   v1.0* [course resource].
7. hyswin. *Example LSTM and GRU for COS30019 A2B*. GitHub. Available at:
   <https://github.com/hyswin/Example-LSTM-and-GRU-for-A2B-COS30019->.
8. Swinburne University of Technology. *regression_tree_2B.ipynb* [course
   example notebook].
9. Swinburne University of Technology. *call_traffic_example.ipynb* and
   *regress_traffic.py* [course example code].

## Submission Reminders

- The assignment brief requires an 8-10 page report, excluding the cover page
  and table of contents. The report also needs instructions, implemented
  features and known missing items, testing results, model insights, a
  conclusion, resources/acknowledgements, and references.
- The course says the example notebooks and scripts are for study/adaptation;
  do not submit them as the team's solution.
- Replace this GRU-only conclusion with a team-level comparison after the
  LSTM and third-model results and integration status are available.
