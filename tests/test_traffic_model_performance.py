import numpy as np
import pytest


MODEL_NAMES = ("GRU", "LSTM")
ERROR_METRICS = ("mae", "mse", "rmse")


def _calculate_metrics(actual, predicted):
    errors = np.asarray(predicted) - np.asarray(actual)
    mse = float(np.mean(np.square(errors)))
    return {
        "mae": float(np.mean(np.abs(errors))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_predictions_have_expected_shape_and_finite_values(
    model_name, model_evaluation
):
    assert model_evaluation["input_shapes"][model_name] == (4, 41)
    assert model_evaluation["prediction_shapes"][model_name] == (
        len(model_evaluation["y_test"]),
        1,
    )
    assert len(model_evaluation["predictions"][model_name]) == len(
        model_evaluation["actual"]
    )
    assert np.isfinite(model_evaluation["predictions"][model_name]).all()


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_predictions_are_nonnegative_traffic_volumes(
    model_name, model_evaluation
):
    predictions = model_evaluation["predictions"][model_name]
    assert np.all(predictions >= 0)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_beats_naive_persistence_on_heldout_accuracy(
    model_name, model_evaluation
):
    actual = model_evaluation["actual"]
    model_metrics = _calculate_metrics(
        actual, model_evaluation["predictions"][model_name]
    )
    baseline_metrics = _calculate_metrics(actual, model_evaluation["naive"])
    worse_metrics = {
        metric: (model_metrics[metric], baseline_metrics[metric])
        for metric in ERROR_METRICS
        if model_metrics[metric] >= baseline_metrics[metric]
    }
    assert not worse_metrics


def test_saved_train_and_test_targets_are_separated_by_the_chronological_cutoff(
    model_evaluation,
):
    train_times = model_evaluation["train_target_times"]
    test_times = model_evaluation["test_target_times"]
    cutoff = model_evaluation["cutoff"]

    assert len(train_times) > 0
    assert len(test_times) == len(model_evaluation["y_test"])
    assert train_times.max() < cutoff
    assert test_times.min() >= cutoff


def test_gru_and_lstm_comparison_uses_the_same_heldout_targets(model_evaluation):
    actual = model_evaluation["actual"]
    assert len(model_evaluation["predictions"]["GRU"]) == len(actual)
    assert len(model_evaluation["predictions"]["LSTM"]) == len(actual)

    for model_name in MODEL_NAMES:
        metrics = _calculate_metrics(
            actual, model_evaluation["predictions"][model_name]
        )
        assert set(metrics) == set(ERROR_METRICS)
        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_peak_and_off_peak_metrics_use_timestamped_predictions(
    model_name, model_evaluation
):
    peak_mask = model_evaluation["peak_mask"]
    actual = model_evaluation["actual"]
    prediction = model_evaluation["predictions"][model_name]
    baseline = model_evaluation["naive"]

    assert peak_mask.any()
    assert (~peak_mask).any()
    for period_mask in (peak_mask, ~peak_mask):
        period_metrics = _calculate_metrics(actual[period_mask], prediction[period_mask])
        baseline_metrics = _calculate_metrics(actual[period_mask], baseline[period_mask])
        assert set(period_metrics) == set(ERROR_METRICS)
        assert all(np.isfinite(value) for value in period_metrics.values())
        assert all(np.isfinite(value) for value in baseline_metrics.values())


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_reported_accuracy_metrics_match_actual_predictions(
    model_name, model_evaluation
):
    actual = model_evaluation["actual"]
    prediction = model_evaluation["predictions"][model_name]
    errors = prediction - actual
    mse = np.mean(np.square(errors))
    metrics = _calculate_metrics(actual, prediction)

    assert metrics["mae"] == pytest.approx(np.mean(np.abs(errors)))
    assert metrics["mse"] == pytest.approx(mse)
    assert metrics["rmse"] == pytest.approx(np.sqrt(mse))
