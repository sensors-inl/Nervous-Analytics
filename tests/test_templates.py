import random

from nervous_analytics import template, PhysioPrediction, PredictionTracker


def get_random_signal():
    return [random.uniform(-10, 10) for _ in range(5000)]


def test_ecg_template():
    process = template.get_ecg_template()
    assert isinstance(process, PhysioPrediction)
    input = [random.uniform(-10, 10) for _ in range(5000)]
    tracker = process.predict(input)
    assert isinstance(tracker, PredictionTracker)


def test_eda_template():
    process = template.get_eda_template()
    assert isinstance(process, PhysioPrediction)
    input = [random.uniform(-10, 10) for _ in range(5000)]
    tracker = process.predict(input)
    assert isinstance(tracker, PredictionTracker)
