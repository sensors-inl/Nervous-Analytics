import random

from nervous_analytics import PhysioPrediction, PredictionTracker, template


def check_process(process):
    assert isinstance(process, PhysioPrediction)
    input = [random.uniform(-10, 10) for _ in range(5000)]
    tracker = process.predict(input)
    assert isinstance(tracker, PredictionTracker)


def test_ecg_template():
    process = template.get_ecg_template()
    check_process(process)


def test_eda_template():
    process = template.get_eda_template()
    check_process(process)
