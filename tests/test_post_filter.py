import random

import numpy as np

from nervous_analytics import postprocessing as post


def test_ECGKnowledge():
    process = post.ECGKnowledge(
        max_derivation=0.3,
        hr_range=[40, 180],
        ecg_sampling_rate=1024,
        next_index=2500,
    )
    assert isinstance(process, post.ECGKnowledge)

    # 2000 to 4000 must be blacklisted
    input1 = [0, 1000, 2000, 4000, 4999]
    # nothing to blacklist
    input2 = [4000, 4999, 6000, 7000]
    input2 = [value - 2500 for value in input2]
    # 7000 to 9000 must be blacklisted
    input3 = [6000, 7000, 9000, 9999]
    input3 = [value - 5000 for value in input3]

    output1 = process.filter(input1)
    output2 = process.filter(input2)
    output3 = process.filter(input3)

    for input, output in zip([input1, input2, input3], [output1, output2, output3]):
        assert isinstance(output, list)
        assert len(output) == len(input) - 1

    assert not all(pair["blacklist_status"] is False for pair in output1)
    assert all(pair["blacklist_status"] is False for pair in output2)
    assert not all(pair["blacklist_status"] is False for pair in output3)


def test_EdgeCutter():
    input = [random.randint(0, 4999) for _ in range(30)]
    input = sorted(input)
    input1 = [input[i * 2] for i in range(15)]
    input2 = [input[i * 2 + 1] for i in range(15)]
    input = [input1, input2]
    process = post.EdgeCutter([1000, 4000])
    assert isinstance(process, post.EdgeCutter)
    output = process.filter(input)
    assert all(onset in input1 for onset in output[0])
    assert all(peak in input2 for peak in output[1])


def test_PredictionSegmenter():
    input = [random.uniform(0, 10) for _ in range(5000)]
    input = [0 if value < 5 else value for value in input]
    process = post.PredictionSegmenter()
    assert isinstance(process, post.PredictionSegmenter)
    output = process.filter(input)
    assert len(output) < len(input)


def test_SCRSegmenter():
    pass  # SCRSegmenter can be used only with PredictionTracker


def test_Threshold():
    input = [random.uniform(0, 10) for _ in range(5000)]
    process1 = post.Threshold(threshold=5, output_thresholded_range=[0.05, 1])
    process2 = post.Threshold(non_zero_data_rate=0.5, output_thresholded_range=[0.05, 1])
    assert isinstance(process1, post.Threshold)
    assert isinstance(process2, post.Threshold)
    output1 = process1.filter(input.copy())
    output2 = process2.filter(input.copy())
    assert max(output1) <= 1
    assert max(output2) <= 1
    assert len(output1) == len(input)
    assert len(output2) == len(input)
    ratio1 = np.count_nonzero(output1) / len(output1)
    ratio2 = np.count_nonzero(output2) / len(output2)
    print(f"ratio1: {ratio1}")
    print(f"ratio2: {ratio2}")
    assert 0.4 < ratio1 < 0.6
    assert 0.4 < ratio2 < 0.6
