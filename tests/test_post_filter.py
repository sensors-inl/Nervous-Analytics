import random

import numpy as np

from nervous_analytics import postprocessing as post


def test_ECGKnowledge():
    input = [random.randint(0, 4999) for _ in range(15)]
    input = sorted(input)
    process = post.ECGKnowledge()
    assert isinstance(process, post.ECGKnowledge)
    output = process.filter(input)
    assert isinstance(output, list)
    assert len(output) == len(input) - 1


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
    ratio2 = np.count_nonzero(output1) / len(output1)  # TODO
    print(ratio1, ratio2)
    assert 0.4 < ratio1 < 0.6
    assert 0.4 < ratio2 < 0.6
