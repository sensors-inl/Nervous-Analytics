import random

from nervous_analytics import preprocessing as pre


def check_type_and_length(process, class_type):
    assert isinstance(process, class_type)
    input = [random.uniform(-10, 10) for _ in range(5000)]
    output = process.filter(input)
    assert len(output) == len(input)


def test_Differentiator():
    check_type_and_length(pre.Differentiator(diff_n=1), pre.Differentiator)


def test_FallSmoother():
    check_type_and_length(pre.FallSmoother(critical_fall_ratio=0.3), pre.FallSmoother)


def test_FreqFilter():
    check_type_and_length(pre.FreqFilter(fs=8, cutoff=[1], order=5, filter_type="low"), pre.FreqFilter)


def test_Normalizer():
    check_type_and_length(pre.Normalizer(norm_min=0, norm_max=1), pre.Normalizer)


def test_PeakCleaner():
    check_type_and_length(
        pre.PeakCleaner(max_ratio=10, avg_min_ratio=20, multi_point=False, max_points=8), pre.PeakCleaner
    )


def test_WaveDecomposer():
    check_type_and_length(
        pre.WaveDecomposer(wavelet="db3", level=5, selected_channel=[0, 1], threshold_ratio=0.3), pre.WaveDecomposer
    )
