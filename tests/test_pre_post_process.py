import random
from nervous_analytics import PreProcess, PostProcess, Normalizer, Threshold


def check_type_and_length(process, class_type):
    assert isinstance(process, class_type)
    input = [random.uniform(-10, 10) for _ in range(5000)]
    output = process.filter(input)
    assert len(output) == len(input)


def test_preprocess():
    preprocess = PreProcess(
        [
            Normalizer(
                norm_min=0,
                norm_max=1
            ),
            Normalizer(
                norm_min=0.25,
                norm_max=0.75
            )
        ]
    )
    check_type_and_length(preprocess, PreProcess)


def test_postprocess():
    postprocess = PostProcess(
        [
            Threshold(
                non_zero_data_rate=0.5,
            ),
            Threshold(
                non_zero_data_rate=0.25,
            )
        ]
    )
    check_type_and_length(postprocess, PostProcess)