from nervous_analytics.physio_prediction import PhysioPrediction
from nervous_analytics.postprocessing import (
    EdgeCutter,
    PostProcess,
    SCRSegmenter,
    Threshold,
)
from nervous_analytics.preprocessing import (
    Differentiator,
    FallSmoother,
    FreqFilter,
    PeakCleaner,
    PreProcess,
)


def get_eda_template() -> PhysioPrediction:
    preprocess = PreProcess(
        [
            PeakCleaner(),
            FreqFilter(fs=8, cutoff=[1], order=5, filter_type="low"),
            Differentiator(diff_n=1),
            Threshold(threshold=0),
            FallSmoother(critical_fall_ratio=0.2),
            Differentiator(diff_n=1),
        ]
    )

    postprocess = PostProcess(
        [
            SCRSegmenter(
                filtered_eda_process_name="FreqFilter",
                diff2_peak_threshold=0.01,
                min_elevation=0.2,
                replacing_margin=8,
                onset_offset=+2,
                peak_offset=-2,
            ),
            EdgeCutter(index_range=[25, 4999 - 25]),
        ]
    )

    return PhysioPrediction(preprocess=preprocess, model_inference=None, postprocess=postprocess)
