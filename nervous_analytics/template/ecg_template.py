import os

import numpy as np
from tensorflow.keras.models import load_model

from nervous_analytics.modeling import get_custom_loss_items
from nervous_analytics.modeling.model_inference import ModelInference
from nervous_analytics.physio_prediction import PhysioPrediction
from nervous_analytics.postprocessing import (
    ECGKnowledge,
    PostProcess,
    PredictionSegmenter,
    Threshold,
)
from nervous_analytics.preprocessing import (
    FreqFilter,
    Normalizer,
    PreProcess,
    WaveDecomposer,
)


def get_ecg_template() -> PhysioPrediction:
    preprocess = PreProcess(
        [
            FreqFilter(fs=1024, cutoff=[0.5, 30], order=4, filter_type="bandpass"),
            WaveDecomposer(
                wavelet="db3",
                level=5,
                selected_channel=[0, 1],
                threshold_ratio=0.3,
            ),
            Normalizer(norm_min=-1.0, norm_max=2.0),
        ]
    )

    postprocess = PostProcess(
        [
            Threshold(non_zero_data_rate=0.02, output_thresholded_range=[0.05, 1]),
            PredictionSegmenter(),
            ECGKnowledge(
                max_derivation=0.3,
                ecg_sampling_rate=150,
                hr_range=[0, 300],
            ),
        ]
    )

    # Necessary to use the absolute path for tests
    base_path = os.path.dirname(os.path.abspath(__file__))
    cnn_path = os.path.join(base_path, "trained_models", "U_net_ECG_CNN.keras")
    lstm_path = os.path.join(base_path, "trained_models", "U_net_ECG_LSTM.keras")
    cnn_model = load_model(cnn_path, custom_objects=get_custom_loss_items())
    lstm_model = load_model(lstm_path, custom_objects=get_custom_loss_items())

    models_dico = {
        "CNN": cnn_model,
        "LSTM": lstm_model,
    }

    def processing_pred_dico_func(pred_dico):
        cnn_pred = pred_dico["CNN"]
        lstm_pred = pred_dico["LSTM"]
        avg_pred = np.mean([cnn_pred, lstm_pred], axis=0)
        return "Avg", avg_pred

    model_inference = ModelInference(
        models_dico=models_dico,
        processing_pred_dico_func=processing_pred_dico_func,
    )

    return PhysioPrediction(
        preprocess=preprocess,
        model_inference=model_inference,
        postprocess=postprocess,
    )


get_ecg_template()
