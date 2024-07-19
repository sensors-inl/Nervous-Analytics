import os
import random

import numpy as np
from tensorflow.keras.models import load_model

from nervous_analytics import PredictionTracker
from nervous_analytics.modeling import get_custom_loss_items, ModelInference


def test_model_inference():
    base_path = os.path.dirname(os.path.abspath(__file__))
    package_directory = os.path.dirname(base_path)
    path = 'nervous_analytics/template/trained_models'
    cnn_path = os.path.join(package_directory, path, 'U_net_ECG_CNN.keras')
    lstm_path = os.path.join(package_directory, path, 'U_net_ECG_LSTM.keras')
    cnn_model = load_model(cnn_path, custom_objects=get_custom_loss_items())
    lstm_model = load_model(lstm_path, custom_objects=get_custom_loss_items())

    models_dico = {
        'CNN': cnn_model,
        'LSTM': lstm_model
    }

    def processing_pred_dico_func(pred_dico):
        cnn_pred = pred_dico['CNN']
        lstm_pred = pred_dico['LSTM']
        avg_pred = np.mean([cnn_pred, lstm_pred], axis=0)
        return 'Avg', avg_pred

    model_inference = ModelInference(
        models_dico=models_dico,
        processing_pred_dico_func=processing_pred_dico_func
    )

    assert isinstance(model_inference, ModelInference)
    input = [random.uniform(-10, 10) for _ in range(5000)]
    tracker = PredictionTracker(input)
    model_inference.infer(tracker)
    output = tracker.get_process_output(step_index=-1)
    assert len(output) == len(input)
    assert max(output) <= 1 and min(output) >= 0