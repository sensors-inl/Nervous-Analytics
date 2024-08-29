import os

from tensorflow.keras import Model
from tensorflow.keras.models import load_model

from nervous_analytics.modeling import get_custom_loss_items


def check_model_import(model_name):
    base_path = os.path.dirname(os.path.abspath(__file__))
    package_directory = os.path.dirname(base_path)
    print('')
    path = os.path.join(
        package_directory,
        "nervous_analytics/template/trained_models",
        model_name,
    )
    model = load_model(path, custom_objects=get_custom_loss_items())
    assert isinstance(model, Model)


def test_cnn_model_import():
    check_model_import("U_net_ECG_CNN.keras")


def test_lstm_model_import():
    check_model_import("U_net_ECG_LSTM.keras")
