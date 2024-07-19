import os

import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Sequential, load_model

from nervous_analytics.modeling import get_custom_loss_items, get_loss_function


def test_custom_loss_items():
    items = get_custom_loss_items()
    assert isinstance(items, dict)
    assert all(isinstance(loss, type) for loss in items.values())
    assert all(issubclass(loss, Loss) for loss in items.values())


def check_simple_model_load(loss_fn):
    model = Sequential(
        [
            Input(shape=(10,)),  # Utilisation de Input
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss=loss_fn)
    model.save("simple_model.keras")
    reconstructed_model = load_model(
        "simple_model.keras", custom_objects=get_custom_loss_items()
    )
    reconstructed_model.compile(optimizer="adam", loss=loss_fn)

    os.remove("simple_model.keras")
    test_input = np.random.uniform(-10, 10, (5000, 10))

    try:
        np.testing.assert_allclose(
            model.predict(test_input), reconstructed_model.predict(test_input)
        )
    except AssertionError:
        return False
    return True


def test_loss_binary_crossentropy():
    loss_fn = get_loss_function("binary_crossentropy")
    assert isinstance(loss_fn, Loss)
    assert check_simple_model_load(loss_fn)


def test_loss_binary_focal_crossentropy():
    loss_fn = get_loss_function("binary_focal_crossentropy")
    assert isinstance(loss_fn, Loss)
    assert check_simple_model_load(loss_fn)


def test_loss_dice():
    loss_fn = get_loss_function("dice")
    assert isinstance(loss_fn, Loss)
    assert check_simple_model_load(loss_fn)


def test_loss_binary_weighted_crossentropy():
    loss_fn = get_loss_function(
        "binary_weighted_crossentropy", weight_one=0.7, weight_zero=0.1
    )
    assert isinstance(loss_fn, Loss)
    assert check_simple_model_load(loss_fn)


def test_loss_binary_balanced_crossentropy():
    loss_fn = get_loss_function(
        "binary_balanced_crossentropy", balance_ratio=0.3
    )
    assert isinstance(loss_fn, Loss)
    assert check_simple_model_load(loss_fn)


def test_loss_categorical_crossentropy():
    loss_fn = get_loss_function("categorical_crossentropy")
    assert isinstance(loss_fn, Loss)
    assert check_simple_model_load(loss_fn)


def test_loss_categorical_focal_crossentropy():
    loss_fn = get_loss_function("categorical_focal_crossentropy")
    assert isinstance(loss_fn, Loss)
    assert check_simple_model_load(loss_fn)
