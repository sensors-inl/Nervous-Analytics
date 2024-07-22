import inspect
from typing import Dict, Type

import tensorflow as tf
import tensorflow.keras.losses as losses


def get_loss_function(loss_function: str, **kwargs) -> losses.Loss:
    try:
        # Single-class classification
        if loss_function == "binary_crossentropy":
            return losses.BinaryCrossentropy(**kwargs)
        elif loss_function == "binary_focal_crossentropy":
            return losses.BinaryFocalCrossentropy(**kwargs)
        elif loss_function == "dice":
            return losses.Dice(**kwargs)
        elif loss_function == "binary_weighted_crossentropy":
            return BinaryWeightedCrossentropy(**kwargs)
        elif loss_function == "binary_balanced_crossentropy":
            return BinaryBalancedCrossentropy(**kwargs)

        # Multi-class classification
        elif loss_function == "categorical_crossentropy":
            return losses.CategoricalCrossentropy(**kwargs)
        elif loss_function == "categorical_focal_crossentropy":
            return losses.CategoricalFocalCrossentropy(**kwargs)

    except TypeError as ex:
        raise ValueError(f"Invalid parameters for loss function '{loss_function}': {ex}")
    raise ValueError(f"Loss function '{loss_function}' not recognized")


def get_custom_loss_items() -> Dict[str, Type[losses.Loss]]:
    current_module = __name__
    classes = inspect.getmembers(__import__(current_module), inspect.isclass)

    custom_loss_classes = {
        name: cls for name, cls in classes if issubclass(cls, losses.Loss) and cls.__module__ == current_module
    }
    return custom_loss_classes


class BinaryWeightedCrossentropy(losses.Loss):
    def __init__(
        self,
        weight_one=0.75,
        weight_zero=0.25,
        name="binary_weighted_crossentropy",
        reduction=losses.Reduction.NONE,
        **kwargs,
    ):
        """:param weight_one: Error weight for a label of 1.
        The larger weight_one is, the greater the error for a prediction close to 0.
        :param weight_zero: Error weight for a label of 0.
        The larger weight_zero is, the greater the error for a prediction close to 1.
        """
        super().__init__(name=name, reduction=reduction)
        self.weight_one = weight_one
        self.weight_zero = weight_zero

    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred, axis=-1)
        weights = y_true * self.weight_one + (1 - y_true) * self.weight_zero
        bce = losses.binary_crossentropy(y_true, y_pred)
        weights = tf.transpose(weights)
        weighted_bce = weights * bce
        return tf.reduce_mean(weighted_bce)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "weight_one": self.weight_one,
                "weight_zero": self.weight_zero,
            }
        )
        return config


# TensorFlow's BinaryFocalCrossentropy is also able to integrate a balancing parameter (alpha)
class BinaryBalancedCrossentropy(losses.Loss):
    def __init__(
        self,
        balance_ratio=0.75,
        name="binary_balanced_crossentropy",
        reduction=losses.Reduction.NONE,
        **kwargs,
    ):
        """:param balance_ratio: Proportion of error weight for a label of 1.
        The remaining proportion is for a label of 0.
        """
        super().__init__(name=name, reduction=reduction)
        self.balance_ratio = balance_ratio

    def call(self, y_true, y_pred):
        bwce = BinaryWeightedCrossentropy(weight_one=self.balance_ratio, weight_zero=1 - self.balance_ratio)
        return bwce(y_true, y_pred)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "balance_ratio": self.balance_ratio,
            }
        )
        return config
