import numpy as np

from .preprocess import PreFilter


class Normalizer(PreFilter):
    """Normalize data to [0, 1] or [-1, 1]"""

    def __init__(self, norm_min, norm_max, center_to_zero=False):
        super().__init__()
        self.offset = norm_min
        self.scale = norm_max - norm_min
        self.center_to_zero = center_to_zero

    def filter(self, data, **kwargs):
        # Normalize data (peak to peak at 0 to 1)
        data = np.subtract(data, self.offset)
        data = np.divide(data, self.scale)

        # Center to zero (peak to peak at -1 to 1)
        if self.center_to_zero:
            data = np.multiply(data, 2)
            data = np.subtract(data, 1)

        # Saturate data
        return np.clip(data, -1 if self.center_to_zero else 0, 1)
