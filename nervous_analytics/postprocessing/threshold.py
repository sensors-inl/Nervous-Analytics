import numpy as np

from nervous_analytics.preprocessing.preprocess import PreFilter

from .postprocess import PostFilter


class Threshold(PreFilter, PostFilter):
    """Threshold to remove low-confidence predictions.
    Possibility to choose between a fixed threshold
    or an adaptive threshold (based on the desired rate of data to keep).
    """

    def __init__(
        self,
        threshold=None,
        non_zero_data_rate=None,
        output_thresholded_range=None,
    ):
        """:param threshold: Fixed threshold to apply to the data.
        :param non_zero_data_rate: Adaptive threshold to apply to the data based on the desired
        rate of data to keep.
        :param output_thresholded_range: If not None, the non-zero predictions will be normalized
        to this range.
        """
        if (threshold is None and non_zero_data_rate is None) or (
            threshold is not None and non_zero_data_rate is not None
        ):
            raise ValueError("You must provide either a threshold or a thresholded_data_rate")

        super().__init__()
        self.threshold = threshold
        self.non_zero_data_rate = non_zero_data_rate
        self.output_thresholded_range = output_thresholded_range

    def filter(self, data, **kwargs):
        data = np.array(data)
        if self.threshold is not None:
            data = self._fixed_threshold_filter(data)
        else:
            data = self._adaptive_threshold_filter(data)

        if self.output_thresholded_range is not None:
            data = self._normalize(data)

        return data

    def _fixed_threshold_filter(self, data):
        data[data < self.threshold] = 0
        return data

    def _adaptive_threshold_filter(self, data):
        a = min(data)
        b = max(data)
        epsilon = 1e-3
        threshold = (a + b) / 2
        old_rate = -1
        same_rate_count = 0
        current_rate = 0

        while abs(current_rate - self.non_zero_data_rate) > epsilon:
            current_rate = np.average(data > threshold)

            if current_rate > self.non_zero_data_rate:
                a = threshold
            else:
                b = threshold

            threshold = (a + b) / 2

            if current_rate == old_rate:
                same_rate_count += 1
                if same_rate_count > 100:
                    break
            else:
                same_rate_count = 0

            old_rate = current_rate

        data[data < threshold] = 0
        return data

    def _normalize(self, data):
        non_zero_data = data[data > 0]
        max_value = np.max(non_zero_data)
        min_value = np.min(non_zero_data)
        up_reach = max(self.output_thresholded_range)
        down_reach = min(self.output_thresholded_range)

        scale = (up_reach - down_reach) / (max_value - min_value)
        offset = down_reach - min_value * scale

        for i in range(len(data)):
            if data[i] > 0:
                data[i] *= scale
                data[i] += offset
                if data[i] > up_reach:
                    data[i] = up_reach
                elif data[i] < down_reach:
                    data[i] = down_reach

        return data
