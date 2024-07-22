from scipy.interpolate import CubicSpline

from .preprocess import PreFilter


class FallSmoother(PreFilter):
    def __init__(self, critical_fall_ratio=0.3):
        """:param critical_fall_ratio: Ratio under which a critical minimum (fake minimum) is considered."""
        super().__init__()
        self.critical_fall_ratio = critical_fall_ratio

    def filter(self, data, **kwargs):
        critical_minima = self._find_critical_minima(data)
        return self._get_interpolated_critical_minima(data, critical_minima)

    def _find_critical_minima(self, data):
        maxi_left_list = []
        maxi_right_list = []
        mini_list = [
            idx
            for idx in range(1, len(data) - 1)
            if data[idx] < data[idx + 1] and data[idx] < data[idx - 1] and data[idx] != 0
        ]

        # Find left and right maxima
        for mini in mini_list:
            maxi_left = mini
            maxi_right = mini

            while data[maxi_left] < data[maxi_left - 1]:
                maxi_left -= 1
                if maxi_left == 0:
                    break

            while data[maxi_right] < data[maxi_right + 1]:
                maxi_right += 1
                if maxi_right == len(data) - 1:
                    break

            maxi_left_list.append(maxi_left)
            maxi_right_list.append(maxi_right)

        return zip(maxi_left_list, mini_list, maxi_right_list)

    def _get_interpolated_critical_minima(self, data, critical_minima):
        for left, center, right in critical_minima:
            maxi = min(data[left], data[right])
            mini = data[center]
            ratio = 1 - mini / maxi

            if ratio < self.critical_fall_ratio:
                # Create a large window to interpolate
                large_left = max(0, left - 50)
                large_right = min(len(data) - 1, right + 51)
                large_window = list(range(large_left, large_right))
                interpolated_window = list(range(left, right + 1))

                # Compute interpolation by removing the critical area
                window_valid_index = [idx for idx in large_window if idx not in interpolated_window]
                window_valid_value = [data[idx] for idx in window_valid_index]
                cs = CubicSpline(window_valid_index, window_valid_value)

                # Replace the critical area by the interpolated values
                interpolated_values = cs(interpolated_window)
                for idx, value in zip(interpolated_window, interpolated_values):
                    data[idx] = value
        return data
