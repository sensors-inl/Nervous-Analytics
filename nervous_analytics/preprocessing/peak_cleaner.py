import numpy as np

from .preprocess import PreFilter


class PeakCleaner(PreFilter):
    """Remove peaks that are too high compared to the surrounding values.
    Developed for EDA due to electronic peak problems.
    """

    def __init__(self, max_ratio=10, avg_min_ratio=20, multi_point=False, max_points=8):
        super().__init__()
        self.max_ratio = max_ratio
        self.avg_min_ratio = avg_min_ratio
        self._filter_function = self._filter_multi_point if multi_point else self._filter_one_point
        self.max_points = max_points

    def filter(self, data, **kwargs):
        return self._filter_function(np.copy(data))

    def _get_amp_min(self, data):
        return (np.average(data) - np.min(data)) / self.avg_min_ratio

    def _filter_one_point(self, data):
        amp_min = self._get_amp_min(data)
        index = np.array(range(1, len(data) - 1))
        # print('amp_min', amp_min)

        for idx in index:
            diff1 = abs(data[idx - 1] - data[idx + 1])
            diff2 = max(abs(data[idx - 1] - data[idx]), abs(data[idx + 1] - data[idx]))
            ratio = diff2 / diff1

            if ratio > self.max_ratio and diff2 > amp_min:
                # print('peak at ', idx)
                data[idx] = (data[idx - 1] + data[idx + 1]) / 2

        return data

    def _filter_multi_point(self, data):
        amp_min = self._get_amp_min(data)
        old_diff = abs(data[1] - data[0])
        current_index = 0
        index_list = []
        # print('amp_min', amp_min)

        while current_index != len(data) - 2:
            index = np.array(range(current_index, len(data) - 1))

            for idx in index:
                current_index = idx
                diff = abs(data[idx + 1] - data[idx])
                ratio = diff / old_diff
                old_diff = diff

                if ratio > self.max_ratio and diff > amp_min:
                    # print('peak at ', idx)
                    idxto = idx + self.max_points
                    size = self._get_peak_size(data[idx:idxto], amp_min)

                    if size > 0:
                        # print(f'  peak size: {size}')
                        index_list.append([idx, idx + size + 1])
                        current_index = idx + size + 1
                        # print(f'  idx: {idx} \t\t\t\tvalue: {data[idx]}')
                        # print(f'  idx+size+1: {idx+size+1} \t\tvalue: {data[idx+size+1]}')
                        old_diff = abs(data[idx] - data[idx + size + 1]) / (size + 1)
                        break

        for idx1, idx2 in index_list:
            x1 = idx1
            x2 = idx2
            y1 = data[x1]
            y2 = data[x2]
            slope = (y2 - y1) / (x2 - x1)

            for idx in range(idx1 + 1, idx2):
                data[idx] = y1 + slope * (idx - x1)

        return data

    def _get_peak_size(self, data, amp_min):
        data = np.subtract(data, data[0])
        data = np.abs(data)
        data = data[1:]

        for idx, value in enumerate(data):
            if value < amp_min * (idx + 1):
                return idx

        # print('Peak size not found')
        return 0
