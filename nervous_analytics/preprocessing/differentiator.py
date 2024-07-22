import numpy as np

from .preprocess import PreFilter


class Differentiator(PreFilter):
    """Differentiate data"""

    def __init__(self, diff_n=1):
        super().__init__()
        self.diff_n = diff_n

    def filter(self, data, **kwargs):
        size = len(data)
        end = True
        data = np.diff(data, n=self.diff_n)

        while len(data) < size:
            if end:
                data = np.append(data, data[0])
            else:
                data = np.insert(data, 0, data[-1])
            end = not end

        return data
