import numpy as np

from .postprocess import PostFilter


class EdgeCutter(PostFilter):
    def __init__(self, index_range: list[int]):
        """:param edge_margin: Margin of the zone to remove localized indexes."""
        super().__init__()
        self.index_range = index_range

    def filter(self, data, **kwargs):
        data = np.array(data)
        data1 = self.index_range[0] < data
        data2 = data < self.index_range[1]
        data3 = data1 & data2
        reversed_columns = list(range(data3.shape[1]))
        reversed_columns.reverse()

        for column_idx in reversed_columns:
            if False in data3[:, column_idx]:
                data = np.delete(data, column_idx, axis=1)

        return data
