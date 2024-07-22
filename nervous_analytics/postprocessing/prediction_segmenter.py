import numpy as np

from .postprocess import PostFilter


class PredictionSegmenter(PostFilter):
    def filter(self, data, **kwargs):
        data = np.array(data)
        zone_found = False
        zone_list = []
        zone = []

        # Segmentation by zones
        for idx, pred in enumerate(data):
            if pred != 0:
                zone.append(idx)
                zone_found = True
            elif zone_found:
                zone_list.append(zone)
                zone = []
                zone_found = False

        # Add last zone if necessary
        if zone_found:
            zone_list.append(zone)

        pred_segmented = []

        # Find the middle point of each zone
        for zone in zone_list:
            pred_values = data[zone]
            zone_avg = np.array(pred_values) * np.array(zone)
            zone_avg = np.sum(zone_avg)
            zone_avg /= np.sum(pred_values)
            idx = int(np.round(zone_avg))
            pred_segmented.append(idx)

        return np.array(pred_segmented)
