import numpy as np
import pywt

from .preprocess import PreFilter


class WaveDecomposer(PreFilter):
    """9 level DWT: Only 4 and 5 channels contain useful information
    5 level DWT: Only 0 and 1 channels contain useful information
    """

    def __init__(self, wavelet, level, selected_channel, threshold_ratio):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.selected_channel = selected_channel
        self.threshold_ratio = threshold_ratio

    def filter(self, data, **kwargs):
        coeffs = self.get_thresholed_coeffs(data, self.selected_channel, self.threshold_ratio)
        return pywt.waverec(coeffs, self.wavelet)

    def get_universal_threshold(self, data):
        return np.sqrt(2 * np.log2(len(data))) * np.median(np.abs(data))

    def get_coeffs(self, data):
        return list(pywt.wavedec(data, self.wavelet, level=self.level))

    def get_thresholed_coeffs(self, data, selected_channel, threshold_ratio=0.3):
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)
        threshold = self.get_universal_threshold(coeffs[0]) * threshold_ratio
        coeffs_thresholded = list(coeffs)

        for i in range(0, len(coeffs)):
            if i in selected_channel:
                coeffs_thresholded[i] = pywt.threshold(coeffs[i], threshold, mode="soft")
            else:
                coeffs_thresholded[i] = pywt.threshold(coeffs[i], threshold * 1000, mode="hard")

        return coeffs_thresholded
