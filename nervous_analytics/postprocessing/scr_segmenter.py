import numpy as np

from .postprocess import PostFilter


class SCRSegmenter(PostFilter):
    def __init__(
        self,
        filtered_eda_process_name,
        diff2_peak_threshold=0.01,
        min_elevation=0.2,
        replacing_margin=8,
        onset_offset=+2,
        peak_offset=-2,
        min_onset_peak_ratio=1 / 4,
    ):
        """:param filtered_eda_process_name: Name of the process that output the filtered EDA data.
        :param diff2_peak_threshold: Threshold to consider a peak.
        :param min_elevation: Minimum elevation to consider a peak.
        :param replacing_margin: Margin of the zone to replace onsets and peaks by finding a minima or maxima.
        :param onset_offset: Known onset offset used to replace if the minima replacing is not possible.
        :param peak_offset: Known peak offset used to replace if the maxima replacing is not possible.
        :param min_onset_peak_ratio: Minimum ratio of the onset's absolute amplitude compared to the peak.
        """
        super().__init__(additional_output_names=[filtered_eda_process_name])
        self.diff2_peak_threshold = diff2_peak_threshold
        self.min_elevation = min_elevation
        self.replacing_margin = replacing_margin
        self.onset_offset = onset_offset
        self.peak_offset = peak_offset
        self.min_onset_peak_ratio = min_onset_peak_ratio

    def filter(self, data, **kwargs):
        first_key = list(kwargs.keys())[0]
        filtered_eda = kwargs[first_key]

        onsets, peaks = self._find_peaks_onsets(data)
        onsets, peaks = self._remove_low_elevated_peaks(filtered_eda, onsets, peaks)
        onsets, peaks = self._replace_onsets_peaks(filtered_eda, onsets, peaks)
        onsets, peaks = self._spread_onsets_to_peaks(onsets, peaks)
        return np.array([onsets, peaks])

    def _find_peaks_onsets(self, diff2):
        diff2_thresh = [-value if -value > self.diff2_peak_threshold else 0 for value in diff2]
        peaks = [
            idx
            for idx in range(1, len(diff2_thresh) - 1)
            if diff2_thresh[idx] > diff2_thresh[idx + 1] and diff2_thresh[idx] > diff2_thresh[idx - 1]
        ]

        idx_to_remove = []
        onsets = []
        peaks.reverse()

        for peak in peaks:
            onset = peak

            while not self._is_onset(diff2, onset, peak):
                onset -= 1
                if onset in peaks:
                    idx_to_remove.append(peaks.index(onset))
                if onset == 0:
                    break

            onsets.append(onset)

        idx_to_remove.reverse()
        for idx in idx_to_remove:
            peaks.pop(idx)
            onsets.pop(idx)

        peaks.reverse()
        onsets.reverse()
        return onsets, peaks

    def _is_onset(self, diff2, onset, real_peak):
        amplitude = diff2[onset]
        ratio = -self.min_onset_peak_ratio * diff2[real_peak]

        if amplitude < ratio:
            return False

        return (
            amplitude > diff2[onset - 1]
            and amplitude > diff2[onset - 2]
            and amplitude > diff2[onset - 3]
            and amplitude > diff2[onset - 4]
        )

    def _remove_low_elevated_peaks(self, data, onsets, peaks):
        for onset, peak in zip(onsets.copy(), peaks.copy()):
            if data[peak] - data[onset] < self.min_elevation:
                idx = onsets.index(onset)
                onsets.pop(idx)
                peaks.pop(idx)

        return onsets, peaks

    def _replace_onsets_peaks(self, data, onsets, peaks):
        margin = int(self.replacing_margin / 2)

        for i, onset in enumerate(onsets.copy()):
            left = max(0, onset - margin)
            right = min(len(data) - 1, onset + margin)
            idx = np.argmin(data[left:right]) + left

            # Minima replacing is possible
            if idx != left and idx != right:
                onsets[i] = idx
            else:
                onsets[i] = min(max(0, onset - self.onset_offset), len(data) - 1)

        for i, peak in enumerate(peaks.copy()):
            left = max(0, peak - margin)
            right = min(len(data) - 1, peak + margin)
            idx = np.argmax(data[left:right]) + left

            # Maxima replacing is possible
            if idx != left and idx != right:
                peaks[i] = idx
            else:
                peaks[i] = min(max(0, peak - self.peak_offset), len(data) - 1)

        return onsets, peaks

    def _spread_onsets_to_peaks(self, onsets, peaks):
        for i, peak in enumerate(peaks):
            try:
                if onsets[i + 1] == peak:
                    onsets[i + 1] = peak + 1
            except IndexError:
                break
        return onsets, peaks
