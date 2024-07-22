from .postprocess import PostFilter


class ECGKnowledge(PostFilter):
    def __init__(
        self,
        max_derivation=0.3,
        hr_range=None,
        ecg_sampling_rate=1024,
        next_index=250,
    ):
        """:param max_derivation: Maximum rate for the rise and fall of the heart rate
        :param hr_range: Largest theoretical heart rate range
        :param ecg_sampling_rate: Sampling rate of the ECG signal
        :param next_index: Index of the previous window representing index 0 of the next window.
                           For example, if next_index equals 250 for a window length of 500,
                           the first four windows will be considered as overlapped by 50% like below:

            +-----------------+-----------------+
            |        1        |        3        | ...
            +-----------------+-----------------+
            :        +-----------------+-----------------+
            :        |        2        |        4        | ...
            :        +-----------------+-----------------+
            :        :        :        :
            :        :        :        :
           1:0      2:250    3:500    4:750

        """
        super().__init__()
        self.max_derivation = max_derivation
        self.ecg_sampling_rate = ecg_sampling_rate
        self.next_index = next_index
        self._hr_min = hr_range[0] if hr_range is not None else 40
        self._hr_max = hr_range[1] if hr_range is not None else 180
        self._last_pair = None

    def filter(self, data, **kwargs):
        pairs = [[data[i], data[i + 1]] for i in range(len(data) - 1)]

        if self._last_pair is not None:
            last_peak_idx = self._last_pair[1]
            first_peak_idx = pairs[0][0]
            overlapped_pair = [last_peak_idx, first_peak_idx]
            pairs = [self._last_pair] + [overlapped_pair] + pairs

        # Check heart rate range
        corrected_pair = self._remove_out_of_hr_range(pairs)
        # Check maximum derivation
        blacklist = self._get_blacklisted_pairs(corrected_pair)

        # Remove values blurred area
        corrected_pair = [idxs for idxs in corrected_pair if idxs is not None]
        # Remove values from blacklist
        corrected_pair = [idxs for idxs in corrected_pair if idxs not in blacklist]

        # Remove full negative pairs
        corrected_pair = [idxs for idxs in corrected_pair if idxs[1] > 0]
        pairs = [idxs for idxs in pairs if idxs[1] > 0]

        self._last_pair = self._get_last_pair(corrected_pair)

        # Pairs labeling
        labeled_pair = []

        for pair in pairs:
            blacklist_status = pair in blacklist
            labeled_pair.append({"pair": pair, "blacklist_status": blacklist_status})

        return labeled_pair

    def _remove_out_of_hr_range(self, pair):
        corrected_pair = []

        for idx1, idx2 in pair:
            bpm = self._idx2bpm(idx2 - idx1)

            try:
                if self._hr_min < bpm < self._hr_max:
                    # add a pair to mean that their bpm is in the normal range
                    corrected_pair.append([idx1, idx2])
                else:
                    # add None to mean that there is a blurred area between two pairs
                    corrected_pair.append(None)

            except TypeError:
                pass

        return corrected_pair

    def _get_blacklisted_pairs(self, corrected_pair):
        previous_bpm = None
        max_rate = 1 + self.max_derivation
        min_rate = 1 / max_rate
        blacklist = []

        for idxs in corrected_pair:
            if idxs is None:
                previous_bpm = None
                continue

            bpm = self._idx2bpm(idxs[1] - idxs[0])

            if previous_bpm is not None:
                current_rate = bpm / previous_bpm
                if not (min_rate < current_rate < max_rate):
                    blacklist.append(idxs)

            previous_bpm = bpm

        return blacklist

    def _get_last_pair(self, pair):
        if not pair:
            return None
        [idx1, idx2] = pair[0]
        idx1 -= self.next_index
        idx2 -= self.next_index
        return [idx1, idx2]

    def _idx2bpm(self, index_diff):
        return index_diff / self.ecg_sampling_rate * 60
