from .postprocess import PostFilter


class ECGKnowledge(PostFilter):
    def __init__(
        self,
        max_derivation=0.3,
        hr_range=None,
        ecg_sampling_rate=1024,
        next_index=2500,
    ):
        """
        :param max_derivation: Maximum rate for the rise and fall of the heart rate
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
        pairs = [idxs for idxs in pairs if idxs[0] >= 0]

        # Pairs labeling
        labeled_pair = []

        for pair in pairs:
            blacklist_status = pair not in corrected_pair
            labeled_pair.append({"pair": pair, "blacklist_status": blacklist_status})

        self._last_pair = self._get_last_pair(corrected_pair)
        return labeled_pair

    def _remove_out_of_hr_range(self, pair):
        corrected_pair = []

        for idx1, idx2 in pair:
            bpm = self._idx2bpm(idx2 - idx1)

            try:
                if self._hr_min < bpm < self._hr_max:
                    # add a pair to mean that their bpm is in the normal range
                    corrected_pair.append([idx1, idx2])
            except TypeError:
                pass

        return corrected_pair

    def _get_blacklisted_pairs(self, corrected_pair):
        last_coherent_idxs = None
        last_is_previous = False
        blacklist = []

        for idxs in corrected_pair:
            result = self._is_evolution_rate_coherent(idxs, last_coherent_idxs, last_is_previous)
            last_is_previous = result

            if result:
                last_coherent_idxs = idxs
            else:
                blacklist.append(idxs)

        return blacklist

    def _is_evolution_rate_coherent(self, idxs, last_coherent_idxs, last_is_previous):
        max_rate = 1 + self.max_derivation
        min_rate = 1 / max_rate
        n = 1

        if last_coherent_idxs is None:
            return True

        bpm = self._idx2bpm(idxs[1] - idxs[0])
        last_bpm = self._idx2bpm(last_coherent_idxs[1] - last_coherent_idxs[0])

        if not last_is_previous:
            diff = last_coherent_idxs[1] - last_coherent_idxs[0]
            next_idx = last_coherent_idxs[1] + diff

            while next_idx < idxs[0]:
                n += 1
                next_idx += diff

            if abs(next_idx - idxs[0]) < abs(next_idx - idxs[1]):
                n += 1

        max_rate = pow(max_rate, n)
        min_rate = pow(min_rate, n)
        current_rate = bpm / last_bpm
        return min_rate < current_rate < max_rate

    def _get_last_pair(self, corrected_pair):
        if not corrected_pair:
            return None

        corrected_pair2 = [
            [idxs[0] - self.next_index, idxs[1] - self.next_index]
            for idxs in corrected_pair
            if idxs[1] < self.next_index
        ]

        return corrected_pair2[-1] if corrected_pair2 else None

    def _idx2bpm(self, index_diff):
        return index_diff / self.ecg_sampling_rate * 60
