from scipy.signal import butter, lfilter

from .preprocess import PreFilter


class FreqFilter(PreFilter):
    def __init__(self, fs, cutoff, order, filter_type):
        """:param fs: Sampling frequency
        :param cutoff: Cutoff frequencies.
        Can be a list of two or a single value.
        :param order: Butterworth order of the filter
        :param filter_type: Type of filter (lowpass, highpass, bandpass, bandstop)
        """
        super().__init__()
        if filter_type not in ["low", "high", "bandpass", "bandstop"]:
            raise ValueError("filter_type must be 'low', 'high', 'bandpass' or 'bandstop'")

        if "band" in filter_type:
            if not isinstance(cutoff, (list, tuple)) or len(cutoff) != 2:
                raise ValueError("cutoff must be a list or tuple with two elements for bandpass or bandstop filter")
        else:
            if isinstance(cutoff, (list, tuple)):
                cutoff = cutoff[0]

        [self.b_coeff, self.a_coeff] = butter(N=order, Wn=cutoff, fs=fs, btype=filter_type)

    def filter(self, data, **kwargs):
        """:param data: Array of data at the sampling frequency
        :return: Filtered data
        """
        return lfilter(self.b_coeff, self.a_coeff, data)
