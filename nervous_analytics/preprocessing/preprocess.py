from abc import ABC, abstractmethod

import numpy as np

from ..prediction_tracker import PredictionTracker


class PreFilter(ABC):
    def __init__(self, additional_output_names: list[str] = None):
        self.additional_output_names = additional_output_names

    @abstractmethod
    def filter(self, data, **kwargs):
        pass


class PreProcess(PreFilter):
    """PreProcess object is a composition of multiple filters.
    Execute the filters in the order they are passed in the constructor.
    """

    def __init__(self, filters: list[PreFilter], additional_output_name=None):
        super().__init__(additional_output_name)
        self.filters = filters

    def filter(self, tracker_or_data, **kwargs):
        if isinstance(tracker_or_data, PredictionTracker):
            self._filter_tracker(tracker=tracker_or_data)
        else:
            return self._filter_data(data=tracker_or_data)

    def _filter_tracker(self, tracker: PredictionTracker):
        """Apply all the filters in the order they are passed in the constructor.
        Add the output of each filter as a step in the InferenceRec object.
        """
        last_output = tracker.get_process_output(step_index=-1)

        for process in self.filters:
            if process.additional_output_names is not None:
                additionnal_outputs = [
                    [output_name, tracker.get_process_output(output_name)]
                    for output_name in process.additional_output_names
                ]
                kwargs = dict(additionnal_outputs)
                process_output = process.filter(last_output, **kwargs)
            else:
                process_output = process.filter(last_output)

            process_name = process.__class__.__name__
            tracker.add_step(process_name, process_output)
            last_output = process_output

    def _filter_data(self, data):
        data = np.copy(data)
        for filter in self.filters:
            data = filter.filter(data)
        return data
