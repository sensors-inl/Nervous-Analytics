import numpy as np

from .modeling import ModelInference
from .postprocessing import PostProcess
from .prediction_tracker import PredictionTracker
from .preprocessing import PreProcess


class PhysioPrediction:
    def __init__(
        self,
        preprocess: PreProcess = None,
        model_inference: ModelInference = None,
        postprocess: PostProcess = None,
    ):
        self.preprocess = preprocess
        self.model_inference = model_inference
        self.postprocess = postprocess

    def predict(self, input):
        """:param input: [Python list / Numpy array] Array of input or single input
        :return: [InferenceRec list] Array of InferenceRec containing each output for each process step
        """
        input = np.array(input)
        return_first = False

        if input.ndim == 1:
            input = [input]
            return_first = True

        infrec_list = [PredictionTracker(data) for data in input]

        for infrec in infrec_list:
            if self.preprocess is not None:
                self.preprocess.filter(infrec)
            if self.model_inference is not None:
                self.model_inference.infer(infrec)
            if self.postprocess is not None:
                self.postprocess.filter(infrec)

        return infrec_list[0] if return_first else infrec_list
