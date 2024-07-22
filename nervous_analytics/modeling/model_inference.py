from typing import Callable, Dict, Tuple

import numpy as np


class ModelInference:
    def __init__(
        self,
        models_dico,
        processing_pred_dico_func: Callable[[Dict], Tuple[str, list]],
    ):
        self.models_dico = models_dico
        self.processing_pred_dico_func = processing_pred_dico_func

    def infer(self, tracker):
        last_output = tracker.get_process_output(step_index=-1)
        pred_dico = {}

        for name, model in self.models_dico.items():
            prediction = model.predict(np.expand_dims(last_output, axis=0))
            reformatted_prediction = [o[0] for o in prediction[0]]
            pred_dico[name] = reformatted_prediction
            tracker.add_step(f"Prediction{name}", reformatted_prediction)

        name, output = self.processing_pred_dico_func(pred_dico)
        tracker.add_step(f"Prediction{name}", output)
