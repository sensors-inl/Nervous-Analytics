import numpy as np
import pandas as pd


class PredictionTracker:
    """InferenceRec is a class to store the output of each process step in the inference pipeline."""

    def __init__(self, input):
        self.steps = []
        self.add_step("Input", input)

    def add_step(self, process_name, process_output):
        """Add a process step to the inference records."""
        i = 0
        for step in self.steps:
            proposed_name = f"{process_name}{i}" if i > 0 else process_name
            if step["process_name"] == proposed_name:
                i += 1

        self.steps.append(
            {
                "process_name": f"{process_name}{i}" if i > 0 else process_name,
                "process_output": np.copy(process_output),
            }
        )

    def get_process_output(self, process_name=None, step_index=None):
        """Get the output of a process step. Either by name or by index."""
        if process_name is not None:
            for step in self.steps:
                if step["process_name"] == process_name:
                    return np.copy(step["process_output"])

        elif step_index is not None:
            step = self.steps[step_index]
            return np.copy(step["process_output"])

        raise ValueError(f"Process {process_name} not found in the inference records.")

    def get_all(self):
        """Get all the steps as a pandas DataFrame."""
        return pd.DataFrame(self.steps)
