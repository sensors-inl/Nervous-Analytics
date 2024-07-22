# Nervous Analytics

Package allowing to extract locations of R-peak from ECG signals in real time.
The ECG process is mainly based on this [paper](https://doi.org/10.1016/j.eswa.2022.117187).

## Installation

```bash
pip install nervous_analytics
```

## Architecture

Our algorithm is composed of three stages. In first the pre-processing stage denoise and prepare the ECG signal using
a wavelet transform and a band-pass filter. Next, in the ML inference stage, the features are extracted from
using two U-net CNN & LSTM based models. Finally, the post-processing stage is responsible for the decision-making...

### 1. Pre-processing

<img src="nervous_analytics/assets/pre_process_modules.png" width="400">

### 2. ML Inference

### 3. Post-processing

<img src="nervous_analytics/assets/post_process_modules.png" width="400">

## Results
