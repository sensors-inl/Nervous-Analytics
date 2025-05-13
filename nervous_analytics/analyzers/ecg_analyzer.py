"""
Enhanced Real-Time ECG Signal Analysis with Advanced Optimizations.

This module provides a comprehensive ECG analysis system that combines multiple
signal processing techniques and heartbeat detection algorithms to provide
robust heart rate calculations in real-time applications.

The main class, ECGAnalyzer, implements a sliding window approach that
processes incoming ECG data, applies various filtering techniques, and
employs multiple peak detection algorithms to identify R-peaks with high
accuracy. The class tracks peak history to calculate heart rate over time
while handling signal discontinuities and artifacts.

Example usage:
    analyzer = ECGAnalyzer(fs=512)  # Initialize with 512 Hz sampling rate
    ecg_data = [...]  # Your ECG signal data
    time_data = [...]  # Corresponding timestamps
    heart_rate, timestamps, r_peak_timestamps = analyzer.update_hr(ecg_data, time_data)
"""

import logging

import neurokit2
import numpy as np
import pywt
import sleepecg
from scipy.signal import butter, filtfilt, find_peaks

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Minimum discontinuity time threshold
TIME_GAP_THRESHOLD = 0.01  # seconds


class ECGAnalyzer:
    """
    Advanced ECG Analysis Class with Performance Optimizations

    This class implements a real-time ECG analysis system that uses a sliding window
    approach to process incoming ECG data. It applies multiple filtering techniques
    and combines three different R-peak detection algorithms to identify heartbeats
    with high accuracy. The class maintains a history of detected peaks to calculate
    heart rate over time while handling signal discontinuities and artifacts.

    Attributes:
        fs (int): Sampling frequency in Hz
        window_duration (int): Duration of analysis window in seconds
        history_size (int): Size of peak history in seconds
        history_r_peak_idx (ndarray): Array of detected R-peak timestamps
        previous_history_r_peak_idx (ndarray): Previous state of R-peak history
        history_rr (ndarray): Array of R-R intervals
        history_rr_timestamp (ndarray): Timestamps for R-R intervals
        ecg_window (ndarray): Sliding window of ECG signal data
        time_window (ndarray): Corresponding timestamps for the ECG window
        lowpass_coeffs (tuple): Pre-computed lowpass filter coefficients
        highpass_coeffs (tuple): Pre-computed highpass filter coefficients
        derivative_coeffs (tuple): Pre-computed derivative filter coefficients
    """

    def __init__(self, fs=512, window_duration=5, history_size=5):
        """
        Initialize ECG analyzer with configurable parameters

        Parameters:
            fs (int): Sampling frequency in Hz
            window_duration (int): Duration of analysis window in seconds
            history_size (int): Size of peak history in seconds
        """
        self.fs = fs
        self.window_duration = window_duration
        self.history_size = history_size

        # Performance-optimized NumPy arrays for history tracking
        self.history_r_peak_idx = np.array([], dtype=float)
        self.previous_history_r_peak_idx = np.array([], dtype=float)

        self.history_rr = np.array([], dtype=float)
        self.history_rr_timestamp = np.array([], dtype=float)

        # Pre-allocate windows
        self.ecg_window = np.zeros(window_duration * fs, dtype=float)
        self.time_window = np.linspace(-(window_duration * fs - 1) / fs, 0, window_duration * fs)

        # Optimization: Pre-compute filter coefficients
        self.lowpass_coeffs = self._butter_filter(12.0, "low")
        self.highpass_coeffs = self._butter_filter(5.0, "high")
        self.derivative_coeffs = np.array([-0.2, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 0.2]), np.array([1])

    def _reinit_history(self):
        """
        Reinitialize history arrays and windows when a discontinuity is detected

        This method resets all the history arrays and windows to their initial state,
        effectively forgetting all previous peaks and signal data. It's called when
        a significant time gap is detected in the incoming data.
        """
        self.history_r_peak_idx = np.array([], dtype=float)
        self.previous_history_r_peak_idx = np.array([], dtype=float)
        self.history_rr = np.array([], dtype=float)
        self.history_rr_timestamp = np.array([], dtype=float)
        self.ecg_window = np.zeros(self.window_duration * self.fs, dtype=np.float32)
        self.time_window = np.linspace(
            -(self.window_duration * self.fs - 1) / self.fs, 0, self.window_duration * self.fs
        )

    def _butter_filter(self, cutoff, filter_type="low", order=4):
        """
        Create Butterworth filter coefficients for later use

        This method generates filter coefficients that can be reused multiple times,
        improving performance by avoiding redundant computations.

        Parameters:
            cutoff (float): Cutoff frequency in Hz
            filter_type (str): Filter type ('low' or 'high')
            order (int): Filter order

        Returns:
            tuple: Filter coefficients (b, a)
        """
        nyquist = 0.5 * self.fs
        normal_cutoff = cutoff / nyquist
        return butter(order, normal_cutoff, btype=filter_type, analog=False)

    def _apply_filter(self, signal, coeffs):
        """
        Apply a pre-computed filter to a signal

        Parameters:
            signal (ndarray): Input signal to filter
            coeffs (tuple): Filter coefficients (b, a)

        Returns:
            ndarray: Filtered signal
        """
        return filtfilt(*coeffs, signal)

    def _wavelet_denoising(self, signal, wavelet="sym7", level=3):
        """
        Apply wavelet denoising to the ECG signal

        This method performs multi-level wavelet decomposition and thresholding
        to remove noise while preserving important signal features.

        Parameters:
            signal (ndarray): Input ECG signal
            wavelet (str): Wavelet type
            level (int): Decomposition level

        Returns:
            ndarray: Denoised signal
        """
        # Wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=level)

        # Multi-level thresholding
        for i in range(len(coeffs)):
            # Soft thresholding for detail coefficients
            if i < 2:
                coeffs[i] = pywt.threshold(coeffs[i], 2000, mode="hard")
            else:
                coeffs[i].fill(0)

        return pywt.waverec(coeffs, wavelet)

    def _pan_tompkins_filter(self, ecg_signal):
        """
        Implementation of the Pan-Tompkins algorithm for QRS detection preprocessing

        This method applies a series of filtering steps to prepare the ECG signal
        for R-peak detection, including denoising, bandpass filtering, derivative,
        squaring, and moving average integration.

        Parameters:
            ecg_signal (ndarray): Raw ECG signal

        Returns:
            ndarray: Processed signal ready for peak detection
        """
        try:
            # Wavelet denoising
            denoised_signal = self._wavelet_denoising(ecg_signal)

            # Bandpass filtering (optimized with pre-computed coefficients)
            bandpass_filtered = self._apply_filter(
                self._apply_filter(denoised_signal, self.highpass_coeffs), self.lowpass_coeffs
            )

            # Derivative and squaring (vectorized)
            derivative_signal = self._apply_filter(bandpass_filtered, self.derivative_coeffs)
            squared_signal = derivative_signal**2

            # Moving average with NumPy convolution
            window = np.ones(100) / 100
            integrated_signal = np.convolve(squared_signal, window, mode="same")

            return integrated_signal

        except (ValueError, TypeError, RuntimeError) as e:
            # More specific exception handling
            logger.error(f"Pan-Tompkins filter error: {e}")
            return ecg_signal

    def _correct_peaks(self, raw_ecg, peak_indices):
        """
        Refine peak locations to the actual R-peak in the raw signal

        This method searches for the true local maximum around each initially
        detected peak to correct for slight offsets in peak detection.

        Parameters:
            raw_ecg (ndarray): Raw ECG signal
            peak_indices (ndarray): Indices of initially detected peaks

        Returns:
            ndarray: Refined peak indices
        """
        # Refined peak location
        refined_peaks = np.zeros_like(peak_indices, dtype=np.int64)
        for i, peak_idx in enumerate(peak_indices):
            # Local search window
            start = max(0, peak_idx - int(0.2 * self.fs))
            end = min(len(raw_ecg), peak_idx + int(0.2 * self.fs))

            # Find precise local maximum
            local_max_idx = start + np.argmax(raw_ecg[start:end])

            refined_peaks[i] = local_max_idx
        return refined_peaks

    def _edge_removal(self, peak_indices, edge_time):
        """
        Remove peaks detected near the edges of the analysis window

        Peaks detected near the edges of the window are more likely to be artifacts
        or incomplete waveforms. This method removes peaks that are within a specified
        distance from the window edges.

        Parameters:
            peak_indices (ndarray): Indices of detected peaks
            edge_time (float): Time in seconds to exclude from edges

        Returns:
            ndarray: Filtered peak indices
        """
        # Edge peak removal
        edge_samples_start = int(edge_time * self.fs)
        edge_samples_end = int((self.window_duration - edge_time) * self.fs)
        valid_mask = (peak_indices > edge_samples_start) & (peak_indices < edge_samples_end)
        return peak_indices[valid_mask]

    def _detect_r_peaks(self, filtered_ecg):
        """
        Detect R-peaks in the processed ECG signal

        This method uses scipy's find_peaks function to identify potential
        R-peaks in the processed signal based on distance and prominence criteria.

        Parameters:
            filtered_ecg (ndarray): Processed ECG signal

        Returns:
            ndarray: Indices of detected R-peaks
        """
        try:
            # Peak detection with multiple constraints
            peak_indices, _ = find_peaks(
                filtered_ecg,
                distance=int(0.3 * self.fs),  # Minimum RR interval
                prominence=50,  # Minimum peak prominence
            )
            return peak_indices
        except (ValueError, TypeError) as e:
            # More specific exception handling
            logger.error(f"R-peak detection error: {e}")
            return np.array([], dtype=np.int64)

    def _update_peak_history(self, new_peaks):
        """
        Update the history of detected R-peaks

        This method adds newly detected peaks to the history, removing duplicates
        and trimming the history to maintain the specified time window.

        Parameters:
            new_peaks (ndarray): Timestamps of newly detected R-peaks
        """
        # If new_peaks is empty, no update needed
        if len(new_peaks) == 0:
            return

        # Minimum peak distance constraint
        min_distance = 0.3

        # Filter out duplicate peaks - handle empty history case
        if len(self.history_r_peak_idx) > 0:
            diff_matrix = np.abs(new_peaks[:, np.newaxis] - self.history_r_peak_idx)
            unique_mask = np.all(diff_matrix >= min_distance, axis=1)
            unique_new_peaks = new_peaks[unique_mask]
        else:
            unique_new_peaks = new_peaks

        # Append new unique peaks
        self.history_r_peak_idx = np.concatenate([self.history_r_peak_idx, unique_new_peaks])

        # Trim history to maintain time window
        while len(self.history_r_peak_idx) > 1 and (
            self.history_r_peak_idx[-1] - self.history_r_peak_idx[0] > self.history_size
        ):
            self.history_r_peak_idx = self.history_r_peak_idx[1:]

    def _calculate_heart_rate(self):
        """
        Calculate heart rate from detected R-peaks

        This method analyzes the RR intervals between consecutive R-peaks to
        calculate heart rate values, applying various filtering criteria to
        exclude invalid intervals.

        Returns:
            tuple: (heart_rate, heart_rate_timestamp) arrays of calculated heart rates
                  and corresponding timestamps, or (None, None) if no valid heart rates
        """
        # No changes
        if np.array_equal(self.history_r_peak_idx, self.previous_history_r_peak_idx):
            return None, None

        # Insufficient peaks
        if len(self.history_r_peak_idx) <= 1:
            self.previous_history_r_peak_idx = self.history_r_peak_idx.copy()
            return None, None

        # Find new peaks
        if len(self.previous_history_r_peak_idx) > 0:
            new_r_peaks_idx = self.history_r_peak_idx[self.history_r_peak_idx > self.previous_history_r_peak_idx[-1]]
        else:
            new_r_peaks_idx = self.history_r_peak_idx

        if len(new_r_peaks_idx) == 0:
            self.previous_history_r_peak_idx = self.history_r_peak_idx.copy()
            return None, None

        # Initialize return lists
        heart_rate = []
        heart_rate_timestamp = []
        rr = []
        rr_timestamp = []

        # To calculate RR intervals, get last peak timestamp
        if len(self.previous_history_r_peak_idx) > 0:
            new_r_peaks_idx = np.concatenate(([self.previous_history_r_peak_idx[-1]], new_r_peaks_idx))

        # Define physiological bounds for heart rate (30-200 BPM)
        min_interval = 0.3  # 200 BPM max
        max_interval = 2.0  # 30 BPM min

        # Threshold for sudden changes (±30% change)
        change_threshold = 0.428  # Derived from 1 - 1/(1+0.3) ≈ 0.428

        if self.history_rr.size > 0:
            # Use median interval in history window as valid reference
            sd_median = np.median(self.history_rr)

            # Run through new peak list to calculate valid heart rate
            i = 1
            while i < len(new_r_peaks_idx):
                interval = new_r_peaks_idx[i] - new_r_peaks_idx[i - 1]

                # Remove peak index if too short (Heart Rate increase more than 30%)
                if (sd_median - interval) / sd_median > change_threshold:
                    logger.debug(f"interval {interval} too short at {new_r_peaks_idx[i]}")
                    new_r_peaks_idx = np.delete(new_r_peaks_idx, i)
                    continue

                # Don't get heart rate if interval is too long (Heart Rate decrease more than 30%)
                if (interval - sd_median) / sd_median > change_threshold:
                    logger.debug(
                        f"peak too long at {new_r_peaks_idx[i]} median is {sd_median} value got is {interval}"
                    )
                    i = i + 1
                    continue

                # Don't get heart rate if out of physiological bounds
                if (interval < min_interval) or (interval > max_interval):
                    i = i + 1
                    continue

                # Add valid interval to RR list
                rr.append(interval)
                rr_timestamp.append(round(float(new_r_peaks_idx[i]), 3))

                # Calculate valid heart rate
                heart_rate.append(round(float(60.0 / interval), 2))
                heart_rate_timestamp.append(round(float(new_r_peaks_idx[i]), 3))
                i = i + 1
                # Using logger instead of print
                logger.debug(f"Current heart rate: {round(float(60.0 / sd_median), 2)} BPM")

        else:
            # Run through new peak list to calculate valid heart rate
            i = 1
            while i < len(new_r_peaks_idx):
                interval = new_r_peaks_idx[i] - new_r_peaks_idx[i - 1]
                # Don't get heart rate if out of physiological bounds
                if (interval < min_interval) or (interval > max_interval):
                    i = i + 1
                    continue

                # Add valid interval to RR list
                rr.append(interval)
                rr_timestamp.append(round(float(new_r_peaks_idx[i]), 3))

                # Calculate valid heart rate
                heart_rate.append(round(float(60.0 / interval), 2))
                heart_rate_timestamp.append(round(float(new_r_peaks_idx[i]), 3))
                i = i + 1

        # Safety check for empty results
        if not rr:
            self.previous_history_r_peak_idx = self.history_r_peak_idx.copy()
            return None, None

        # Update rr history
        self.history_rr = np.concatenate([self.history_rr, np.array(rr)])
        self.history_rr_timestamp = np.concatenate([self.history_rr_timestamp, np.array(rr_timestamp)])

        # Trim RR history to maintain time window
        while len(self.history_rr_timestamp) > 1 and (
            self.history_rr_timestamp[-1] - self.history_rr_timestamp[0] > self.history_size
        ):
            self.history_rr_timestamp = self.history_rr_timestamp[1:]
            self.history_rr = self.history_rr[1:]

        # Update previous peak history
        self.previous_history_r_peak_idx = self.history_r_peak_idx.copy()

        return heart_rate, heart_rate_timestamp

    def update_hr(self, ecg_data, time_data):
        """
        Main interface method for updating ECG data and calculating heart rate

        This method takes new ECG data and corresponding timestamps, updates the
        sliding windows, applies filtering and peak detection, and calculates
        heart rate values.

        Parameters:
            ecg_data (array-like): New ECG signal data points
            time_data (array-like): Corresponding timestamps

        Returns:
            tuple: (heart_rate, heart_rate_timestamp, r_peak_timestamp)
                  Lists of calculated heart rates, corresponding timestamps, and
                  detected R-peak timestamps. Returns (None, None, None) if no valid heart rates.
        """
        # Input type conversion and validation
        ecg_data = -np.asarray(ecg_data, dtype=np.float32)
        time_data = np.asarray(time_data, dtype=np.float32)

        if (time_data.size == 0) or (ecg_data.size == 0):
            return None, None, None

        # Handle initial negative time values
        if time_data[0] < 0:
            start_idx = np.argmax(time_data > 0)
            ecg_data = ecg_data[start_idx:]
            time_data = time_data[start_idx:]

            # If all time data was negative, return
            if ecg_data.size == 0 or time_data.size == 0:
                return None, None, None

        # Detect discontinuity
        if abs(time_data[0] - self.time_window[-1]) > TIME_GAP_THRESHOLD:
            logger.info(f"ECG Measurement discontinuity detected at {time_data[0]} vs {self.time_window[-1]}")
            self._reinit_history()

        # Update sliding windows
        self.ecg_window = np.roll(self.ecg_window, -len(ecg_data))
        self.ecg_window[-len(ecg_data) :] = ecg_data

        self.time_window = np.roll(self.time_window, -len(time_data))
        self.time_window[-len(time_data) :] = time_data

        # Pan-Tompkins filtering
        filtered_ecg = self._pan_tompkins_filter(self.ecg_window)

        # R-peak detection using multiple algorithms
        try:
            # 1. Pan-Tompkins algorithm
            r_peaks_pt = self._detect_r_peaks(filtered_ecg)
            r_peaks_pt = self._correct_peaks(self.ecg_window, r_peaks_pt)
            r_peaks_pt = self._edge_removal(r_peaks_pt, 0.5)

            # 2. SleepECG algorithm
            r_peaks_se = sleepecg.detect_heartbeats(self.ecg_window, fs=self.fs)
            r_peaks_se = self._correct_peaks(self.ecg_window, r_peaks_se)
            r_peaks_se = self._edge_removal(r_peaks_se, 0.5)

            # 3. NeuroKit2 algorithm
            _, results = neurokit2.ecg_peaks(self.ecg_window, sampling_rate=self.fs)
            r_peaks_nk = np.array(results["ECG_R_Peaks"], dtype=np.int64)
            r_peaks_nk = self._correct_peaks(self.ecg_window, r_peaks_nk)
            r_peaks_nk = self._edge_removal(r_peaks_nk, 0.5)

            # Get intersection of all arrays for robustness
            # Check if arrays are not empty before intersection
            if r_peaks_pt.size > 0 and r_peaks_se.size > 0:
                r_peaks_pt = np.intersect1d(r_peaks_pt, r_peaks_se)
            elif r_peaks_se.size > 0:
                r_peaks_pt = r_peaks_se

            if r_peaks_pt.size > 0 and r_peaks_nk.size > 0:
                r_peaks_pt = np.intersect1d(r_peaks_pt, r_peaks_nk)
            elif r_peaks_nk.size > 0 and r_peaks_pt.size == 0:
                r_peaks_pt = r_peaks_nk

        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"R-peak detection algorithm error: {e}")
            return None, None, None

        # Safety check for empty results
        if r_peaks_pt.size == 0:
            return None, None, None

        # Update peak history
        r_peak_timestamp = list(self.time_window[r_peaks_pt])
        self._update_peak_history(self.time_window[r_peaks_pt])

        # Calculate heart rate
        heart_rate, heart_rate_timestamp = self._calculate_heart_rate()

        # Return appropriate values
        if heart_rate is None:
            return None, None, r_peak_timestamp
        elif len(heart_rate) == 0:
            return None, None, r_peak_timestamp
        else:
            return heart_rate, heart_rate_timestamp, r_peak_timestamp
