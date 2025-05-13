"""
Enhanced Real-Time EDA Signal Analysis with Advanced Optimizations.

This module provides a comprehensive EDA analysis system that combines multiple
signal processing techniques to provide robust eda peaks detections in real-time
applications.

The main class, EDAAnalyzer, implements a sliding window approach that
processes incoming EDA data, applies various filtering techniques, and
employs multiple EDA peak detection algorithms to identify EDA peaks with high
accuracy.

Example usage:
    analyzer = EDAAnalyzer(fs=8)  # Initialize with 8 Hz sampling rate
    eda_data = [...]  # Your EDA signal data
    time_data = [...]  # Corresponding timestamps
    amplitude, duration, levelSCR, timestamp, = analyzer.update_eda_peak(eda_data, time_data)
"""

import logging

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt, savgol_coeffs

from . import eda_decisiontree

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Minimum discontinuity time threshold
TIME_GAP_THRESHOLD = 0.3  # seconds (because > 1/8s = 0.125s)


class EDAAnalyzer:
    """
    Advanced EDA Analysis Class with Performance Optimizations

    This class implements a real-time EDA analysis system that uses a sliding window
    approach to process incoming EDA data. It applies multiple filtering techniques
    and combines differents EDA-peak detection algorithms to identify electrodermal
    activity with high accuracy.

    Attributes:
        fs (int): Sampling frequency in Hz
        window_duration (int): Duration of analysis window in seconds
        history_size (int): Size of peak history in seconds
        history_eda_min_peak_idx (ndarray): Array of detected EDA-MAX-peak timestamps
        previous_history_eda_min_peak_idx (ndarray): Previous state of EDA-MAX-peak history
        history_eda_max_peak_idx (ndarray): Array of detected EDA-MAX-peak timestamps
        previous_history_eda_max_peak_idx (ndarray): Previous state of EDA-MAX-peak history
        eda_window (ndarray): Sliding window of EDA signal data
        time_window (ndarray): Corresponding timestamps for the EDA window
        savitzky_golay_coeffs (tuple): Pre-computed Savitzky-Golay filter coefficients
    """

    def __init__(self, fs=8, window_duration=20, history_size=20):
        """
        Initialize EDA analyzer with configurable parameters

        Parameters:
            fs (int): Sampling frequency in Hz
            window_duration (int): Duration of analysis window in seconds
            history_size (int): Size of peak history in seconds
        """
        if window_duration < 5:
            logging.error("Window duration must be higher than 5 seconds")

        self.fs = fs
        self.window_duration = window_duration
        self.history_size = history_size

        # Performance-optimized NumPy arrays for history tracking
        self.history_eda_max_peak_idx = np.array([], dtype=float)
        self.history_eda_min_peak_idx = np.array([], dtype=float)
        self.previous_history_eda_max_peak_idx = np.array([], dtype=float)
        self.previous_history_eda_min_peak_idx = np.array([], dtype=float)
        self.history_eda_max_peak_value = np.array([], dtype=float)
        self.history_eda_min_peak_value = np.array([], dtype=float)

        # Pre-allocate windows
        self.eda_window = np.zeros(window_duration * fs, dtype=float)
        self.time_window = np.linspace(-(window_duration * fs - 1) / fs, 0, window_duration * fs)

        # Optimization: Pre-compute filter coefficients
        self.highpass_coeffs = self._butter_filter(0.1, "high")
        self.savitzky_golay_coeffs = np.array(savgol_coeffs(8, 3)), np.array([1])

    def _reinit_history(self, new_time_start):
        """
        Reinitialize history arrays and windows when a discontinuity is detected

        This method resets all the history arrays and windows to their initial state,
        effectively forgetting all previous peaks and signal data. It's called when
        a significant time gap is detected in the incoming data.

        Parameters:
            new_time_start (float): New start for time_window initialisation
        """
        self.history_eda_max_peak_idx = np.array([], dtype=float)
        self.history_eda_min_peak_idx = np.array([], dtype=float)
        self.previous_history_eda_max_peak_idx = np.array([], dtype=float)
        self.previous_history_eda_min_peak_idx = np.array([], dtype=float)
        self.history_eda_max_peak_value = np.array([], dtype=float)
        self.history_eda_min_peak_value = np.array([], dtype=float)
        self.eda_window = np.zeros(self.window_duration * self.fs, dtype=np.float32)
        self.time_window = np.linspace(
            new_time_start - (self.window_duration * self.fs - 1) / self.fs,
            new_time_start,
            self.window_duration * self.fs,
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

    def _savitzky_golay_filter(self, eda_signal):
        """
        Apply a Savitzky-Golay filter to smooth an EDA signal.

        This method uses precomputed Savitzky-Golay filter coefficients to smooth the input signal.
        It handles specific error cases with appropriate recovery strategies.

        Parameters:
            eda_signal (ndarray): Raw EDA signal to be smoothed.

        Returns:
            ndarray: Smoothed EDA signal. If an error occurs, returns the original signal.
        """
        # Validation d'entrée
        if eda_signal is None or len(eda_signal) == 0:
            logger.warning("Empty EDA signal received in Savitzky-Golay filter, returning empty array")
            return np.array([], dtype=float)

        if len(eda_signal) < 8:  # Valeur minimale pour le filtre (taille du fenêtrage)
            logger.warning(
                f"EDA signal too short for Savitzky-Golay filtering: {len(eda_signal)} samples."
                + "Returning original signal."
            )
            return eda_signal

        try:
            # Smooth the signal via Savitzky Golay Filter
            sav_gol_signal = self._apply_filter(eda_signal, self.savitzky_golay_coeffs)

            # Vérification des valeurs NaN ou infinies dans le résultat
            if np.any(np.isnan(sav_gol_signal)) or np.any(np.isinf(sav_gol_signal)):
                logger.warning("Savitzky-Golay filter produced NaN or infinite values. Using original signal instead.")
                return eda_signal

            return sav_gol_signal

        except ValueError as e:
            logger.error(f"Savitzky-Golay parameter error: {e}. Check window length and polynomial order.")
            return eda_signal
        except np.linalg.LinAlgError as e:
            logger.error(f"Savitzky-Golay linear algebra error: {e}. Signal might contain problematic patterns.")
            return eda_signal
        except Exception as e:
            logger.error(f"Unexpected error in Savitzky-Golay filter: {e}")
            return eda_signal

    def _detect_eda_peaks(self, filtered_eda):
        """
        Detect positive and negative peaks in a filtered EDA signal, excluding the first and last 2.5 seconds.

        Identifies zero-crossings in the signal's derivative to detect local minima and maxima.
        Classifies each zero-crossing as a peak or a trough based on the signal's slope direction.
        Analysis focuses on the central region of the signal, excluding the first and last 20 samples.
        In case of an error during peak detection, logs the error and returns empty arrays.

        Parameters:
            filtered_eda (ndarray): Preprocessed EDA signal, typically filtered to remove noise.

        Returns:
            tuple: Two lists of indices:
                - min_peak_idx (list): Indices of detected minima (troughs).
                - max_peak_idx (list): Indices of detected maxima (peaks).
        """
        # Validation d'entrée
        if filtered_eda is None:
            logger.error("Received None as filtered_eda in peak detection")
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        if len(filtered_eda) < 41:  # Besoin d'au moins 41 points pour exclure 20 de chaque côté
            logger.warning(f"Signal too short for peak detection: {len(filtered_eda)} samples")
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        try:
            min_peak_idx = []
            max_peak_idx = []

            # Define borders to exclude (first and last 20 samples, approximately 2.5s)
            start_idx = round(2.5 * self.fs)
            end_idx = len(filtered_eda) - round(2.5 * self.fs)

            # Only process the central part of the signal
            central_signal = filtered_eda[start_idx:end_idx]

            if len(central_signal) <= 1:
                logger.warning("Central signal too short after border exclusion")
                return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

            # Detect Variation Changement in the central region
            try:
                # Détection sécurisée des changements de signe (zéro-crossings)
                idx = np.where(central_signal[:-1] * central_signal[1:] < 0)[0] + 1
                idx = idx.tolist()
            except IndexError as e:
                logger.error(f"Index error during zero-crossing detection: {e}")
                return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

            # Check if it's a max or a min
            for i in idx:
                if i >= len(central_signal):
                    logger.warning(f"Index {i} out of bounds for central_signal length {len(central_signal)}")
                    continue

                # Adjust index to reference the original signal
                orig_idx = i + start_idx

                if central_signal[i - 1] < 0:
                    min_peak_idx.append(orig_idx)
                elif central_signal[i - 1] > 0:
                    max_peak_idx.append(orig_idx)

            return np.array(min_peak_idx, dtype=np.int64), np.array(max_peak_idx, dtype=np.int64)

        except IndexError as e:
            logger.error(f"Index error in EDA peak detection: {e}")
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        except ValueError as e:
            logger.error(f"Value error in EDA peak detection: {e}")
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        except Exception as e:
            logger.error(f"Unexpected error in EDA peak detection: {e}")
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    def _filter_indices(self, values, min_idx, max_idx, value_threshold, time_threshold, slope_threshold):
        """
        Filter peak index pairs based on amplitude, duration, and slope thresholds.

        This method filters the detected peaks (minima and maxima) based on three criteria:
        - The difference in signal values between each minimum and maximum must exceed a specified threshold.
        - The time difference between the minimum and maximum must meet a minimum duration threshold.
        - The slope between the minimum and maximum points must exceed a specified slope threshold.

        Parameters:
            values (ndarray): The original signal values.
            min_idx (list or ndarray): Indices of detected minima (troughs).
            max_idx (list or ndarray): Indices of detected maxima (peaks).
            value_threshold (float): Minimum required difference in amplitude between the minimum and maximum.
            time_threshold (float): Minimum required time duration between the minimum and maximum indices.
            slope_threshold (float): Minimum required slope (difference in amplitude / duration) between the peaks.

        Returns:
            tuple: Two ndarrays of filtered indices:
                - min_idx_filtered (ndarray): Filtered minima indices.
                - max_idx_filtered (ndarray): Filtered maxima indices.
        """
        # Convert to NumPy arrays for efficient processing
        min_idx = np.array(min_idx)
        max_idx = np.array(max_idx)
        values = np.array(values)

        empty_array = np.array([], dtype=float)

        if len(min_idx) == 0 or len(max_idx) == 0:
            return empty_array, empty_array

        # Adjust lengths if necessary
        if len(min_idx) != len(max_idx):
            min_len = min(len(min_idx), len(max_idx))
            if min_idx[0] > max_idx[0]:
                min_idx = min_idx[-min_len:]
                max_idx = max_idx[-min_len:]
            else:
                min_idx = min_idx[:min_len]
                max_idx = max_idx[:min_len]

        # Ensure the first index is a minimum
        if len(min_idx) == len(max_idx):
            if min_idx[0] < max_idx[0]:
                pass
            else:
                max_idx = max_idx[-len(max_idx) + 1 :]
                min_idx = min_idx[: len(min_idx) - 1]

        # Special case: empty arrays
        if len(min_idx) <= 0 or len(max_idx) <= 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        # Vectorized calculations for time difference and amplitude difference
        time_diff = np.abs(max_idx - min_idx)
        value_diff = []
        for min_i, max_i in zip(min_idx, max_idx):
            value_diff.append(np.abs(values[int(max_i)] - values[int(min_i)]))
        value_diff = np.array(value_diff)

        # Avoid division by zero for slope calculation
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.divide(value_diff, time_diff)
            # Replace NaN or infinite values with 0
            slope = np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)

        # Create mask based on the three conditions
        mask = (value_diff >= value_threshold) & (time_diff >= time_threshold) & (slope >= slope_threshold)

        # Apply the mask to get the filtered index arrays
        min_idx_filtered = min_idx[mask]
        max_idx_filtered = max_idx[mask]

        return min_idx_filtered, max_idx_filtered

    def _update_peak_history(self, new_min_peaks, new_max_peaks, new_min_peaks_value, new_max_peaks_value):
        """
        Update the history of detected EDA-peaks

        This method adds newly detected peaks to the history, removing duplicates
        and trimming the history to maintain the specified time window.

        Parameters:
            new_peaks (ndarray): Timestamps of newly detected R-peaks

        Note:
            This method could have issues with empty arrays in the vectorized operations.
            Additional checks for empty arrays should be added.
        """

        # Minimum peak distance constraint
        min_distance = 1 / self.fs

        # Filter out duplicate peaks
        unique_min_mask = np.all(
            np.abs(new_min_peaks[:, np.newaxis] - self.history_eda_min_peak_idx) >= min_distance, axis=1
        )

        unique_max_mask = np.all(
            np.abs(new_max_peaks[:, np.newaxis] - self.history_eda_max_peak_idx) >= min_distance, axis=1
        )

        # Append new unique peaks
        self.history_eda_min_peak_idx = np.concatenate([self.history_eda_min_peak_idx, new_min_peaks[unique_min_mask]])
        self.history_eda_max_peak_idx = np.concatenate([self.history_eda_max_peak_idx, new_max_peaks[unique_max_mask]])
        # Append associated values (only if the peak is added)
        self.history_eda_min_peak_value = np.concatenate(
            [self.history_eda_min_peak_value, new_min_peaks_value[unique_min_mask]]
        )
        self.history_eda_max_peak_value = np.concatenate(
            [self.history_eda_max_peak_value, new_max_peaks_value[unique_max_mask]]
        )

        # Trim history to maintain time window
        while len(self.history_eda_min_peak_idx) > 1 and (
            self.history_eda_min_peak_idx[-1] - self.history_eda_min_peak_idx[0] > self.history_size * self.fs
        ):
            self.history_eda_min_peak_idx = self.history_eda_min_peak_idx[1:]
            self.history_eda_min_peak_value = self.history_eda_min_peak_value[1:]

        while len(self.history_eda_max_peak_idx) > 1 and (
            self.history_eda_max_peak_idx[-1] - self.history_eda_max_peak_idx[0] > self.history_size * self.fs
        ):
            self.history_eda_max_peak_idx = self.history_eda_max_peak_idx[1:]
            self.history_eda_max_peak_value = self.history_eda_max_peak_value[1:]

    def _polynomial(self, x, coefs):
        """
        Evaluate a 3rd-degree polynomial at given points x using the provided coefficients.

        This method evaluates the polynomial of the form:
            p(x) = a * x^3 + b * x^2 + c * x + d

        Parameters:
            x (ndarray or scalar): The input value(s) where the polynomial is evaluated.
            coefs (tuple): The coefficients of the polynomial (a, b, c, d), where:
                - a: Coefficient of x^3
                - b: Coefficient of x^2
                - c: Coefficient of x
                - d: Constant term

        Returns:
            ndarray or scalar: The result of the polynomial evaluation at the given points x.
        """
        a, b, c, d = coefs
        return a * x**3 + b * x**2 + c * x + d

    def _interpolate_polynomial3(
        self,
        x1,
        y1,
        x1a,
        y1a,
        x1b,
        y1b,
        x1c,
        y1c,
        x2,
        y2,
        x2a,
        y2a,
        x2b,
        y2b,
        x2c,
        y2c,
        xa,
        ya,
        xb,
        yb,
        xc,
        yc,
        xd,
        yd,
    ):
        """
        Perform cubic polynomial interpolation with constraints at given points.

        This method tries to find the best-fitting cubic polynomial that passes through
        the provided set of points, with conditions on the values and derivatives at
        specific points. If the direct solution approach fails, it uses optimization
        to minimize the least square error with constraints. If that also fails, it falls
        back to a simple cubic interpolation using key points.

        Parameters:
            x1, y1, x1a, y1a, x1b, y1b, x1c, y1c: Coordinates for the first interval.
            x2, y2, x2a, y2a, x2b, y2b, x2c, y2c: Coordinates for the second interval.
            xa, ya, xb, yb, xc, yc, xd, yd: Additional interpolation points.

        Returns:
            ndarray: Coefficients [a, b, c, d] of the cubic polynomial p(x) = a*x^3 + b*x^2 + c*x + d.
                    Returns a simple linear interpolation if all methods fail.
        """

        # Define the 3rd-degree polynomial function
        def polynomial_of_degree_3(x, coeffs):
            a, b, c, d = coeffs
            return a * x**3 + b * x**2 + c * x + d

        # Define the derivative of the 3rd-degree polynomial
        def derivative_polynomial_of_degree_3(x, coeffs):
            a, b, c, d = coeffs
            return 3 * a * x**2 + 2 * b * x + c

        # Approach 1: Direct solution method
        try:
            # Create the matrix and vector for the imposed points method
            # We impose P(x1) = y1, P'(x1) = 0, P(x2) = y2, P'(x2) = 0
            A = np.array(
                [[x1**3, x1**2, x1, 1], [3 * x1**2, 2 * x1, 1, 0], [x2**3, x2**2, x2, 1], [3 * x2**2, 2 * x2, 1, 0]]
            )
            b = np.array([y1, 0, y2, 0])

            # Solve the linear system to find coefficients
            coeffs = solve(A, b)

            # Verify the quality of the fit
            points = np.array(
                [
                    [x1, y1],
                    [x1a, y1a],
                    [x1b, y1b],
                    [x1c, y1c],
                    [x2, y2],
                    [x2a, y2a],
                    [x2b, y2b],
                    [x2c, y2c],
                    [xa, ya],
                    [xb, yb],
                    [xc, yc],
                    [xd, yd],
                ]
            )
            x_data, y_data = points[:, 0], points[:, 1]
            y_pred = polynomial_of_degree_3(x_data, coeffs)
            error = np.sum((y_data - y_pred) ** 2)

            # If the error is reasonable, return these coefficients
            if error < 10.0:  # Adjust this threshold as needed
                return coeffs
        except np.linalg.LinAlgError:
            pass  # If there's an error, continue with optimization approach

        # Approach 2: Optimization with better-formulated constraints
        # Format the points as a numpy array
        points = np.array(
            [
                [x1, y1],
                [x1a, y1a],
                [x1b, y1b],
                [x1c, y1c],
                [x2, y2],
                [x2a, y2a],
                [x2b, y2b],
                [x2c, y2c],
                [xa, ya],
                [xb, yb],
                [xc, yc],
                [xd, yd],
            ]
        )

        # Function to calculate the least squares error
        def least_squares_error(coeffs):
            x_data, y_data = points[:, 0], points[:, 1]
            y_pred = polynomial_of_degree_3(x_data, coeffs)
            return np.sum((y_data - y_pred) ** 2)

        # Define the constraints
        constraints = [
            {"type": "eq", "fun": lambda coeffs: polynomial_of_degree_3(x1, coeffs) - y1},
            {"type": "eq", "fun": lambda coeffs: polynomial_of_degree_3(x2, coeffs) - y2},
            {"type": "eq", "fun": lambda coeffs: derivative_polynomial_of_degree_3(x1, coeffs)},
            {"type": "eq", "fun": lambda coeffs: derivative_polynomial_of_degree_3(x2, coeffs)},
        ]

        # Multiple starting points to avoid local minima
        starting_points = [
            [0, 0, 0, 0],
            [0.001, 0.001, 0.001, 0.001],
            [-0.001, 0.001, -0.001, 0.001],
            [0.1, -0.1, 0.1, -0.1],
        ]

        # Use previous coefficients if available
        if hasattr(self, "previous_coeffs"):
            starting_points.append(self.previous_coeffs)

        best_result = None
        best_error = float("inf")

        # Try each starting point
        for start_point in starting_points:
            try:
                result = minimize(
                    fun=least_squares_error,
                    x0=start_point,
                    constraints=constraints,
                    method="SLSQP",  # Reliable method for this problem
                    options={"maxiter": 1000, "ftol": 1e-6, "disp": False},
                )

                # Check if this result is better
                if result.fun < best_error:
                    best_error = result.fun
                    best_result = result
            except Exception:
                continue

        # Check if the optimization result is successful
        if best_result is not None and best_result.success:
            self.previous_coeffs = best_result.x
            return best_result.x

        # Approach 3: Fallback solution - classical interpolation using 4 key points
        # Use a simple polynomial interpolation with 4 key points
        key_points = np.array([[x1, y1], [xa, ya], [xb, yb], [x2, y2]])
        A = np.vstack([key_points[:, 0] ** 3, key_points[:, 0] ** 2, key_points[:, 0], np.ones(4)]).T
        try:
            coeffs = solve(A, key_points[:, 1])
            self.previous_coeffs = coeffs
            return coeffs
        except Exception:
            # Last resort: simple linear polynomial as fallback
            return [0, 0, (y2 - y1) / (x2 - x1), y1 - (y2 - y1) / (x2 - x1) * x1]

    def _calculate_eda_response(self):
        """
        This function processes the new peaks in the EDA signal and calculates the response parameters,
        including amplitude, duration, and slope, using polynomial interpolation of order 3.

        Returns:
            amplitude (list): The amplitude of each EDA response (difference between max and min peaks).
            duration (list): The duration of each EDA response (time difference between max and min peaks).
            level_scr (list): The level of the skin conductance response at each minimum peak.
            timestamp (list): The timestamp (index) of the minimum peaks.
            after_slope (list): The slope of the EDA signal after the maximum peak.
            coeffcubicA, coeffcubicB, coeffcubicC, coeffcubicD (lists): Coefficients for polynomial
                                                                        interpolation of order 3 for each response.
            x_curve_interpolate (list): List of x-values used to plot the interpolated curve for each response.
            y_curve_interpolate (list): List of y-values used to plot the interpolated curve for each response.
        """

        empty_array = np.array([], dtype=float)

        # Check if there are changes in the peaks
        if np.array_equal(self.history_eda_max_peak_idx, self.previous_history_eda_max_peak_idx) or np.array_equal(
            self.history_eda_min_peak_idx, self.previous_history_eda_min_peak_idx
        ):
            return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array, empty_array

        # Identify new peaks since last history
        if len(self.previous_history_eda_max_peak_idx) > 0:
            new_eda_max_peaks_idx = self.history_eda_max_peak_idx[
                self.history_eda_max_peak_idx > self.previous_history_eda_max_peak_idx[-1]
            ]
            new_eda_min_peaks_idx = self.history_eda_min_peak_idx[
                self.history_eda_min_peak_idx > self.previous_history_eda_min_peak_idx[-1]
            ]
            new_eda_max_peaks_value = self.history_eda_max_peak_value[
                self.history_eda_max_peak_idx > self.previous_history_eda_max_peak_idx[-1]
            ]
            new_eda_min_peaks_value = self.history_eda_min_peak_value[
                self.history_eda_min_peak_idx > self.previous_history_eda_min_peak_idx[-1]
            ]

        else:
            new_eda_max_peaks_idx = self.history_eda_max_peak_idx
            new_eda_min_peaks_idx = self.history_eda_min_peak_idx
            new_eda_max_peaks_value = self.history_eda_max_peak_value
            new_eda_min_peaks_value = self.history_eda_min_peak_value

        if len(new_eda_max_peaks_idx) == 0:
            self.previous_history_eda_max_peak_idx = self.history_eda_max_peak_idx.copy()
            self.previous_history_eda_min_peak_idx = self.history_eda_min_peak_idx.copy()

            return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array, empty_array

        # Initialize lists for response parameters
        amplitude = []
        duration = []
        level_scr = []
        timestamp = []
        after_slope = []

        coeffcubicA = []
        coeffcubicB = []
        coefficientCubic = []

        # Process each new peak
        for i in range(len(new_eda_max_peaks_idx)):
            value = round((new_eda_max_peaks_value[i] - new_eda_min_peaks_value[i]), 3)
            dt = round((new_eda_max_peaks_idx[i] - new_eda_min_peaks_idx[i]), 3)

            amplitude.append(value)
            duration.append(dt)
            level_scr.append(round(new_eda_min_peaks_value[i], 3))
            timestamp.append(round(new_eda_min_peaks_idx[i], 3))

            indice_max_peak = np.where(
                (self.time_window >= (new_eda_max_peaks_idx[i] - 1 / (self.fs * 2)))
                & (self.time_window <= (new_eda_max_peaks_idx[i] + 1 / (self.fs * 2)))
            )[0]
            indice_min_peak = np.where(
                (self.time_window >= (new_eda_min_peaks_idx[i] - 1 / (self.fs * 2)))
                & (self.time_window <= (new_eda_min_peaks_idx[i] + 1 / (self.fs * 2)))
            )[0]
            indice_max_peak = int(indice_max_peak[0])
            indice_min_peak = int(indice_min_peak[0])

            after_slope.append(
                2
                * round(
                    self.eda_window[int(indice_max_peak + self.fs / 2)] - new_eda_max_peaks_value[i],
                    3,
                )
            )

            coefficientCubic = self._interpolate_polynomial3(
                round(new_eda_min_peaks_idx[i] - self.time_window[0], 3),  # x1
                round(self.eda_window[int(indice_min_peak)], 3),  # y1
                round(new_eda_min_peaks_idx[i] + (1 / 4) * dt / 5, 3) - round(new_eda_min_peaks_idx[i], 3),  # x1a
                round(
                    self.eda_window[int(indice_min_peak + int((1 / 4) * dt / 5 * self.fs))],
                    3,
                )
                - round(self.eda_window[int(indice_min_peak)], 3),  # y1a
                round(new_eda_min_peaks_idx[i] + (2 / 4) * dt / 5, 3) - round(new_eda_min_peaks_idx[i], 3),  # x1b
                round(
                    self.eda_window[int(indice_min_peak + int((2 / 4) * dt / 5 * self.fs))],
                    3,
                )
                - round(self.eda_window[int(indice_min_peak)], 3),  # y1b
                round(new_eda_min_peaks_idx[i] + (3 / 4) * dt / 5, 3) - round(new_eda_min_peaks_idx[i], 3),  # x1c
                round(
                    self.eda_window[int(indice_min_peak + int((3 / 4) * dt / 5 * self.fs))],
                    3,
                )
                - round(self.eda_window[int(indice_min_peak)], 3),  # y1c
                round(new_eda_max_peaks_idx[i], 3) - round(new_eda_min_peaks_idx[i], 3),  # x2
                round(self.eda_window[int(indice_max_peak)], 3)
                - round(self.eda_window[int(indice_min_peak)], 3),  # y2
                round(new_eda_max_peaks_idx[i] - (1 / 4) * dt / 5, 3) - round(new_eda_min_peaks_idx[i], 3),  # x2a
                round(
                    self.eda_window[int(indice_max_peak - int((1 / 4) * dt / 5 * self.fs))],
                    3,
                )
                - round(self.eda_window[int(indice_min_peak)], 3),  # y2a
                round(new_eda_max_peaks_idx[i] - (2 / 4) * dt / 5, 3) - round(new_eda_min_peaks_idx[i], 3),  # x2b
                round(
                    self.eda_window[int(indice_max_peak - int((2 / 4) * dt / 5 * self.fs))],
                    3,
                )
                - round(self.eda_window[int(indice_min_peak)], 3),  # y2b
                round(new_eda_max_peaks_idx[i] - (3 / 4) * dt / 5, 3) - round(new_eda_min_peaks_idx[i], 3),  # x2c
                round(
                    self.eda_window[int(indice_max_peak - int((3 / 4) * dt / 5 * self.fs))],
                    3,
                )
                - round(self.eda_window[int(indice_min_peak)], 3),  # y2c
                round(new_eda_min_peaks_idx[i] + 1 * dt / 5, 3) - round(new_eda_min_peaks_idx[i], 3),  # xa
                round(self.eda_window[int(indice_min_peak + int(1 * dt / 5 * self.fs))], 3)
                - round(self.eda_window[int(indice_min_peak)], 3),  # ya
                round(new_eda_min_peaks_idx[i] + 2 * dt / 5, 3) - round(new_eda_min_peaks_idx[i], 3),  # xb
                round(self.eda_window[int(indice_min_peak + int(2 * dt / 5 * self.fs))], 3)
                - round(self.eda_window[int(indice_min_peak)], 3),  # yb
                round(new_eda_min_peaks_idx[i] + 3 * dt / 5, 3) - round(new_eda_min_peaks_idx[i], 3),  # xc
                round(self.eda_window[int(indice_min_peak + int(3 * dt / 5 * self.fs))], 3)
                - round(self.eda_window[int(indice_min_peak)], 3),  # yc
                round(new_eda_min_peaks_idx[i] + 4 * dt / 5, 3) - round(new_eda_min_peaks_idx[i], 3),  # xd
                round(self.eda_window[int(indice_min_peak + int(4 * dt / 5 * self.fs))], 3)
                - round(self.eda_window[int(indice_min_peak)], 3),  # yd
            )

            # Store coefficientCubics for plotting and debugging
            coeffcubicA.append(coefficientCubic[0])
            coeffcubicB.append(coefficientCubic[1])

        # Update previous peak history
        self.previous_history_eda_max_peak_idx = self.history_eda_max_peak_idx.copy()
        self.previous_history_eda_min_peak_idx = self.history_eda_min_peak_idx.copy()

        return amplitude, duration, level_scr, timestamp, after_slope, coeffcubicA, coeffcubicB

    def _get_eda_values(self, eda_peaks_min_idx, eda_peaks_max_idx, eda_window):
        """
        Extracts the EDA values at the specified indices of the minimum and maximum peaks.

        Args:
            eda_peaks_min_idx (list or np.array): Indices of the minimum peaks in the EDA signal.
            eda_peaks_max_idx (list or np.array): Indices of the maximum peaks in the EDA signal.
            eda_window (np.array): Array containing the EDA values over time.

        Returns:
            np.array, np.array: Two arrays, one for the values at the minimum peak
                                indices and another for the values at the maximum peak indices.
        """

        empty_array = np.array([], dtype=float)

        # Check if any minimum or maximum peak indices are provided
        if not eda_peaks_min_idx.any() or not eda_peaks_max_idx.any():
            return empty_array, empty_array

        # Ensure indices and window are numpy arrays
        eda_peaks_min_idx = np.array(eda_peaks_min_idx, dtype=int)
        eda_peaks_max_idx = np.array(eda_peaks_max_idx, dtype=int)
        eda_window = np.array(eda_window)

        # Validate indices to avoid out-of-bounds errors
        valid_min_indices = (eda_peaks_min_idx >= 0) & (eda_peaks_min_idx < len(eda_window))
        valid_max_indices = (eda_peaks_max_idx >= 0) & (eda_peaks_max_idx < len(eda_window))

        # Apply masks to get only the valid indices
        valid_min_indices_array = eda_peaks_min_idx[valid_min_indices]
        valid_max_indices_array = eda_peaks_max_idx[valid_max_indices]

        # Extract the corresponding values from the EDA window
        peak_min_values = eda_window[valid_min_indices_array]
        peak_max_values = eda_window[valid_max_indices_array]

        return peak_min_values, peak_max_values

    def update_eda_peak(self, eda_data, time_data):
        """
        Main interface method for updating EDA data and locating EDA response.

        This method takes new EDA data and corresponding timestamps, updates the
        sliding windows, applies filtering and EDA peak detection, and localizes EDA
        response.

        Args:
            eda_data (array-like): New EDA signal data points.
            time_data (array-like): Corresponding timestamps.

        Returns:
            tuple: (amplitude_tosend, duration_tosend, levelSCR_tosend,
                    timestamp_tosend, x_curve_tosend, y_curve_tosend)
                Lists of localized EDA response data to be sent, or empty lists if no valid EDA responses are detected.
        """
        empty_array = np.array([], dtype=float)

        # Convert input data to numpy arrays for consistency
        eda_data = np.asarray(eda_data, dtype=np.float32)
        time_data = np.asarray(time_data, dtype=np.float32)

        # Check for valid input data
        if not time_data.any() or not eda_data.any():
            return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array

        # Handle initial negative time values
        if time_data[0] < 0:
            start_idx = np.argmax(time_data > 0)
            eda_data = eda_data[start_idx:]
            time_data = time_data[start_idx:]

        # Check for measurement discontinuity
        if abs(time_data[0] - self.time_window[-1]) > TIME_GAP_THRESHOLD:
            logger.info(f"EDA Measurement discontinuity detected at {time_data[0]} vs {self.time_window[-1]}")
            self._reinit_history(time_data[0])

        # Update sliding windows with new data
        self.eda_window = np.roll(self.eda_window, -len(eda_data))
        self.eda_window[-len(eda_data) :] = eda_data

        # Fill missing values in the EDA window (if any) by propagating the next non-zero value
        for i in range(len(self.eda_window) - 2, -1, -1):
            if self.eda_window[i] == 0:
                for j in range(i + 1, len(self.eda_window)):
                    if self.eda_window[j] != 0:
                        self.eda_window[i] = self.eda_window[j]
                        break

        # Update the time window
        self.time_window = np.roll(self.time_window, -len(time_data))
        self.time_window[-len(time_data) :] = time_data

        # Apply Savitzky-Golay smoothing filter to the EDA signal
        smooth_eda = self._savitzky_golay_filter(self.eda_window)

        # Derivative of the smoothed EDA signal for peak detection
        filtered_eda = np.diff(smooth_eda)

        # EDA peak detection (minima and maxima)
        eda_peaks_min_idx, eda_peaks_max_idx = self._detect_eda_peaks(filtered_eda)

        # Filter and remove edge effects from the detected peaks
        eda_peaks_min_idx, eda_peaks_max_idx = self._filter_indices(
            self.eda_window, eda_peaks_min_idx, eda_peaks_max_idx, 0.01, 0.5, 0
        )

        # Get the values of the detected peaks from the EDA signal
        eda_peaks_min_val, eda_peaks_max_val = self._get_eda_values(eda_peaks_min_idx, eda_peaks_max_idx, smooth_eda)

        # Update peak history if new peaks are found
        if eda_peaks_min_val.any() and eda_peaks_max_val.any():
            self._update_peak_history(
                self.time_window[eda_peaks_min_idx],
                self.time_window[eda_peaks_max_idx],
                eda_peaks_min_val,
                eda_peaks_max_val,
            )

        # Initialize lists for the EDA response data
        amplitude = []
        duration = []
        levelSCR = []
        timestamp = []
        after_slope = []
        coeffCubicA = []
        coeffCubicB = []

        # Calculate the EDA response (amplitude, duration, level, etc.)
        amplitude, duration, levelSCR, timestamp, after_slope, coeffCubicA, coeffCubicB = (
            self._calculate_eda_response()
        )

        # Initialize lists to store the results to be sent
        amplitude_tosend = []
        duration_tosend = []
        levelSCR_tosend = []
        timestamp_tosend = []

        # Apply decision tree to determine which EDA responses to send
        for i in range(len(amplitude)):
            # Calculate additional features for decision tree prediction
            slope = amplitude[i] / duration[i]
            A_Level = amplitude[i] / levelSCR[i]
            A_Level_duration = A_Level / duration[i]
            inflection_slope = 3 * coeffCubicA[i] * ((-coeffCubicB[i]) / (3 * coeffCubicA[i])) ** 2 + 2 * coeffCubicB[
                i
            ] * (-coeffCubicB[i]) / (3 * coeffCubicA[i])
            cubic_A_B = coeffCubicA[i] / coeffCubicB[i]
            squared_A = 3 * coeffCubicA[i]
            squared_B = 2 * coeffCubicB[i]
            squared_A_B = squared_A / squared_B

            # Create the feature vector for decision tree prediction
            features = [
                round(amplitude[i], 4),
                round(duration[i], 4),
                round(levelSCR[i], 4),
                round(slope, 4),
                round(A_Level, 4),
                round(A_Level_duration, 4),
                round(coeffCubicB[i], 4),
                round(coeffCubicA[i], 4),
                round(after_slope[i], 4),
                round(inflection_slope, 4),
                round(cubic_A_B, 4),
                round(squared_B, 4),
                round(squared_A, 4),
                round(squared_A_B, 4),
            ]

            # Predict the class using the decision tree model
            prediction = eda_decisiontree.predict(features, model="embc2025")

            # If the prediction is positive, add the data to the lists to send
            if prediction:
                amplitude_tosend.append(amplitude[i])
                duration_tosend.append(duration[i])
                levelSCR_tosend.append(levelSCR[i])
                timestamp_tosend.append(timestamp[i])

        return amplitude_tosend, duration_tosend, levelSCR_tosend, timestamp_tosend
