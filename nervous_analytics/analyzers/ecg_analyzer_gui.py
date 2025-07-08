import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .ecg_analyzer import ECGAnalyzer


class ECGAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Analyzer Tool")
        self.root.geometry("1000x700")

        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.status_text = tk.StringVar(value="Ready")
        self.progress_value = tk.DoubleVar(value=0)

        # Create frames
        self.create_widgets()
        self.setup_layout()

        # Set default output file
        self.on_input_file_change()

    def create_widgets(self):
        # Frame for file selection and controls
        self.control_frame = ttk.LabelFrame(self.root, text="Controls")

        # Input file selection
        ttk.Label(self.control_frame, text="Input ECG File:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(self.control_frame, textvariable=self.input_file, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.control_frame, text="Browse...", command=self.browse_input_file).grid(
            row=0, column=2, padx=5, pady=5
        )

        # Output file selection
        ttk.Label(self.control_frame, text="Output HR File:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(self.control_frame, textvariable=self.output_file, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.control_frame, text="Browse...", command=self.browse_output_file).grid(
            row=1, column=2, padx=5, pady=5
        )

        # Process button
        self.process_button = ttk.Button(self.control_frame, text="Process ECG Data", command=self.process_ecg_data)
        self.process_button.grid(row=2, column=1, padx=5, pady=10)

        # Status bar and progress
        self.status_frame = ttk.Frame(self.root)
        ttk.Label(self.status_frame, text="Status:").pack(side=tk.LEFT, padx=5)
        ttk.Label(self.status_frame, textvariable=self.status_text).pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(
            self.status_frame, variable=self.progress_value, length=200, mode="determinate"
        )
        self.progress.pack(side=tk.RIGHT, padx=10)

        # Plot frame
        self.plot_frame = ttk.LabelFrame(self.root, text="ECG and Heart Rate Visualization")
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add toolbar
        self.toolbar_frame = ttk.Frame(self.plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_layout(self):
        # Pack frames
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)

    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            title="Select ECG CSV File", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            self.on_input_file_change()

    def on_input_file_change(self):
        if self.input_file.get():
            input_path = self.input_file.get()
            dirname, basename = os.path.split(input_path)
            filename, ext = os.path.splitext(basename)
            self.output_file.set(os.path.join(dirname, f"{filename}_HR{ext}"))

    def browse_output_file(self):
        filename = filedialog.asksaveasfilename(
            title="Save Heart Rate Data",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            defaultextension=".csv",
        )
        if filename:
            self.output_file.set(filename)

    def find_column_name(self, df, keyword):
        """Find a column name containing the keyword (case insensitive)"""
        for col in df.columns:
            if keyword.lower() in col.lower():
                return col
        return None

    def process_ecg_data(self):
        # Validate files
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input ECG file.")
            return

        if not self.output_file.get():
            messagebox.showerror("Error", "Please specify an output file.")
            return

        try:
            # Load data
            self.status_text.set("Loading ECG data...")
            self.root.update()

            df = pd.read_csv(self.input_file.get(), delimiter=";")

            # Find ECG and Time columns
            ecg_col = self.find_column_name(df, "ECG")
            time_col = self.find_column_name(df, "Time")

            if not ecg_col:
                messagebox.showerror("Error", "Could not find a column containing 'ECG' in the CSV file.")
                self.status_text.set("Error: No ECG column found")
                return

            if not time_col:
                messagebox.showerror("Error", "Could not find a column containing 'Time' in the CSV file.")
                self.status_text.set("Error: No Time column found")
                return

            # Extract data
            ecg_data = df[ecg_col].values
            time_data = df[time_col].values

            # Initialize analyzer
            self.status_text.set("Initializing ECG analyzer...")
            self.root.update()

            # Get sampling frequency
            if len(time_data) > 1:
                fs = int(1.0 / (time_data[1] - time_data[0]))
            else:
                fs = 512  # Default

            analyzer = ECGAnalyzer(fs=fs)

            # Process data in windows
            window_size = 512
            hr_results = []
            hr_times = []
            r_peaks = []

            total_windows = len(ecg_data) // window_size + (1 if len(ecg_data) % window_size else 0)

            for i in range(0, len(ecg_data), window_size):
                # Update progress
                progress_pct = min(100, (i // window_size) / total_windows * 100)
                self.progress_value.set(progress_pct)
                self.status_text.set(f"Processing window {i // window_size + 1}/{total_windows}...")
                self.root.update()

                # Get window data
                end_idx = min(i + window_size, len(ecg_data))
                ecg_window = ecg_data[i:end_idx]
                time_window = time_data[i:end_idx]

                # Skip if window is too small
                if len(ecg_window) < 10:  # Arbitrary small number
                    continue

                # Process window
                hr, hr_timestamp, r_peak_timestamps = analyzer.update_hr(ecg_window, time_window)

                # Store results
                if hr and hr_timestamp:
                    hr_results.extend(hr)
                    hr_times.extend(hr_timestamp)
                if r_peak_timestamps:
                    r_peaks.extend(r_peak_timestamps)

            # Create output dataframe
            self.status_text.set("Saving results...")
            self.root.update()

            hr_df = pd.DataFrame({"Time (s)": hr_times, "HR (BPM)": hr_results})

            # Save to CSV
            hr_df.to_csv(self.output_file.get(), index=False)

            # Plot results
            self.plot_results(ecg_data, time_data, hr_results, hr_times, r_peaks)

            self.status_text.set("Processing completed successfully")
            self.progress_value.set(100)
            messagebox.showinfo("Success", f"ECG processing complete. Results saved to {self.output_file.get()}")

        except Exception as e:
            self.status_text.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            import traceback

            traceback.print_exc()

    def plot_results(self, ecg_data, time_data, hr_results, hr_times, r_peaks):
        """Plot ECG and heart rate results"""
        self.fig.clear()

        # Create subplot for ECG
        ax1 = self.fig.add_subplot(211)
        ax1.plot(time_data, ecg_data, "b-", linewidth=0.5)
        ax1.set_title("ECG Signal")
        ax1.set_ylabel("Amplitude")

        # Mark R-peaks if available
        if r_peaks:
            # Filter r_peaks that are within the time range
            valid_r_peaks = [t for t in r_peaks if time_data[0] <= t <= time_data[-1]]
            y_values = np.interp(valid_r_peaks, time_data, ecg_data)
            ax1.plot(valid_r_peaks, y_values, "ro", markersize=3)

        # Create subplot for heart rate
        ax2 = self.fig.add_subplot(212, sharex=ax1)
        if hr_results and hr_times:
            ax2.plot(hr_times, hr_results, "g-", linewidth=1)
            ax2.set_title("Heart Rate")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Heart Rate (BPM)")

            # Set reasonable y-axis limits for heart rate
            min_hr = max(30, min(hr_results) - 10)
            max_hr = min(200, max(hr_results) + 10)
            ax2.set_ylim(min_hr, max_hr)
        else:
            ax2.set_title("No heart rate data available")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Heart Rate (BPM)")

        self.fig.tight_layout()
        self.canvas.draw()


def run_gui():
    root = tk.Tk()
    ECGAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
