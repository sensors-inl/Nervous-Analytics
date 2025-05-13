import os
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .eda_analyzer import EDAAnalyzer


class EDAAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EDA Analyzer")
        self.root.geometry("1000x700")

        # Variables
        self.file_path = None
        self.analyzer = None
        self.data = None
        self.results = None

        # Main interface
        self.create_widgets()

    def create_widgets(self):
        # Top frame for controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        # Button to load CSV file
        self.load_button = ttk.Button(control_frame, text="Load CSV File", command=self.load_csv)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Label to display file path
        self.file_label = ttk.Label(control_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # Button to analyze data
        self.analyze_button = ttk.Button(control_frame, text="Analyze", command=self.analyze_data, state=tk.DISABLED)
        self.analyze_button.pack(side=tk.LEFT, padx=5)

        # Button to export results
        self.export_button = ttk.Button(
            control_frame, text="Export to Excel", command=self.export_to_excel, state=tk.DISABLED
        )
        self.export_button.pack(side=tk.LEFT, padx=5)

        # Frame for parameters
        param_frame = ttk.LabelFrame(self.root, text="Parameters", padding="10")
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        # Parameter fs (sampling frequency)
        ttk.Label(param_frame, text="Sampling Frequency (Hz):").grid(row=0, column=0, sticky=tk.W)
        self.fs_var = tk.IntVar(value=8)
        ttk.Spinbox(param_frame, from_=1, to=100, textvariable=self.fs_var, width=5).grid(
            row=0, column=1, padx=5, pady=2
        )

        # Parameter window_duration
        ttk.Label(param_frame, text="Window Duration (s):").grid(row=1, column=0, sticky=tk.W)
        self.window_var = tk.IntVar(value=20)
        ttk.Spinbox(param_frame, from_=5, to=100, textvariable=self.window_var, width=5).grid(
            row=1, column=1, padx=5, pady=2
        )

        # Parameter history_size
        ttk.Label(param_frame, text="History Size (s):").grid(row=2, column=0, sticky=tk.W)
        self.history_var = tk.IntVar(value=20)
        ttk.Spinbox(param_frame, from_=5, to=100, textvariable=self.history_var, width=5).grid(
            row=2, column=1, padx=5, pady=2
        )

        # Parameter batch_size_seconds
        ttk.Label(param_frame, text="Batch Size (s):").grid(row=3, column=0, sticky=tk.W)
        self.batch_size_seconds_var = tk.DoubleVar(value=5.0)
        ttk.Spinbox(
            param_frame, from_=0.1, to=60.0, increment=0.1, textvariable=self.batch_size_seconds_var, width=5
        ).grid(row=3, column=1, padx=5, pady=2)

        # CSV Columns
        ttk.Label(param_frame, text="EDA Column:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.eda_col_var = tk.StringVar(value="EDA (uS)")
        ttk.Entry(param_frame, textvariable=self.eda_col_var, width=15).grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(param_frame, text="Time Column:").grid(row=1, column=2, sticky=tk.W, padx=(20, 0))
        self.time_col_var = tk.StringVar(value="Time (s)")
        ttk.Entry(param_frame, textvariable=self.time_col_var, width=15).grid(row=1, column=3, padx=5, pady=2)

        # Notebook to display data and results
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab for raw data
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Raw Data")

        # Tab for results
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")

        # Tab for graph
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="Graph")

        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Ready")

        # Progress bar
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # Create tables for data and results
        self.create_data_table()
        self.create_results_table()

    def create_data_table(self):
        # Frame with scroll for data table
        frame = ttk.Frame(self.data_frame)
        frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        h_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        v_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Treeview to display data
        self.data_table = ttk.Treeview(
            frame, selectmode="extended", xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set
        )
        self.data_table.pack(fill=tk.BOTH, expand=True)

        # Configure scrollbars
        h_scroll.config(command=self.data_table.xview)
        v_scroll.config(command=self.data_table.yview)

    def create_results_table(self):
        # Frame with scroll for results table
        frame = ttk.Frame(self.results_frame)
        frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        h_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        v_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Treeview to display results
        self.results_table = ttk.Treeview(
            frame, selectmode="extended", xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set
        )
        self.results_table.pack(fill=tk.BOTH, expand=True)

        # Configure scrollbars
        h_scroll.config(command=self.results_table.xview)
        v_scroll.config(command=self.results_table.yview)

        # Columns for results
        self.results_table["columns"] = ("timestamp", "amplitude", "duration", "level_scr")
        self.results_table.column("#0", width=0, stretch=tk.NO)
        self.results_table.column("timestamp", anchor=tk.CENTER, width=120)
        self.results_table.column("amplitude", anchor=tk.CENTER, width=120)
        self.results_table.column("duration", anchor=tk.CENTER, width=120)
        self.results_table.column("level_scr", anchor=tk.CENTER, width=120)

        # Headers
        self.results_table.heading("#0", text="", anchor=tk.CENTER)
        self.results_table.heading("timestamp", text="Timestamp", anchor=tk.CENTER)
        self.results_table.heading("amplitude", text="Amplitude", anchor=tk.CENTER)
        self.results_table.heading("duration", text="Duration", anchor=tk.CENTER)
        self.results_table.heading("level_scr", text="SCR Level", anchor=tk.CENTER)

    def load_csv(self):
        # Open dialog to select a CSV file
        file_path = filedialog.askopenfilename(
            title="Select CSV File", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )

        if not file_path:
            return

        try:
            # Read CSV file
            self.file_path = file_path
            self.file_label.config(text=os.path.basename(file_path))

            # Load data
            self.data = pd.read_csv(file_path, sep=";")

            # Display data in table
            self.display_data(self.data)

            # Enable analyze button
            self.analyze_button.config(state=tk.NORMAL)

            self.status_var.set(f"File loaded: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Unable to load file: {str(e)}")
            self.status_var.set("Error loading file")

    def display_data(self, data):
        # Reset table
        for item in self.data_table.get_children():
            self.data_table.delete(item)

        # Configure columns
        columns = list(data.columns)
        self.data_table["columns"] = columns

        # Configure columns display
        self.data_table.column("#0", width=0, stretch=tk.NO)
        for col in columns:
            self.data_table.column(col, anchor=tk.CENTER, width=100)
            self.data_table.heading(col, text=col, anchor=tk.CENTER)

        # Limit display to first 1000 rows for performance
        display_data = data.head(1000)

        # Add data
        for i, row in display_data.iterrows():
            values = [row[col] for col in columns]
            self.data_table.insert("", tk.END, text="", values=values)

        self.status_var.set(f"Displaying first {len(display_data)} rows (out of {len(data)} total)")

    def analyze_data(self):
        # Check that specified columns exist
        eda_col = self.eda_col_var.get()
        time_col = self.time_col_var.get()

        if eda_col not in self.data.columns or time_col not in self.data.columns:
            messagebox.showerror("Error", "Specified columns do not exist in the CSV file")
            return

        # Disable buttons during analysis
        self.analyze_button.config(state=tk.DISABLED)
        self.load_button.config(state=tk.DISABLED)

        # Update status
        self.status_var.set("Analysis in progress...")
        self.progress["value"] = 0

        # Launch analysis in a separate thread to avoid blocking the interface
        threading.Thread(target=self._run_analysis, args=(eda_col, time_col)).start()

    def _run_analysis(self, eda_col, time_col):
        try:
            # Create analyzer with specified parameters
            fs = self.fs_var.get()
            window_duration = self.window_var.get()
            history_size = self.history_var.get()
            batch_size_seconds = self.batch_size_seconds_var.get()

            self.analyzer = EDAAnalyzer(fs=fs, window_duration=window_duration, history_size=history_size)

            # Extract EDA and time data
            eda_data = self.data[eda_col].values
            time_data = self.data[time_col].values

            # Calculate batch size in samples based on frequency and seconds
            batch_size_samples = int(batch_size_seconds * fs)

            # Make sure batch size is at least 1 sample
            batch_size_samples = max(1, batch_size_samples)

            # Initialize lists to store results
            all_amplitudes = []
            all_durations = []
            all_levels = []
            all_timestamps = []

            # Process data in batches to show progress
            num_batches = len(eda_data) // batch_size_samples + (1 if len(eda_data) % batch_size_samples > 0 else 0)

            for i in range(num_batches):
                # Update progress
                self.root.after(0, lambda val=int((i / num_batches) * 100): self.progress.config(value=val))

                # Calculate indices for this batch
                start_idx = i * batch_size_samples
                end_idx = min((i + 1) * batch_size_samples, len(eda_data))

                # Process this batch of data
                batch_eda = eda_data[start_idx:end_idx]
                batch_time = time_data[start_idx:end_idx]

                # Check if electrodes are connected
                min_value = min(batch_eda)
                if min_value < 0.2:
                    continue

                # Analyze this batch
                amplitude, duration, level_scr, timestamp = self.analyzer.update_eda_peak(batch_eda, batch_time)

                # Add results to our lists
                if len(amplitude) > 0:
                    all_amplitudes.extend(amplitude)
                    all_durations.extend(duration)
                    all_levels.extend(level_scr)
                    all_timestamps.extend(timestamp)

            # Create DataFrame with results
            results = pd.DataFrame(
                {
                    "timestamp": all_timestamps,
                    "amplitude": all_amplitudes,
                    "duration": all_durations,
                    "level_scr": all_levels,
                }
            )

            self.results = results

            # Display results in the user interface
            self.root.after(0, self._display_results)

            # Create and display the graph
            self.root.after(0, self._plot_results)

        except Exception as error:
            e = error
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
        finally:
            # Re-enable buttons
            self.root.after(0, lambda: self.analyze_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.load_button.config(state=tk.NORMAL))
            self.root.after(
                0, lambda: self.export_button.config(state=tk.NORMAL if self.results is not None else tk.DISABLED)
            )
            self.root.after(0, lambda: self.progress.config(value=100))

    def _display_results(self):
        # Clear results table
        for item in self.results_table.get_children():
            self.results_table.delete(item)

        # Display results
        if self.results is not None and not self.results.empty:
            for i, row in self.results.iterrows():
                values = (
                    f"{row['timestamp']:.3f}",
                    f"{row['amplitude']:.3f}",
                    f"{row['duration']:.3f}",
                    f"{row['level_scr']:.3f}",
                )
                self.results_table.insert("", tk.END, text="", values=values)

            self.status_var.set(f"Analysis complete. {len(self.results)} EDA peaks detected.")
        else:
            self.status_var.set("Analysis complete. No EDA peaks detected.")

    def _plot_results(self):
        # Clean graph frame
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Create figure with one subplot
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))

        # Graph 1: Raw EDA signal with detected peaks
        ax1.plot(
            self.data[self.time_col_var.get()], self.data[self.eda_col_var.get()], "b-", alpha=0.7, label="EDA Signal"
        )

        if self.results is not None and not self.results.empty:
            # Mark detected peaks
            for _, row in self.results.iterrows():
                timestamp = row["timestamp"]
                level = row["level_scr"]
                amplitude = row["amplitude"]

                # Mark minimum point
                ax1.plot(timestamp, level, "ro", markersize=5)

                # Mark maximum point
                ax1.plot(timestamp, level + amplitude, "go", markersize=5)

                # Line connecting the two points
                ax1.plot([timestamp, timestamp], [level, level + amplitude], "r-", alpha=0.5)

        ax1.set_title("EDA Signal with detected peaks")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("EDA Amplitude")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Display graph in interface
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def export_to_excel(self):
        if self.results is None or self.results.empty:
            messagebox.showinfo("Information", "No results to export")
            return

        # Open dialog to select Excel file location
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")],
            initialfile=f"EDA_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        )

        if not file_path:
            return

        try:
            # Export results to Excel
            self.results.to_excel(file_path, index=False)

            self.status_var.set(f"Results exported to: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"Results have been successfully exported to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Unable to export results: {str(e)}")
            self.status_var.set("Error during export")


def run_gui():
    root = tk.Tk()
    EDAAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
