"""Output log panel + status bar + progress bar for the GUI."""

import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime


class OutputPanel(ttk.LabelFrame):
    """Scrollable text area for log messages (bottom of the window)."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="Output", **kwargs)

        # Top bar with Clear button
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=4, pady=(4, 0))
        ttk.Button(btn_frame, text="Clear Log",
                    command=self.clear).pack(side=tk.RIGHT)

        self.text = scrolledtext.ScrolledText(
            self, height=8, state="disabled", wrap=tk.WORD,
            font=("Courier", 10)
        )
        self.text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def log(self, msg: str, level: str = "INFO"):
        """Append a timestamped message to the output log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] {level:8s}: "
        self.text.configure(state="normal")
        self.text.insert(tk.END, prefix + msg + "\n")
        self.text.see(tk.END)
        self.text.configure(state="disabled")

    def clear(self):
        """Clear all log messages."""
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.configure(state="disabled")


class StatusBar(ttk.Frame):
    """Status label + progress bar at the bottom of the window."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.status_var = tk.StringVar(value="Bayesian classifier not trained")
        self.status_label = ttk.Label(
            self, textvariable=self.status_var, anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=8)

        self.progress = ttk.Progressbar(
            self, length=300, mode="determinate"
        )
        self.progress.pack(side=tk.RIGHT, padx=8, pady=2)

    def set_status(self, text: str):
        self.status_var.set(text)

    def set_progress(self, fraction: float):
        """Set progress bar value (0.0 to 1.0)."""
        self.progress["value"] = fraction * 100

    def reset(self):
        self.progress["value"] = 0
