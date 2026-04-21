"""Main application window — contains training and prediction tabs."""

import tkinter as tk
from tkinter import ttk

from core.config import TrainOptions, PredictOptions
from gui.output_panel import OutputPanel, StatusBar
from gui.train_tab import TrainTab
from gui.predict_tab import PredictTab


class App(tk.Tk):
    """Top-level tkinter window for the Pedestrian Detection application."""

    WIDTH = 900
    HEIGHT = 700

    def __init__(self):
        super().__init__()
        self.title("Detect Pedestrian – Python Port")
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.minsize(800, 600)

        # Shared state
        self.classifier = None          # trained sklearn model
        self.train_options = TrainOptions()
        self.predict_options = PredictOptions()

        self._build_ui()

    def _build_ui(self):
        # ── Notebook (tabs) ──
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.train_tab = TrainTab(self.notebook, app=self)
        self.predict_tab = PredictTab(self.notebook, app=self)

        self.notebook.add(self.train_tab, text="Train")
        self.notebook.add(self.predict_tab, text="Predict")

        # ── Output log ──
        self.output_panel = OutputPanel(self)
        self.output_panel.pack(fill=tk.BOTH, expand=False, padx=4, pady=(0, 2))

        # ── Status bar ──
        self.status_bar = StatusBar(self)
        self.status_bar.pack(fill=tk.X, padx=4, pady=(0, 4))

    # ── Shared helpers for tabs ──

    def set_running(self, message: str):
        """Put the UI into 'busy' state while training or predicting."""
        self.status_bar.set_status(message)
        self.train_tab.train_btn.configure(state="disabled")
        self.predict_tab.detect_btn.configure(state="disabled")

    def set_ready(self, message: str, enable_predict: bool = False):
        """Restore the UI after a background task completes."""
        self.status_bar.set_status(message)
        self.train_tab.train_btn.configure(state="normal")
        if enable_predict:
            self.predict_tab.enable_detect(True)

    def run(self):
        """Start the tkinter event loop."""
        self.mainloop()
