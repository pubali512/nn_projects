"""Train tab: UI for configuring and launching training."""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

from core.config import TrainOptions
from core.trainer import PedestrianTrainer
from gui.result_dialogs import ResultDialog
from utils.file_utils import dir_contains_images, ensure_dir, create_or_clean_dir


class TrainTab(ttk.Frame):
    """GUI tab for configuring training parameters and running training."""

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self.app = app

        self._build_ui()
        self._init_defaults()

    def _build_ui(self):
        # ── Train Data Files group ──
        data_frame = ttk.LabelFrame(self, text="Train Data Files")
        data_frame.pack(fill=tk.X, padx=8, pady=4)

        # Pedestrian training set size
        row = 0
        ttk.Label(data_frame, text="Size of Pedestrian Training Set").grid(
            row=row, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.ped_size_var = tk.StringVar(value="50")
        self.ped_size_entry = ttk.Entry(
            data_frame, textvariable=self.ped_size_var, width=24
        )
        self.ped_size_entry.grid(row=row, column=1, padx=8, pady=4)

        # Pedestrian directory
        row = 1
        ttk.Label(data_frame, text="Pedestrian Training Image Directory").grid(
            row=row, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.ped_dir_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.ped_dir_var, width=60).grid(
            row=row, column=1, padx=8, pady=4
        )
        ttk.Button(data_frame, text="Browse",
                    command=self._browse_ped_dir).grid(
            row=row, column=2, padx=8, pady=4
        )

        # Non-pedestrian training set size
        row = 2
        ttk.Label(data_frame, text="Size of Non Pedestrian Training Set").grid(
            row=row, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.non_ped_size_var = tk.StringVar(value="50")
        self.non_ped_size_entry = ttk.Entry(
            data_frame, textvariable=self.non_ped_size_var, width=24
        )
        self.non_ped_size_entry.grid(row=row, column=1, padx=8, pady=4)

        # Non-pedestrian directory
        row = 3
        ttk.Label(data_frame, text="Non Pedestrian Training Image Directory").grid(
            row=row, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.non_ped_dir_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.non_ped_dir_var, width=60).grid(
            row=row, column=1, padx=8, pady=4
        )
        ttk.Button(data_frame, text="Browse",
                    command=self._browse_non_ped_dir).grid(
            row=row, column=2, padx=8, pady=4
        )

        # Training output directory
        row = 4
        ttk.Label(data_frame, text="Training Output Directory").grid(
            row=row, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.train_outdir_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.train_outdir_var, width=60).grid(
            row=row, column=1, padx=8, pady=4
        )
        ttk.Button(data_frame, text="Browse",
                    command=self._browse_train_outdir).grid(
            row=row, column=2, padx=8, pady=4
        )

        # ── Non-Ped Sample Parameters group ──
        sample_frame = ttk.LabelFrame(
            self, text="Non Pedestrian Training Sample Parameters"
        )
        sample_frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(sample_frame, text="Sample Image Type").grid(
            row=0, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.sample_type_var = tk.StringVar(value="Cut")
        self.sample_type_combo = ttk.Combobox(
            sample_frame, textvariable=self.sample_type_var,
            values=["Cut", "Scaled"], state="readonly", width=20
        )
        self.sample_type_combo.grid(row=0, column=1, padx=8, pady=4)
        self.sample_type_combo.bind("<<ComboboxSelected>>",
                                     self._on_sample_type_changed)

        ttk.Label(sample_frame, text="Nr. of Cuts").grid(
            row=1, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.nr_cuts_var = tk.StringVar(value="1")
        self.nr_cuts_combo = ttk.Combobox(
            sample_frame, textvariable=self.nr_cuts_var,
            values=["1", "2", "3", "4"], state="readonly", width=20
        )
        self.nr_cuts_combo.grid(row=1, column=1, padx=8, pady=4)

        ttk.Label(sample_frame, text="Size of Cuts (w.r.t. Pedestrian Image)").grid(
            row=2, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.cut_size_var = tk.StringVar(value="4x")
        self.cut_size_combo = ttk.Combobox(
            sample_frame, textvariable=self.cut_size_var,
            values=["1x", "2x", "3x", "4x", "5x", "6x"],
            state="readonly", width=20
        )
        self.cut_size_combo.grid(row=2, column=1, padx=8, pady=4)

        # ── Training Parameters group ──
        param_frame = ttk.LabelFrame(self, text="Training Parameters")
        param_frame.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(param_frame, text="Feature Vector Type").grid(
            row=0, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.feature_type_var = tk.StringVar(value="HOG")
        self.feature_type_combo = ttk.Combobox(
            param_frame, textvariable=self.feature_type_var,
            values=["Greyscale", "HOG"], state="readonly", width=20
        )
        self.feature_type_combo.grid(row=0, column=1, padx=8, pady=4)
        self.feature_type_combo.bind("<<ComboboxSelected>>",
                                      self._on_feature_type_changed)

        ttk.Label(param_frame, text="HOG Block Size").grid(
            row=1, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.hog_block_var = tk.StringVar(value="12x12")
        self.hog_block_combo = ttk.Combobox(
            param_frame, textvariable=self.hog_block_var,
            values=["12x12", "18x18"], state="readonly", width=20
        )
        self.hog_block_combo.grid(row=1, column=1, padx=8, pady=4)

        # Train button
        self.train_btn = ttk.Button(
            param_frame, text="Train Classifier",
            command=self._on_train_click
        )
        self.train_btn.grid(row=2, column=1, padx=8, pady=8)

    def _init_defaults(self):
        """Set default paths based on the data directory."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base, "data")

        # If the downloaded Daimler data exists, use split 1
        ped_dir = os.path.join(data_dir, "1", "ped_examples")
        non_ped_dir = os.path.join(data_dir, "1", "non-ped_examples")
        output_dir = os.path.join(base, "output")

        if os.path.isdir(ped_dir):
            self.ped_dir_var.set(ped_dir)
        if os.path.isdir(non_ped_dir):
            self.non_ped_dir_var.set(non_ped_dir)

        self.train_outdir_var.set(output_dir)

    # ── Event handlers ──

    def _browse_ped_dir(self):
        d = filedialog.askdirectory(title="Select Pedestrian Training Directory")
        if d:
            self.ped_dir_var.set(d)

    def _browse_non_ped_dir(self):
        d = filedialog.askdirectory(title="Select Non-Pedestrian Training Directory")
        if d:
            self.non_ped_dir_var.set(d)

    def _browse_train_outdir(self):
        d = filedialog.askdirectory(title="Select Training Output Directory")
        if d:
            self.train_outdir_var.set(d)

    def _on_sample_type_changed(self, event=None):
        if self.sample_type_var.get() == "Scaled":
            self.nr_cuts_combo.configure(state="disabled")
            self.cut_size_combo.configure(state="disabled")
        else:
            self.nr_cuts_combo.configure(state="readonly")
            self.cut_size_combo.configure(state="readonly")

    def _on_feature_type_changed(self, event=None):
        if self.feature_type_var.get() == "Greyscale":
            self.hog_block_combo.configure(state="disabled")
        else:
            self.hog_block_combo.configure(state="readonly")

    def _on_train_click(self):
        """Validate inputs and run training in a background thread."""
        # Build TrainOptions from GUI
        opts = self.app.train_options

        try:
            opts.ped_train_data_size = int(self.ped_size_var.get())
            opts.non_ped_train_data_size = int(self.non_ped_size_var.get())
            if opts.ped_train_data_size <= 0 or opts.non_ped_train_data_size <= 0:
                raise ValueError("Must be positive")
        except ValueError:
            messagebox.showerror(
                "Error", "Training set sizes must be positive integers."
            )
            return

        opts.ped_train_data_path = self.ped_dir_var.get()
        if not dir_contains_images(opts.ped_train_data_path):
            messagebox.showerror(
                "Error",
                f"Pedestrian directory is empty or invalid:\n{opts.ped_train_data_path}"
            )
            return

        opts.non_ped_train_data_path = self.non_ped_dir_var.get()
        if not dir_contains_images(opts.non_ped_train_data_path):
            messagebox.showerror(
                "Error",
                f"Non-pedestrian directory is empty or invalid:\n{opts.non_ped_train_data_path}"
            )
            return

        opts.scaled_image = (self.sample_type_var.get() == "Scaled")
        if not opts.scaled_image:
            opts.cuts = int(self.nr_cuts_var.get())
            cut_size_str = self.cut_size_var.get()
            opts.cut_size_factor = int(cut_size_str.replace("x", ""))

        opts.feature_vec_type = 0 if self.feature_type_var.get() == "Greyscale" else 1
        if opts.feature_vec_type == 1:
            block_str = self.hog_block_var.get()
            block_val = int(block_str.split("x")[0])
            opts.hog_block_size = block_val
            opts.hog_cell_size = block_val // 2

        opts.training_output_dir = self.train_outdir_var.get()
        if not ensure_dir(opts.training_output_dir):
            messagebox.showerror(
                "Error",
                f"Could not create output directory:\n{opts.training_output_dir}"
            )
            return

        # Disable UI and run training in background
        self.app.status_bar.reset()
        self.app.set_running("Training classifier ...")
        self.train_btn.configure(state="disabled")

        def train_thread():
            try:
                trainer = PedestrianTrainer(
                    opts,
                    log_callback=self._thread_safe_log,
                    progress_callback=self._thread_safe_progress
                )
                classifier = trainer.train()

                # Store classifier in app for prediction
                self.app.classifier = classifier
                self.app.train_options = opts

                self.app.after(0, lambda: self._training_done(True))
            except Exception as e:
                err_msg = str(e)
                self.app.after(0, lambda: self._training_done(False, err_msg))

        t = threading.Thread(target=train_thread, daemon=True)
        t.start()

    def _thread_safe_log(self, msg: str):
        self.app.after(0, lambda: self.app.output_panel.log(msg))

    def _thread_safe_progress(self, frac: float):
        self.app.after(0, lambda: self.app.status_bar.set_progress(frac))

    def _training_done(self, success: bool, error_msg: str = ""):
        self.train_btn.configure(state="normal")
        self.app.status_bar.reset()

        if success:
            self.app.set_ready("Ready for prediction", enable_predict=True)
            opts = self.app.train_options
            feature_type = "HOG" if opts.feature_vec_type == 1 else "Greyscale"
            ResultDialog(
                self.app,
                title="Training Complete",
                success=True,
                status_text="Training Successful",
                subtitle="Gaussian Naive Bayes classifier is ready.",
                metrics={
                    "Feature Type": feature_type,
                    "Feature Dimensions": str(opts.feature_vector_size),
                    "Pedestrian Samples": str(opts.ped_train_data_size),
                    "Non-Pedestrian Samples": str(opts.non_ped_train_data_size),
                },
                detail="Switch to the Predict tab to run pedestrian detection."
            )
        else:
            self.app.set_ready("Bayesian classifier not trained",
                               enable_predict=False)
            ResultDialog(
                self.app,
                title="Training Failed",
                success=False,
                status_text="Training Failed",
                subtitle="The classifier could not be trained.",
                error=error_msg
            )
