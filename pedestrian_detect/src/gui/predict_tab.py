"""Predict tab: UI for configuring and launching pedestrian detection."""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

from core.config import PredictOptions
from core.predictor import PedestrianPredictor
from core.evaluator import Evaluator
from gui.result_dialogs import ResultDialog, ImageGalleryDialog
from utils.file_utils import dir_contains_images, ensure_dir, create_or_clean_dir


class PredictTab(ttk.Frame):
    """GUI tab for configuring prediction parameters and running detection."""

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self.app = app

        self._build_ui()
        self._init_defaults()

    def _build_ui(self):
        # ── Predict Files group ──
        files_frame = ttk.LabelFrame(self, text="Predict Files")
        files_frame.pack(fill=tk.X, padx=8, pady=4)

        # Read from file checkbox
        self.from_file_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            files_frame, text="Read Test Image Names from File",
            variable=self.from_file_var,
            command=self._on_from_file_changed
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=8, pady=4)

        # Test image list file
        ttk.Label(files_frame, text="Test Image List File").grid(
            row=1, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.test_file_var = tk.StringVar()
        self.test_file_entry = ttk.Entry(
            files_frame, textvariable=self.test_file_var, width=60
        )
        self.test_file_entry.grid(row=1, column=1, padx=8, pady=4)
        self.test_file_btn = ttk.Button(
            files_frame, text="Browse",
            command=self._browse_test_file
        )
        self.test_file_btn.grid(row=1, column=2, padx=8, pady=4)

        # Test image directory
        ttk.Label(files_frame, text="Test Image Directory").grid(
            row=2, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.test_dir_var = tk.StringVar()
        self.test_dir_entry = ttk.Entry(
            files_frame, textvariable=self.test_dir_var, width=60
        )
        self.test_dir_entry.grid(row=2, column=1, padx=8, pady=4)
        self.test_dir_btn = ttk.Button(
            files_frame, text="Browse",
            command=self._browse_test_dir
        )
        self.test_dir_btn.grid(row=2, column=2, padx=8, pady=4)

        # Test image patterns
        ttk.Label(
            files_frame, text="Test Image Name Patterns (Comma Separated)"
        ).grid(row=3, column=0, sticky=tk.W, padx=8, pady=4)
        self.test_pattern_var = tk.StringVar(value="*.pgm")
        ttk.Entry(
            files_frame, textvariable=self.test_pattern_var, width=60
        ).grid(row=3, column=1, padx=8, pady=4)

        # Prediction output directory
        ttk.Label(files_frame, text="Prediction Output Directory").grid(
            row=4, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.predict_outdir_var = tk.StringVar()
        ttk.Entry(
            files_frame, textvariable=self.predict_outdir_var, width=60
        ).grid(row=4, column=1, padx=8, pady=4)
        ttk.Button(
            files_frame, text="Browse",
            command=self._browse_predict_outdir
        ).grid(row=4, column=2, padx=8, pady=4)

        # ── Prediction Options group ──
        opts_frame = ttk.LabelFrame(self, text="Prediction Options")
        opts_frame.pack(fill=tk.X, padx=8, pady=4)

        # Prediction effort
        ttk.Label(opts_frame, text="Prediction Effort").grid(
            row=0, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.effort_var = tk.StringVar(value="Low")
        ttk.Combobox(
            opts_frame, textvariable=self.effort_var,
            values=["Low", "High"], state="readonly", width=20
        ).grid(row=0, column=1, padx=8, pady=4)

        # Display predictions
        self.display_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts_frame, text="Display Predictions",
            variable=self.display_var,
            command=self._on_display_changed
        ).grid(row=1, column=0, sticky=tk.W, padx=8, pady=4)

        ttk.Label(opts_frame, text="Delay (ms)").grid(
            row=1, column=1, sticky=tk.E, padx=8, pady=4
        )
        self.delay_var = tk.StringVar(value="1000")
        self.delay_entry = ttk.Entry(
            opts_frame, textvariable=self.delay_var, width=12
        )
        self.delay_entry.grid(row=1, column=2, padx=8, pady=4)

        # Reference prediction
        ttk.Label(opts_frame, text="Reference prediction").grid(
            row=2, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.ref_type_var = tk.StringVar(
            value="Compare with Default HOG People Detector"
        )
        self.ref_type_combo = ttk.Combobox(
            opts_frame, textvariable=self.ref_type_var,
            values=[
                "Complete Image Pedestrian",
                "Image Contains no Pedestrian",
                "Compare with Default HOG People Detector",
                "Read from File"
            ],
            state="readonly", width=40
        )
        self.ref_type_combo.grid(row=2, column=1, columnspan=2, padx=8, pady=4)
        self.ref_type_combo.bind("<<ComboboxSelected>>",
                                  self._on_ref_type_changed)

        # Reference file
        ttk.Label(opts_frame, text="Reference prediction file").grid(
            row=3, column=0, sticky=tk.W, padx=8, pady=4
        )
        self.ref_file_var = tk.StringVar()
        self.ref_file_entry = ttk.Entry(
            opts_frame, textvariable=self.ref_file_var, width=60
        )
        self.ref_file_entry.grid(row=3, column=1, padx=8, pady=4)
        self.ref_file_btn = ttk.Button(
            opts_frame, text="Browse",
            command=self._browse_ref_file
        )
        self.ref_file_btn.grid(row=3, column=2, padx=8, pady=4)

        # Detect button
        self.detect_btn = ttk.Button(
            opts_frame, text="Detect Pedestrians",
            command=self._on_detect_click
        )
        self.detect_btn.grid(row=4, column=1, padx=8, pady=8)
        self.detect_btn.configure(state="disabled")

        # Initial states
        self.test_file_entry.configure(state="disabled")
        self.test_file_btn.configure(state="disabled")
        self.ref_file_entry.configure(state="disabled")
        self.ref_file_btn.configure(state="disabled")

    def _init_defaults(self):
        """Set default paths."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base, "data")

        # Use test split T1 by default
        test_dir = os.path.join(data_dir, "T1", "ped_examples")
        if os.path.isdir(test_dir):
            self.test_dir_var.set(test_dir)

        output_dir = os.path.join(base, "output")
        self.predict_outdir_var.set(output_dir)

    # ── Event handlers ──

    def _on_from_file_changed(self):
        if self.from_file_var.get():
            self.test_file_entry.configure(state="normal")
            self.test_file_btn.configure(state="normal")
            self.test_dir_entry.configure(state="disabled")
            self.test_dir_btn.configure(state="disabled")
        else:
            self.test_file_entry.configure(state="disabled")
            self.test_file_btn.configure(state="disabled")
            self.test_dir_entry.configure(state="normal")
            self.test_dir_btn.configure(state="normal")

    def _on_display_changed(self):
        if self.display_var.get():
            self.delay_entry.configure(state="normal")
        else:
            self.delay_entry.configure(state="disabled")

    def _on_ref_type_changed(self, event=None):
        if self.ref_type_var.get() == "Read from File":
            self.ref_file_entry.configure(state="normal")
            self.ref_file_btn.configure(state="normal")
        else:
            self.ref_file_entry.configure(state="disabled")
            self.ref_file_btn.configure(state="disabled")

    def _browse_test_file(self):
        f = filedialog.askopenfilename(
            title="Select Test Image List",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if f:
            self.test_file_var.set(f)

    def _browse_test_dir(self):
        d = filedialog.askdirectory(title="Select Test Image Directory")
        if d:
            self.test_dir_var.set(d)

    def _browse_predict_outdir(self):
        d = filedialog.askdirectory(title="Select Prediction Output Directory")
        if d:
            self.predict_outdir_var.set(d)

    def _browse_ref_file(self):
        f = filedialog.askopenfilename(
            title="Select Reference Prediction File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if f:
            self.ref_file_var.set(f)

    def enable_detect(self, enabled: bool = True):
        """Enable or disable the Detect Pedestrians button."""
        self.detect_btn.configure(
            state="normal" if enabled else "disabled"
        )

    def _on_detect_click(self):
        """Validate inputs and run prediction in a background thread."""
        if self.app.classifier is None:
            messagebox.showerror("Error", "Train the classifier first!")
            return

        opts = self.app.predict_options

        # Read from file or directory
        if self.from_file_var.get():
            opts.read_test_imgs_from_file = True
            opts.test_img_list_file = self.test_file_var.get()
            if not os.path.isfile(opts.test_img_list_file):
                messagebox.showerror(
                    "Error",
                    f"Test image list file not found:\n{opts.test_img_list_file}"
                )
                return
        else:
            opts.read_test_imgs_from_file = False
            opts.test_img_dir = self.test_dir_var.get()
            if not dir_contains_images(opts.test_img_dir):
                messagebox.showerror(
                    "Error",
                    f"Test directory is empty or invalid:\n{opts.test_img_dir}"
                )
                return
            opts.test_img_patterns = self.test_pattern_var.get()

        opts.prediction_effort = 0 if self.effort_var.get() == "Low" else 1
        opts.display_predictions = self.display_var.get()

        if opts.display_predictions:
            try:
                opts.display_delay = int(self.delay_var.get())
                if opts.display_delay < 0:
                    raise ValueError("Must be non-negative")
            except ValueError:
                messagebox.showerror(
                    "Error", "Display delay must be a non-negative integer."
                )
                return

        opts.prediction_output_dir = self.predict_outdir_var.get()
        if not ensure_dir(opts.prediction_output_dir):
            messagebox.showerror(
                "Error",
                f"Could not create output directory:\n{opts.prediction_output_dir}"
            )
            return

        # Reference comparison mode
        ref_mode = self.ref_type_var.get()
        opts.complete_img_ped = False
        opts.img_has_no_ped = False
        opts.compare_with_default_hog = False
        opts.read_from_reference = False

        if ref_mode == "Complete Image Pedestrian":
            opts.complete_img_ped = True
        elif ref_mode == "Image Contains no Pedestrian":
            opts.img_has_no_ped = True
        elif ref_mode == "Compare with Default HOG People Detector":
            opts.compare_with_default_hog = True
        elif ref_mode == "Read from File":
            opts.read_from_reference = True
            opts.ref_file = self.ref_file_var.get()
            if not os.path.isfile(opts.ref_file):
                messagebox.showerror(
                    "Error",
                    f"Reference file not found:\n{opts.ref_file}"
                )
                return

        # Disable UI and run
        self.app.status_bar.reset()
        self.app.set_running("Running prediction ...")
        self.detect_btn.configure(state="disabled")

        def predict_thread():
            try:
                predictor = PedestrianPredictor(
                    self.app.classifier,
                    self.app.train_options,
                    opts,
                    log_callback=self._thread_safe_log,
                    progress_callback=self._thread_safe_progress
                )
                predictor.run()

                evaluator = Evaluator(
                    predictor, self.app.train_options, opts
                )
                evaluator.write_all()

                # Capture results for dialog
                img_acc = predictor.img_summary.accuracy()
                rect_acc = predictor.rect_summary.accuracy()
                img_s = predictor.img_summary
                n_tested = len(predictor.all_images)
                n_detected = len(predictor.ped_images)
                pred_dir = os.path.join(
                    opts.prediction_output_dir,
                    opts.prediction_subdir
                )

                self.app.after(0, lambda: self._predict_done(
                    True, img_acc=img_acc, rect_acc=rect_acc,
                    n_tested=n_tested, n_detected=n_detected,
                    tp=img_s.true_pos, tn=img_s.true_neg,
                    fp=img_s.false_pos, fn=img_s.false_neg,
                    pred_dir=pred_dir
                ))
            except Exception as e:
                err_msg = str(e)
                self.app.after(0, lambda: self._predict_done(
                    False, error_msg=err_msg
                ))

        t = threading.Thread(target=predict_thread, daemon=True)
        t.start()

    def _thread_safe_log(self, msg: str):
        self.app.after(0, lambda: self.app.output_panel.log(msg))

    def _thread_safe_progress(self, frac: float):
        self.app.after(0, lambda: self.app.status_bar.set_progress(frac))

    def _predict_done(self, success: bool, error_msg: str = "",
                      img_acc: float = 0, rect_acc: float = 0,
                      n_tested: int = 0, n_detected: int = 0,
                      tp: int = 0, tn: int = 0, fp: int = 0, fn: int = 0,
                      pred_dir: str = ""):
        self.detect_btn.configure(state="normal")
        self.app.status_bar.reset()
        self.app.set_ready("Ready for prediction", enable_predict=True)

        import cv2
        cv2.destroyAllWindows()

        if success:
            def _view_gallery():
                ImageGalleryDialog(self.app, pred_dir,
                                    title="Detected Pedestrians")

            ResultDialog(
                self.app,
                title="Detection Complete",
                success=True,
                status_text="Pedestrian Detection Complete",
                subtitle=f"Processed {n_tested} images in total.",
                metrics={
                    "Images Tested": str(n_tested),
                    "Pedestrians Detected": f"{n_detected} / {n_tested}",
                    "Image Accuracy": f"{img_acc:.2%}",
                    "Rectangle Accuracy": f"{rect_acc:.2%}",
                    "True Positives": str(tp),
                    "True Negatives": str(tn),
                    "False Positives": str(fp),
                    "False Negatives": str(fn),
                },
                detail=f"Results saved to: {pred_dir}",
                view_callback=_view_gallery
            )
        else:
            ResultDialog(
                self.app,
                title="Detection Failed",
                success=False,
                status_text="Detection Failed",
                subtitle="An error occurred during prediction.",
                error=error_msg
            )
