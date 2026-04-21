"""Professional result dialogs and image gallery viewer."""

import os
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk


class ResultDialog(tk.Toplevel):
    """Professional-styled result dialog with metrics display."""

    def __init__(self, parent, title: str, success: bool, **kwargs):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.update_idletasks()
        self.geometry("+%d+%d" % (
            parent.winfo_rootx() + 100,
            parent.winfo_rooty() + 80
        ))

        self._success = success
        self._build_ui(**kwargs)

        # Focus and bind Enter/Escape
        self.bind("<Return>", lambda e: self.destroy())
        self.bind("<Escape>", lambda e: self.destroy())
        self.focus_set()

    def _build_ui(self, **kwargs):
        # Container with padding
        container = ttk.Frame(self, padding=20)
        container.pack(fill=tk.BOTH, expand=True)

        # Icon + Title row
        icon_frame = ttk.Frame(container)
        icon_frame.pack(fill=tk.X, pady=(0, 15))

        icon_char = "\u2705" if self._success else "\u274C"
        icon_color = "#2E7D32" if self._success else "#C62828"
        status_text = kwargs.get("status_text", "Complete" if self._success else "Failed")

        icon_label = tk.Label(icon_frame, text=icon_char, font=("Helvetica", 28))
        icon_label.pack(side=tk.LEFT, padx=(0, 12))

        title_frame = ttk.Frame(icon_frame)
        title_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(title_frame, text=status_text,
                  font=("Helvetica", 16, "bold")).pack(anchor=tk.W)

        subtitle = kwargs.get("subtitle", "")
        if subtitle:
            ttk.Label(title_frame, text=subtitle,
                      font=("Helvetica", 10)).pack(anchor=tk.W)

        # Separator
        ttk.Separator(container, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # Metrics section
        metrics = kwargs.get("metrics", {})
        if metrics:
            metrics_frame = ttk.LabelFrame(container, text="Results", padding=10)
            metrics_frame.pack(fill=tk.X, pady=(5, 10))

            for i, (label, value) in enumerate(metrics.items()):
                ttk.Label(metrics_frame, text=label,
                          font=("Helvetica", 11)).grid(
                    row=i, column=0, sticky=tk.W, padx=(5, 20), pady=3
                )
                val_label = ttk.Label(metrics_frame, text=str(value),
                                       font=("Helvetica", 11, "bold"))
                val_label.grid(row=i, column=1, sticky=tk.E, padx=5, pady=3)

            metrics_frame.columnconfigure(1, weight=1)

        # Detail text
        detail = kwargs.get("detail", "")
        if detail:
            ttk.Label(container, text=detail, wraplength=380,
                      font=("Helvetica", 10)).pack(anchor=tk.W, pady=(5, 10))

        # Error message for failures
        error = kwargs.get("error", "")
        if error:
            err_frame = ttk.LabelFrame(container, text="Error Details", padding=8)
            err_frame.pack(fill=tk.X, pady=(5, 10))
            err_text = tk.Text(err_frame, height=4, wrap=tk.WORD,
                                font=("Courier", 9), bg="#FFF3F3")
            err_text.insert(tk.END, error)
            err_text.configure(state="disabled")
            err_text.pack(fill=tk.X)

        # Buttons
        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        # View Results button (for prediction success)
        self._view_callback = kwargs.get("view_callback")
        if self._view_callback and self._success:
            ttk.Button(btn_frame, text="View Detections",
                       command=self._view_and_close).pack(
                side=tk.LEFT, padx=(0, 8)
            )

        ttk.Button(btn_frame, text="OK", command=self.destroy,
                   style="Accent.TButton").pack(side=tk.RIGHT)

        # Set minimum size
        self.minsize(420, 200)

    def _view_and_close(self):
        """Close dialog then open the gallery after a brief delay."""
        callback = self._view_callback
        parent = self.master
        self.destroy()
        # Schedule gallery opening after dialog fully closes
        if parent and callback:
            parent.after(100, callback)


class ImageGalleryDialog(tk.Toplevel):
    """Gallery viewer for prediction result images."""

    THUMB_SIZE = (180, 360)
    COLS = 5

    def __init__(self, parent, image_dir: str, title: str = "Detection Results"):
        super().__init__(parent)
        self.title(title)
        self.geometry("980x650")
        self.transient(parent)

        self._image_dir = image_dir
        self._photo_cache = []  # Keep references to prevent GC
        self._current_page = 0
        self._images_per_page = 20
        self._image_files = []

        self._load_image_list()
        self._build_ui()
        self._show_page(0)

        self.bind("<Escape>", lambda e: self.destroy())

    def _load_image_list(self):
        """Scan for prediction output images."""
        if not os.path.isdir(self._image_dir):
            return
        exts = {".ppm", ".png", ".jpg", ".jpeg", ".bmp", ".pgm"}
        files = []
        for f in sorted(os.listdir(self._image_dir)):
            if os.path.splitext(f)[1].lower() in exts:
                # Skip _ref_compare images in the main view
                if "_ref_compare" not in f:
                    files.append(os.path.join(self._image_dir, f))
        self._image_files = files

    def _build_ui(self):
        # Top info bar
        info_frame = ttk.Frame(self, padding=(10, 5))
        info_frame.pack(fill=tk.X)

        self._info_label = ttk.Label(
            info_frame,
            text=f"Detected pedestrians: {len(self._image_files)} images",
            font=("Helvetica", 11, "bold")
        )
        self._info_label.pack(side=tk.LEFT)

        # Open folder button
        ttk.Button(info_frame, text="Open Folder",
                    command=self._open_folder).pack(side=tk.RIGHT, padx=5)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # Scrollable canvas for thumbnails
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(canvas_frame, bg="#F5F5F5")
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL,
                                   command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner_frame = ttk.Frame(self._canvas)
        self._canvas.create_window((0, 0), window=self._inner_frame,
                                    anchor=tk.NW)

        self._inner_frame.bind("<Configure>",
                                lambda e: self._canvas.configure(
                                    scrollregion=self._canvas.bbox("all")
                                ))

        # Mouse wheel scrolling
        self._canvas.bind_all("<MouseWheel>",
                               lambda e: self._canvas.yview_scroll(
                                   int(-1 * (e.delta / 120)), "units"
                               ))

        # Navigation bar
        nav_frame = ttk.Frame(self, padding=5)
        nav_frame.pack(fill=tk.X)

        self._prev_btn = ttk.Button(nav_frame, text="< Prev",
                                     command=self._prev_page)
        self._prev_btn.pack(side=tk.LEFT, padx=5)

        self._page_label = ttk.Label(nav_frame, text="",
                                      font=("Helvetica", 10))
        self._page_label.pack(side=tk.LEFT, expand=True)

        self._next_btn = ttk.Button(nav_frame, text="Next >",
                                     command=self._next_page)
        self._next_btn.pack(side=tk.RIGHT, padx=5)

    def _show_page(self, page: int):
        """Display a page of thumbnails."""
        # Clear current
        for w in self._inner_frame.winfo_children():
            w.destroy()
        self._photo_cache.clear()

        total = len(self._image_files)
        if total == 0:
            ttk.Label(self._inner_frame,
                      text="No detection images found.",
                      font=("Helvetica", 12)).pack(padx=20, pady=40)
            self._page_label.configure(text="0 / 0")
            return

        start = page * self._images_per_page
        end = min(start + self._images_per_page, total)
        total_pages = (total + self._images_per_page - 1) // self._images_per_page

        self._current_page = page
        self._page_label.configure(
            text=f"Page {page + 1} / {total_pages}  "
                 f"(showing {start + 1}-{end} of {total})"
        )
        self._prev_btn.configure(state="normal" if page > 0 else "disabled")
        self._next_btn.configure(
            state="normal" if end < total else "disabled"
        )

        page_files = self._image_files[start:end]

        for idx, filepath in enumerate(page_files):
            row = idx // self.COLS
            col = idx % self.COLS

            cell = ttk.Frame(self._inner_frame, padding=4)
            cell.grid(row=row, column=col, padx=4, pady=4)

            try:
                img = cv2.imread(filepath)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)

                    # Scale to fit thumbnail while keeping aspect ratio
                    tw, th = self.THUMB_SIZE
                    pil_img.thumbnail((tw, th), Image.LANCZOS)

                    photo = ImageTk.PhotoImage(pil_img)
                    self._photo_cache.append(photo)

                    label = tk.Label(cell, image=photo, borderwidth=2,
                                      relief=tk.GROOVE, bg="#FFFFFF")
                    label.pack()
                else:
                    ttk.Label(cell, text="[Error]").pack()
            except Exception:
                ttk.Label(cell, text="[Error]").pack()

            fname = os.path.basename(filepath)
            ttk.Label(cell, text=fname, font=("Helvetica", 8),
                      wraplength=self.THUMB_SIZE[0]).pack()

        self._canvas.yview_moveto(0)

    def _prev_page(self):
        if self._current_page > 0:
            self._show_page(self._current_page - 1)

    def _next_page(self):
        total_pages = (len(self._image_files) + self._images_per_page - 1) // self._images_per_page
        if self._current_page < total_pages - 1:
            self._show_page(self._current_page + 1)

    def _open_folder(self):
        """Open the predictions folder in the OS file manager."""
        import subprocess
        import sys
        if sys.platform == "darwin":
            subprocess.Popen(["open", self._image_dir])
        elif sys.platform == "win32":
            os.startfile(self._image_dir)
        else:
            subprocess.Popen(["xdg-open", self._image_dir])
