"""Training pipeline: read images, extract features, train Naive Bayes classifier."""

import os
import numpy as np
from sklearn.naive_bayes import GaussianNB

from core.config import (
    TrainOptions, IMG_WIDTH, IMG_HEIGHT, PED_CLASS, NON_PED_CLASS
)
from core.feature_extraction import (
    extract_feature_vector, compute_feature_vector_size
)
from utils.image_utils import read_image, get_non_ped_feature_vectors
from utils.file_utils import list_images, ensure_dir, create_or_clean_dir
from utils.arff_writer import write_arff


class PedestrianTrainer:
    """Reads training images, extracts features, trains a GaussianNB classifier.

    This is a faithful port of the C# TrainClassifier / ReadTrainingImages /
    ExtractFeatureVector / WriteArffFile logic.
    """

    def __init__(self, train_options: TrainOptions,
                 log_callback=None, progress_callback=None):
        """
        Args:
            train_options: Training configuration.
            log_callback: callable(msg: str) for logging.
            progress_callback: callable(fraction: float) for progress bar.
        """
        self.opts = train_options
        self.classifier = GaussianNB()

        self.training_data: np.ndarray = None       # (N, D) float32
        self.training_labels: np.ndarray = None      # (N,) int

        self.actual_ped_samples: int = 0
        self.actual_non_ped_samples: int = 0

        self._log = log_callback or (lambda msg: None)
        self._progress = progress_callback or (lambda frac: None)

    def _determine_feature_vector_length(self):
        """Compute feature vector size based on options."""
        self.opts.feature_vector_size = compute_feature_vector_size(
            self.opts.feature_vec_type, self.opts.hog_block_size
        )

    def _determine_non_ped_train_data_length(self):
        """Compute total non-pedestrian samples (images * samples_per_image)."""
        if self.opts.scaled_image:
            samples_per_img = 1
        else:
            samples_per_img = self.opts.cuts
        self.opts.total_non_ped_train_data_size = (
            self.opts.non_ped_train_data_size * samples_per_img
        )

    def _extract_fn(self, img: np.ndarray) -> np.ndarray:
        """Shorthand to call extract_feature_vector with current options."""
        return extract_feature_vector(
            img,
            self.opts.feature_vec_type,
            self.opts.hog_block_size,
            self.opts.hog_cell_size
        )

    def _read_training_images(self):
        """Read pedestrian and non-pedestrian images, fill training matrices."""
        self.actual_ped_samples = 0
        self.actual_non_ped_samples = 0

        ped_dump_dir = os.path.join(
            self.opts.training_output_dir, self.opts.ped_train_dump
        )
        non_ped_dump_dir = os.path.join(
            self.opts.training_output_dir, self.opts.non_ped_train_dump
        )

        ped_files = list_images(self.opts.ped_train_data_path, "*.pgm")
        non_ped_files = list_images(self.opts.non_ped_train_data_path, "*.pgm")

        train_data_idx = 0

        # Read pedestrian images
        for f in ped_files:
            self._log(f"Pedestrian sample => {f}")

            img = read_image(f)
            h, w = img.shape[:2]

            if h != IMG_HEIGHT or w != IMG_WIDTH:
                raise ValueError(
                    f"Pedestrian image {f} has dimensions {w}x{h}, "
                    f"expected {IMG_WIDTH}x{IMG_HEIGHT}"
                )

            fv = self._extract_fn(img)
            self.actual_ped_samples += 1

            # Fill training data
            self.training_labels[train_data_idx] = PED_CLASS
            self.training_data[train_data_idx, :] = fv

            # Dump debug image
            fname = os.path.splitext(os.path.basename(f))[0]
            dump_file = os.path.join(
                ped_dump_dir, f"{fname}_{train_data_idx:05d}.ppm"
            )
            cv2_import_write(dump_file, img)

            train_data_idx += 1
            if train_data_idx >= self.opts.ped_train_data_size:
                break

        # Read non-pedestrian images
        non_ped_sample_idx = 0
        non_ped_img_idx = 0

        for f in non_ped_files:
            self._log(f"Non pedestrian sample => {f}")

            fname = os.path.splitext(os.path.basename(f))[0]
            dump_file = os.path.join(
                non_ped_dump_dir, f"{fname}_{non_ped_img_idx:05d}.ppm"
            )
            non_ped_img_idx += 1

            fv_list = get_non_ped_feature_vectors(
                f,
                self.opts.scaled_image,
                self.opts.cuts,
                self.opts.cut_size_factor,
                self._extract_fn,
                dump_file
            )

            for fv in fv_list:
                self.training_labels[train_data_idx] = NON_PED_CLASS
                self.training_data[train_data_idx, :] = fv
                train_data_idx += 1
                non_ped_sample_idx += 1
                self.actual_non_ped_samples += 1

            if non_ped_sample_idx >= self.opts.total_non_ped_train_data_size:
                break

    def _write_arff(self):
        """Write training data to ARFF file."""
        arff_path = os.path.join(
            self.opts.training_output_dir, self.opts.arff_file
        )
        total_samples = self.actual_ped_samples + self.actual_non_ped_samples
        write_arff(
            arff_path,
            self.training_data[:total_samples],
            self.training_labels[:total_samples],
            self.opts.feature_vec_type,
            self.opts.feature_vector_size
        )

    def _do_training(self):
        """Fit the GaussianNB classifier."""
        total_samples = self.actual_ped_samples + self.actual_non_ped_samples
        X = self.training_data[:total_samples]
        y = self.training_labels[:total_samples]
        self.classifier.fit(X, y)

    def train(self) -> GaussianNB:
        """Full training pipeline.

        Returns:
            Trained GaussianNB classifier.

        Raises:
            Various exceptions on invalid data or I/O errors.
        """
        self._determine_feature_vector_length()
        self._determine_non_ped_train_data_length()

        total_train = (
            self.opts.ped_train_data_size +
            self.opts.total_non_ped_train_data_size
        )

        self.training_data = np.zeros(
            (total_train, self.opts.feature_vector_size), dtype=np.float32
        )
        self.training_labels = np.zeros(total_train, dtype=np.int32)

        # Prepare output directories
        ped_dump_dir = os.path.join(
            self.opts.training_output_dir, self.opts.ped_train_dump
        )
        non_ped_dump_dir = os.path.join(
            self.opts.training_output_dir, self.opts.non_ped_train_dump
        )
        create_or_clean_dir(ped_dump_dir)
        create_or_clean_dir(non_ped_dump_dir)

        self._log("Reading training images")
        self._progress(0.1)
        self._read_training_images()
        self._progress(0.4)

        # Validate that we have actual training data
        total_actual = self.actual_ped_samples + self.actual_non_ped_samples
        if total_actual == 0:
            raise ValueError(
                "No training images were loaded. Check your data directories."
            )
        if self.actual_ped_samples < self.opts.ped_train_data_size:
            self._log(
                f"Warning: Requested {self.opts.ped_train_data_size} ped "
                f"samples but only found {self.actual_ped_samples}"
            )
        if self.actual_non_ped_samples < self.opts.total_non_ped_train_data_size:
            self._log(
                f"Warning: Requested {self.opts.total_non_ped_train_data_size} "
                f"non-ped samples but only found {self.actual_non_ped_samples}"
            )

        self._log("Writing .arff file")
        self._write_arff()
        self._progress(0.6)

        self._log("Training the Bayesian Classifier")
        self._do_training()
        self._progress(1.0)

        self._log(
            f"Training complete. "
            f"Ped samples: {self.actual_ped_samples}, "
            f"Non-ped samples: {self.actual_non_ped_samples}, "
            f"Feature vector size: {self.opts.feature_vector_size}"
        )

        return self.classifier


def cv2_import_write(path: str, img):
    """Write image using cv2 (delayed import to keep module lightweight)."""
    import cv2
    cv2.imwrite(path, img)
