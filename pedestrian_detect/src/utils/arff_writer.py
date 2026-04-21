"""Write training data to ARFF format (Weka compatibility)."""

import os
import numpy as np

from core.config import IMG_WIDTH, IMG_HEIGHT, PED_CLASS


def write_arff(filepath: str,
               training_data: np.ndarray,
               training_labels: np.ndarray,
               feature_vec_type: int,
               feature_vector_size: int) -> None:
    """Write training data to an ARFF file.

    Args:
        filepath: Output ARFF file path.
        training_data: (N, D) float32 matrix  of feature vectors.
        training_labels: (N,) int array of class labels.
        feature_vec_type: 0 = greyscale, 1 = HOG.
        feature_vector_size: Length of feature vectors.
    """
    with open(filepath, "w") as f:
        # Header
        f.write("@RELATION Pedestrians\n")

        if feature_vec_type == 0:
            for row_idx in range(IMG_HEIGHT):
                for col_idx in range(IMG_WIDTH):
                    f.write(f"@ATTRIBUTE Pixel_{row_idx:02d}_{col_idx:02d} NUMERIC\n")
        else:
            for elem_idx in range(feature_vector_size):
                f.write(f"@ATTRIBUTE HOGBin_{elem_idx:03d} NUMERIC\n")

        f.write("@ATTRIBUTE Class {Ped, NonPed}\n")

        # Data
        f.write("@DATA\n")
        n_rows, n_cols = training_data.shape

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                val = training_data[row_idx, col_idx]
                if feature_vec_type == 0:
                    f.write(f"{int(val)},")
                else:
                    f.write(f"{val},")

            label = training_labels[row_idx]
            if label == PED_CLASS:
                f.write("Ped\n")
            else:
                f.write("NonPed\n")
