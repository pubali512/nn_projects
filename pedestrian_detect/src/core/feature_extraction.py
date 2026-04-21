"""Feature extraction: Greyscale pixel values and HOG descriptors."""

import numpy as np
import cv2

from core.config import IMG_WIDTH, IMG_HEIGHT


def compute_feature_vector_size(feature_vec_type: int,
                                 hog_block_size: int = 12) -> int:
    """Compute the feature vector length for the given configuration.

    Args:
        feature_vec_type: 0 for greyscale, 1 for HOG.
        hog_block_size: HOG block size (12 or 18).

    Returns:
        Length of the feature vector.
    """
    if feature_vec_type == 0:
        # Greyscale: one value per pixel
        return IMG_HEIGHT * IMG_WIDTH  # 648
    else:
        # HOG: compute number of blocks dynamically
        cell_size = hog_block_size // 2
        cells_per_block = hog_block_size // cell_size  # typically 2
        blocks_x = (IMG_WIDTH - hog_block_size) // cell_size + 1
        blocks_y = (IMG_HEIGHT - hog_block_size) // cell_size + 1
        nr_of_blocks = blocks_x * blocks_y
        cells_per_block_total = cells_per_block * cells_per_block  # 4
        return nr_of_blocks * cells_per_block_total * 9


def extract_greyscale_features(img: np.ndarray) -> np.ndarray:
    """Flatten an image to greyscale pixel intensity vector.

    Args:
        img: BGR or greyscale image (will be resized to 18x36).

    Returns:
        1-D float32 array of length IMG_WIDTH * IMG_HEIGHT.
    """
    if len(img.shape) == 3:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey = img

    resized = cv2.resize(grey, (IMG_WIDTH, IMG_HEIGHT))
    return resized.flatten().astype(np.float32)


def extract_hog_features(img: np.ndarray,
                         block_size: int = 12,
                         cell_size: int = 6) -> np.ndarray:
    """Extract HOG descriptor from an image.

    Args:
        img: BGR or greyscale image (will be resized to 18x36).
        block_size: HOG block size in pixels.
        cell_size: HOG cell size in pixels.

    Returns:
        1-D float32 array of HOG features.
    """
    if len(img.shape) == 3:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey = img

    resized = cv2.resize(grey, (IMG_WIDTH, IMG_HEIGHT))

    hog = cv2.HOGDescriptor(
        _winSize=(IMG_WIDTH, IMG_HEIGHT),
        _blockSize=(block_size, block_size),
        _blockStride=(cell_size, cell_size),
        _cellSize=(cell_size, cell_size),
        _nbins=9
    )

    features = hog.compute(resized)
    return features.flatten().astype(np.float32)


def extract_feature_vector(img: np.ndarray,
                           feature_vec_type: int = 1,
                           block_size: int = 12,
                           cell_size: int = 6) -> np.ndarray:
    """Unified feature extraction entry point.

    Args:
        img: Input image (BGR or greyscale).
        feature_vec_type: 0 = greyscale, 1 = HOG.
        block_size: HOG block size.
        cell_size: HOG cell size.

    Returns:
        1-D float32 feature vector.
    """
    if feature_vec_type == 0:
        return extract_greyscale_features(img)
    else:
        return extract_hog_features(img, block_size, cell_size)
