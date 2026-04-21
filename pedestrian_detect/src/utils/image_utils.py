"""Image I/O, resizing, and cutting utility functions."""

import math
import os
import numpy as np
import cv2

from core.config import IMG_WIDTH, IMG_HEIGHT


def read_image(filepath: str) -> np.ndarray:
    """Read an image from disk using OpenCV.

    Args:
        filepath: Path to the image file.

    Returns:
        Image as numpy array (BGR or greyscale depending on file).
    """
    img = cv2.imread(filepath)
    if img is None:
        raise IOError(f"Could not read image: {filepath}")
    return img


def determine_cut_rects(img: np.ndarray, cuts: int,
                        cut_size_factor: int) -> list:
    """Determine random cut rectangles from a non-pedestrian image.

    Matches the C# DetermineCutRects logic: generates random positions
    for cuts of size (IMG_WIDTH * factor, IMG_HEIGHT * factor).

    Args:
        img: Source image.
        cuts: Number of cuts to make.
        cut_size_factor: Size multiplier relative to base image.

    Returns:
        List of (x, y, w, h) tuples.
    """
    cut_height = IMG_HEIGHT * cut_size_factor
    cut_width = IMG_WIDTH * cut_size_factor

    img_height, img_width = img.shape[:2]

    if cut_height > img_height or cut_width > img_width:
        # Fall back to scaling when image is too small for cuts
        scaled = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        rects = [(0, 0, img_width, img_height)]
        return rects

    end_x = img_width - cut_width
    end_y = img_height - cut_height

    rects = []
    for _ in range(cuts):
        x = int(np.random.uniform(0, end_x)) if end_x > 0 else 0
        y = int(np.random.uniform(0, end_y)) if end_y > 0 else 0
        rects.append((x, y, cut_width, cut_height))

    return rects


def get_non_ped_feature_vectors(filepath: str,
                                 scaled_image: bool,
                                 cuts: int,
                                 cut_size_factor: int,
                                 extract_fn,
                                 dump_file: str = None) -> list:
    """Extract feature vectors from a non-pedestrian image.

    Either scales the entire image or cuts random patches, then extracts
    feature vectors from each.

    Args:
        filepath: Path to the non-pedestrian image.
        scaled_image: If True, scale entire image.
        cuts: Number of random cuts.
        cut_size_factor: Size multiplier for cuts.
        extract_fn: Feature extraction function (img -> np.ndarray).
        dump_file: Optional path to save a debug image.

    Returns:
        List of 1-D float32 feature vectors.
    """
    img = read_image(filepath)
    height, width = img.shape[:2]

    if height < IMG_HEIGHT or width < IMG_WIDTH:
        raise ValueError(
            f"Non-pedestrian image {filepath} is smaller than "
            f"minimum ({IMG_WIDTH}x{IMG_HEIGHT})"
        )

    fv_list = []

    if scaled_image:
        # Scale entire image to 18x36
        img_ratio = height / width
        expected_ratio = IMG_HEIGHT / IMG_WIDTH
        if abs(img_ratio - expected_ratio) > 0.01:
            raise ValueError(
                f"Non-pedestrian image {filepath} doesn't have "
                f"2:1 height-to-width ratio for scaling"
            )

        if dump_file:
            cv2.imwrite(dump_file, img)

        scaled = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        fv = extract_fn(scaled)
        fv_list.append(fv)
        return fv_list

    # Cut random patches
    cut_rects = determine_cut_rects(img, cuts, cut_size_factor)

    for (x, y, w, h) in cut_rects:
        cut = img[y:y + h, x:x + w]
        scaled_cut = cv2.resize(cut, (IMG_WIDTH, IMG_HEIGHT))
        fv = extract_fn(scaled_cut)
        fv_list.append(fv)

    if dump_file:
        debug_img = img.copy()
        for (x, y, w, h) in cut_rects:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h),
                          (0, 0, 255), 2)
        cv2.imwrite(dump_file, debug_img)

    return fv_list
