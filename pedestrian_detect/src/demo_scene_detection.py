#!/usr/bin/env python3
"""Create a composite scene image from dataset patches and run detection.

This demonstrates the pedestrian detector on a larger image where
bounding boxes are drawn around detected pedestrian regions.
"""

import os
import cv2
import numpy as np
import random

from core.config import TrainOptions, PredictOptions, IMG_WIDTH, IMG_HEIGHT
from core.trainer import PedestrianTrainer
from core.predictor import PedestrianPredictor
from utils.file_utils import list_images, ensure_dir, create_or_clean_dir


def create_composite_scene(ped_dir: str, non_ped_dir: str,
                           output_path: str,
                           grid_cols: int = 20, grid_rows: int = 10) -> str:
    """Create a composite scene by arranging ped/non-ped patches in a grid.

    Pedestrian patches are placed randomly among non-pedestrian patches.
    """
    ped_files = list_images(ped_dir, "*.pgm")
    non_ped_files = list_images(non_ped_dir, "*.pgm")

    total_cells = grid_cols * grid_rows

    # Place ~30% pedestrians randomly
    n_peds = int(total_cells * 0.3)
    ped_positions = set(random.sample(range(total_cells), n_peds))

    # Select random images
    random.seed(42)
    ped_sample = random.choices(ped_files, k=n_peds)
    non_ped_sample = random.choices(non_ped_files, k=total_cells - n_peds)

    # Create canvas (with 2px border between patches for clarity)
    border = 2
    cell_w = IMG_WIDTH + border
    cell_h = IMG_HEIGHT + border
    canvas_w = grid_cols * cell_w + border
    canvas_h = grid_rows * cell_h + border

    # Dark gray background
    canvas = np.full((canvas_h, canvas_w, 3), 80, dtype=np.uint8)

    ped_idx = 0
    non_ped_idx = 0
    ground_truth = []  # Track where pedestrians are placed

    for cell_idx in range(total_cells):
        row = cell_idx // grid_cols
        col = cell_idx % grid_cols
        y = border + row * cell_h
        x = border + col * cell_w

        if cell_idx in ped_positions:
            img = cv2.imread(ped_sample[ped_idx])
            ped_idx += 1
            ground_truth.append((x, y, IMG_WIDTH, IMG_HEIGHT))
        else:
            img = cv2.imread(non_ped_sample[non_ped_idx])
            non_ped_idx += 1

        if img is not None:
            # Convert grayscale to BGR if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            canvas[y:y + IMG_HEIGHT, x:x + IMG_WIDTH] = img

    cv2.imwrite(output_path, canvas)
    print(f"Created composite scene: {canvas_w}x{canvas_h} pixels")
    print(f"  Grid: {grid_cols}x{grid_rows} = {total_cells} cells")
    print(f"  Pedestrian patches: {n_peds}")
    print(f"  Non-pedestrian patches: {total_cells - n_peds}")

    return output_path


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "data")
    out_dir = os.path.join(base, "demo_output")

    ensure_dir(out_dir)

    # Create composite scene
    print("=" * 60)
    print("Step 1: Creating composite scene image")
    print("=" * 60)
    scene_path = os.path.join(out_dir, "composite_scene.png")
    create_composite_scene(
        os.path.join(data_dir, "1", "ped_examples"),
        os.path.join(data_dir, "1", "non-ped_examples"),
        scene_path,
        grid_cols=20, grid_rows=10
    )

    # Train classifier
    print()
    print("=" * 60)
    print("Step 2: Training classifier")
    print("=" * 60)
    train_opts = TrainOptions()
    train_opts.ped_train_data_path = os.path.join(data_dir, "1", "ped_examples")
    train_opts.non_ped_train_data_path = os.path.join(
        data_dir, "1", "non-ped_examples"
    )
    train_opts.ped_train_data_size = 200
    train_opts.non_ped_train_data_size = 200
    train_opts.scaled_image = True
    train_opts.feature_vec_type = 1  # HOG
    train_opts.hog_block_size = 12
    train_opts.hog_cell_size = 6
    train_opts.training_output_dir = out_dir

    ensure_dir(os.path.join(out_dir, train_opts.ped_train_dump))
    ensure_dir(os.path.join(out_dir, train_opts.non_ped_train_dump))

    trainer = PedestrianTrainer(train_opts)
    classifier = trainer.train()
    print(f"Trained: {trainer.actual_ped_samples} ped + "
          f"{trainer.actual_non_ped_samples} non-ped samples")

    # Run detection on the composite scene
    print()
    print("=" * 60)
    print("Step 3: Running detection on composite scene")
    print("=" * 60)

    pred_opts = PredictOptions()
    pred_opts.test_img_dir = out_dir
    pred_opts.test_img_patterns = "composite_scene.png"
    pred_opts.prediction_output_dir = out_dir
    pred_opts.display_predictions = False
    pred_opts.complete_img_ped = False
    pred_opts.img_has_no_ped = False
    pred_opts.compare_with_default_hog = False

    create_or_clean_dir(os.path.join(out_dir, pred_opts.prediction_subdir))

    predictor = PedestrianPredictor(classifier, train_opts, pred_opts)
    predictor.run()

    n_detected = len(predictor.ped_images)
    print(f"Detection complete: {n_detected} image(s) with pedestrians found")

    # Also create a high-resolution annotated version
    print()
    print("=" * 60)
    print("Step 4: Creating annotated result image")
    print("=" * 60)

    result_path = os.path.join(out_dir, "Predictions", "composite_scene.ppm")
    if os.path.isfile(result_path):
        print(f"Result with bounding boxes saved to: {result_path}")

        # Scale up the result for better visibility
        result_img = cv2.imread(result_path)
        if result_img is not None:
            scale = 3
            h, w = result_img.shape[:2]
            large = cv2.resize(result_img, (w * scale, h * scale),
                               interpolation=cv2.INTER_NEAREST)
            large_path = os.path.join(out_dir, "detection_result_3x.png")
            cv2.imwrite(large_path, large)
            print(f"Scaled (3x) result saved to: {large_path}")
            print(f"  Size: {w * scale}x{h * scale} pixels")
    else:
        print("No detection result image found (no pedestrians detected in scene)")
        # Still save the original scene with no detections
        print(f"Original scene: {scene_path}")

    print()
    print("Open demo_output/ folder to see the results:")
    print(f"  1. composite_scene.png - Original scene (ped+non-ped patches)")
    print(f"  2. Predictions/composite_scene.ppm - Scene with detection boxes")
    print(f"  3. detection_result_3x.png - 3x scaled version for visibility")


if __name__ == "__main__":
    main()
