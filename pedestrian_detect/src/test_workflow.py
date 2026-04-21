#!/usr/bin/env python3
"""End-to-end test: train → predict → evaluate (headless, no GUI)."""

import os
import sys
import time

# Project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_output")

from core.config import TrainOptions, PredictOptions
from core.trainer import PedestrianTrainer
from core.predictor import PedestrianPredictor
from core.evaluator import Evaluator


def log(msg):
    print(f"  [LOG] {msg}")


def progress(frac):
    bar = int(frac * 40)
    print(f"\r  [PROGRESS] [{'#' * bar}{'.' * (40 - bar)}] {frac:.0%}", end="", flush=True)
    if frac >= 1.0:
        print()


def main():
    print("=" * 60)
    print("  End-to-End Workflow Test")
    print("=" * 60)

    # ── 1. Configure Training ──
    print("\n[STEP 1] Configuring training options...")
    train_opts = TrainOptions(
        ped_train_data_size=50,
        non_ped_train_data_size=50,
        ped_train_data_path=os.path.join(DATA_DIR, "1", "ped_examples"),
        non_ped_train_data_path=os.path.join(DATA_DIR, "1", "non-ped_examples"),
        training_output_dir=OUTPUT_DIR,
        feature_vec_type=1,      # HOG
        hog_block_size=12,
        hog_cell_size=6,
        scaled_image=True,       # Non-ped images are 18x36, same as ped
        cuts=1,
        cut_size_factor=1,
    )

    # Verify data directories exist
    assert os.path.isdir(train_opts.ped_train_data_path), \
        f"Ped dir not found: {train_opts.ped_train_data_path}"
    assert os.path.isdir(train_opts.non_ped_train_data_path), \
        f"Non-ped dir not found: {train_opts.non_ped_train_data_path}"
    print(f"  Ped dir:     {train_opts.ped_train_data_path}")
    print(f"  Non-ped dir: {train_opts.non_ped_train_data_path}")
    print(f"  Output dir:  {OUTPUT_DIR}")

    # ── 2. Train ──
    print("\n[STEP 2] Training Naive Bayes classifier...")
    t0 = time.time()

    trainer = PedestrianTrainer(train_opts, log_callback=log, progress_callback=progress)
    classifier = trainer.train()

    t1 = time.time()
    print(f"  Training took {t1 - t0:.2f}s")
    print(f"  Ped samples:     {trainer.actual_ped_samples}")
    print(f"  Non-ped samples: {trainer.actual_non_ped_samples}")
    print(f"  Feature dim:     {train_opts.feature_vector_size}")

    # Verify ARFF was written
    arff_path = os.path.join(OUTPUT_DIR, train_opts.arff_file)
    assert os.path.isfile(arff_path), f"ARFF file not created: {arff_path}"
    print(f"  ARFF file:       {arff_path} ({os.path.getsize(arff_path)} bytes)")

    # ── 3. Configure Prediction ──
    print("\n[STEP 3] Configuring prediction options...")
    predict_opts = PredictOptions(
        test_img_dir=os.path.join(DATA_DIR, "T1", "ped_examples"),
        test_img_patterns="*.pgm",
        read_test_imgs_from_file=False,
        prediction_effort=0,              # Low
        display_predictions=False,        # No GUI display
        display_delay=0,
        prediction_output_dir=OUTPUT_DIR,
        complete_img_ped=True,            # All test images are pedestrians
        img_has_no_ped=False,
        compare_with_default_hog=False,
        read_from_reference=False,
    )

    assert os.path.isdir(predict_opts.test_img_dir), \
        f"Test dir not found: {predict_opts.test_img_dir}"
    print(f"  Test dir: {predict_opts.test_img_dir}")

    # ── 4. Run Prediction ──
    print("\n[STEP 4] Running prediction (sliding window)...")
    t0 = time.time()

    predictor = PedestrianPredictor(
        classifier, train_opts, predict_opts,
        log_callback=log, progress_callback=progress
    )
    predictor.run()

    t1 = time.time()
    print(f"  Prediction took {t1 - t0:.2f}s")
    print(f"  Images tested:   {len(predictor.all_images)}")
    print(f"  Pedestrian imgs: {len(predictor.ped_images)}")

    # ── 5. Evaluate ──
    print("\n[STEP 5] Running evaluation...")
    evaluator = Evaluator(predictor, train_opts, predict_opts)
    evaluator.write_all()

    img_s = predictor.img_summary
    rect_s = predictor.rect_summary

    print(f"  Image Accuracy:     {img_s.accuracy():.2%}")
    print(f"    TP={img_s.true_pos} TN={img_s.true_neg} FP={img_s.false_pos} FN={img_s.false_neg}")
    print(f"  Rectangle Accuracy: {rect_s.accuracy():.2%}")
    print(f"    TP={rect_s.true_pos} TN={rect_s.true_neg} FP={rect_s.false_pos} FN={rect_s.false_neg}")

    # Verify output files
    summary_path = os.path.join(OUTPUT_DIR, predict_opts.predict_summary)
    cm_path = os.path.join(OUTPUT_DIR, predict_opts.confusion_matrix)
    details_path = os.path.join(OUTPUT_DIR, predict_opts.per_file_details)

    for p in [summary_path, cm_path, details_path]:
        assert os.path.isfile(p), f"Output file not created: {p}"
        print(f"  Output: {os.path.basename(p)} ({os.path.getsize(p)} bytes)")

    print("\n" + "=" * 60)
    print("  ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
