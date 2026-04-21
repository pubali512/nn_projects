"""Prediction pipeline: sliding window detection, rect unification, comparison."""

import os
import math
import numpy as np
import cv2

from core.config import (
    TrainOptions, PredictOptions, FilePredictionDetails, SummaryData,
    IMG_WIDTH, IMG_HEIGHT, PED_CLASS, NON_PED_CLASS
)
from core.feature_extraction import extract_feature_vector
from utils.file_utils import list_images, read_file_list, create_or_clean_dir

from sklearn.naive_bayes import GaussianNB


class PedestrianPredictor:
    """Slides a multi-scale window across test images and classifies patches.

    Faithful port of the C# RunClassifier / FindPedestrian /
    FindPedestrianRects / UnifyRects logic.
    """

    def __init__(self, classifier: GaussianNB,
                 train_opts: TrainOptions,
                 predict_opts: PredictOptions,
                 log_callback=None,
                 progress_callback=None):
        self.classifier = classifier
        self.train_opts = train_opts
        self.predict_opts = predict_opts
        self._log = log_callback or (lambda msg: None)
        self._progress = progress_callback or (lambda frac: None)

        # Result accumulators
        self.examined_rects: dict = {}          # rect_key -> count
        self.pedestrian_rects: dict = {}        # rect_key -> count
        self.pedestrian_rects_unified: dict = {}

        self.pred_details: dict = {}            # filepath -> FilePredictionDetails
        self.pred_details_unified: dict = {}

        self.ped_images: set = set()
        self.all_images: list = []

        self.ref_preds: dict = {}               # filepath -> RefPrediction

        self.img_summary = SummaryData()
        self.rect_summary = SummaryData()
        self.rect_summary_unified = SummaryData()

        # HOG detector for reference comparison
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # ── Rect key helpers ──

    @staticmethod
    def _rect_key(rect: tuple) -> str:
        """Generate a string key from a rect for counting."""
        x, y, w, h = rect
        return f"[{w}x{h}]"

    def _count_examined_rect(self, rect: tuple):
        key = self._rect_key(rect)
        self.examined_rects[key] = self.examined_rects.get(key, 0) + 1

    def _count_pedestrian_rect(self, rect: tuple):
        key = self._rect_key(rect)
        self.pedestrian_rects[key] = self.pedestrian_rects.get(key, 0) + 1

    def _count_pedestrian_rect_unified(self, rect: tuple):
        key = self._rect_key(rect)
        self.pedestrian_rects_unified[key] = (
            self.pedestrian_rects_unified.get(key, 0) + 1
        )

    # ── Classification ──

    def _is_pedestrian(self, sample: np.ndarray) -> bool:
        """Classify a single feature vector."""
        prediction = self.classifier.predict(sample.reshape(1, -1))[0]
        return prediction > 0

    def _extract_feature_vector(self, img: np.ndarray) -> np.ndarray:
        """Extract feature vector using current training options."""
        return extract_feature_vector(
            img,
            self.train_opts.feature_vec_type,
            self.train_opts.hog_block_size,
            self.train_opts.hog_cell_size
        )

    # ── Sliding window detection ──

    def _find_pedestrian_rects(self, img: np.ndarray, factor: int,
                                rects: list) -> tuple:
        """Slide a window at a specific scale factor.

        Args:
            img: Input image.
            factor: Scale factor (window = IMG_WIDTH*factor x IMG_HEIGHT*factor).
            rects: List to append detected rects to.

        Returns:
            (image_size_exceeded: bool, total_examined: int)
        """
        win_width = IMG_WIDTH * factor
        win_height = IMG_HEIGHT * factor

        height, width = img.shape[:2]
        total_examined = 0

        if win_height >= height and win_width >= width:
            resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            sample = self._extract_feature_vector(resized)
            cut_rect = (0, 0, width, height)

            self._count_examined_rect(cut_rect)
            total_examined += 1

            if self._is_pedestrian(sample):
                self._count_pedestrian_rect(cut_rect)
                rects.append(cut_rect)

            return True, total_examined  # image_size_exceeded

        # Determine step sizes based on effort
        if self.predict_opts.prediction_effort == 0:
            steps = [18]
        else:
            steps = [6, 12, 18]

        for step in steps:
            y = 0
            while y + win_height <= height:
                x = 0
                while x + win_width <= width:
                    cut_rect = (x, y, win_width, win_height)
                    cut_img = img[y:y + win_height, x:x + win_width]
                    resized = cv2.resize(cut_img, (IMG_WIDTH, IMG_HEIGHT))
                    sample = self._extract_feature_vector(resized)

                    self._count_examined_rect(cut_rect)
                    total_examined += 1

                    if self._is_pedestrian(sample):
                        self._count_pedestrian_rect(cut_rect)
                        rects.append(cut_rect)

                    x += step
                y += step

        return False, total_examined

    # ── Rect unification ──

    @staticmethod
    def _not_contained_in_larger(rect: tuple, start_factor: int,
                                  max_factor: int,
                                  rects_per_factor: dict) -> bool:
        """Check if a rect is NOT contained in any larger rect."""
        if start_factor > max_factor:
            return True

        x_start, y_start, w, h = rect
        x_end = x_start + w - 1
        y_end = y_start + h - 1

        for i in range(start_factor, max_factor + 1):
            if i not in rects_per_factor:
                continue
            for larger in rects_per_factor[i]:
                lx, ly, lw, lh = larger
                lx_end = lx + lw - 1
                ly_end = ly + lh - 1

                if (lx <= x_start and ly <= y_start and
                        x_end <= lx_end and y_end <= ly_end):
                    return False

        return True

    def _unify_rects(self, rects: list) -> list:
        """Remove smaller rects that are contained in larger ones."""
        if not rects:
            return []

        # Group by shrink factor
        rects_per_factor = {}
        factors_list = []

        for r in rects:
            x, y, w, h = r
            factor = math.ceil(h / IMG_HEIGHT)

            if factor not in rects_per_factor:
                rects_per_factor[factor] = []
                factors_list.append(factor)
            rects_per_factor[factor].append(r)

        factors_list.sort()
        max_factor = factors_list[-1]

        unified = []

        for factor in range(1, max_factor):
            if factor not in rects_per_factor:
                continue
            for r in rects_per_factor[factor]:
                if self._not_contained_in_larger(
                    r, factor + 1, max_factor, rects_per_factor
                ):
                    unified.append(r)
                    self._count_pedestrian_rect_unified(r)

        if max_factor in rects_per_factor:
            for r in rects_per_factor[max_factor]:
                unified.append(r)
                self._count_pedestrian_rect_unified(r)

        return unified

    # ── Reference comparison ──

    def _read_reference_predictions(self):
        """Read reference predictions from file."""
        ref_file = self.predict_opts.ref_file
        if not os.path.isfile(ref_file):
            return

        cur_file = ""
        with open(ref_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("=", 1)
                if len(parts) != 2:
                    continue

                key, val = parts[0].strip(), parts[1].strip()

                if key == "FILE":
                    cur_file = val
                elif key == "RECT" and cur_file:
                    fields = val.split(",")
                    if len(fields) == 4:
                        try:
                            rect = tuple(int(x.strip()) for x in fields)
                            if cur_file not in self.ref_preds:
                                self.ref_preds[cur_file] = []
                            self.ref_preds[cur_file].append(rect)
                        except ValueError:
                            pass

    def _get_reference_rects(self, filepath: str,
                              img: np.ndarray) -> list:
        """Get reference rects (from HOG detector or file)."""
        h, w = img.shape[:2]

        hog_win = self._hog.winSize
        if w < hog_win[0] or h < hog_win[1]:
            return None

        if self.predict_opts.read_from_reference:
            return self.ref_preds.get(filepath, None)
        else:
            # Use OpenCV default HOG people detector
            rects, _weights = self._hog.detectMultiScale(
                img, winStride=(8, 8), scale=1.05, groupThreshold=2
            )
            if rects is not None and len(rects) > 0:
                return [tuple(r) for r in rects]
            return None

    @staticmethod
    def _compare_rects(predicted: list, ref_rects: list) -> tuple:
        """Compare predicted rects against reference rects.

        Returns:
            (matches, not_covered_references)
        """
        matched_pred_indices = set()
        not_covered = set(range(len(ref_rects)))

        for rr_idx, rr in enumerate(ref_rects):
            rx, ry, rw, rh = rr
            for pr_idx, pr in enumerate(predicted):
                if pr_idx in matched_pred_indices:
                    continue  # already counted as a match
                px, py, pw, ph = pr

                # Calculate intersection
                ix = max(rx, px)
                iy = max(ry, py)
                ix2 = min(rx + rw, px + pw)
                iy2 = min(ry + rh, py + ph)

                if ix >= ix2 or iy >= iy2:
                    continue  # No intersection

                inter_area = (ix2 - ix) * (iy2 - iy)
                rr_area = rw * rh
                pr_area = pw * ph

                rr_fraction = inter_area / rr_area if rr_area else 0
                pr_fraction = inter_area / pr_area if pr_area else 0

                if rr_fraction > 0.5 and pr_fraction > 0.25:
                    matched_pred_indices.add(pr_idx)
                    not_covered.discard(rr_idx)
                    break  # move to next ref rect

        matches = [predicted[i] for i in matched_pred_indices]
        not_covered_rects = [ref_rects[i] for i in not_covered]
        return matches, not_covered_rects

    def _compare_with_reference(self, filepath: str, img: np.ndarray,
                                 predicted_rects: list,
                                 total_examined: int,
                                 output_dict: dict) -> tuple:
        """Compare predictions against reference and store details.

        Returns:
            (ref_rects, matching_rects) for visualization.
        """
        details = FilePredictionDetails(
            file=filepath,
            total_examined_rects=total_examined
        )

        ref_rects = None
        matching_rects = None
        opts = self.predict_opts

        if opts.complete_img_ped:
            details.img_reference_ped = True
            h, w = img.shape[:2]

            if len(predicted_rects) == 0:
                details.false_neg = 1
                details.true_neg = total_examined - 1
            else:
                # Any detection counts as a true positive
                details.true_pos = 1
                details.false_pos = max(0, len(predicted_rects) - 1)
                details.true_neg = total_examined - len(predicted_rects)
                details.img_predicted_ped = True

        elif opts.img_has_no_ped:
            details.false_pos = len(predicted_rects)
            details.true_neg = total_examined - len(predicted_rects)
            if len(predicted_rects) > 0:
                details.img_predicted_ped = True

        else:
            # Compare with reference (HOG or file)
            ref_rects = self._get_reference_rects(filepath, img)

            if len(predicted_rects) > 0:
                details.img_predicted_ped = True

            if ref_rects and len(ref_rects) > 0:
                matching_rects, not_covered = self._compare_rects(
                    predicted_rects, ref_rects
                )
                details.img_reference_ped = True
                details.true_pos = len(matching_rects)
                details.false_pos = len(predicted_rects) - len(matching_rects)
                details.false_neg = len(not_covered)
                tn = total_examined - (
                    details.true_pos + details.false_pos + details.false_neg
                )
                details.true_neg = max(0, tn)
            else:
                details.false_pos = len(predicted_rects)
                details.true_neg = total_examined - len(predicted_rects)

        output_dict[filepath] = details
        return ref_rects, matching_rects

    # ── Main detection ──

    def _find_pedestrian(self, filepath: str) -> bool:
        """Run multi-scale sliding window detection on a single image."""
        img = cv2.imread(filepath)
        if img is None:
            self._log(f"Could not read: {filepath}")
            return False

        shrink_factors = [1, 2, 3, 4, 5, 6]
        rects = []
        total_examined = 0

        for factor in shrink_factors:
            exceeded, examined = self._find_pedestrian_rects(
                img, factor, rects
            )
            total_examined += examined
            if exceeded:
                break

        # Compare with reference (before unification)
        ref_rects, matching_rects = self._compare_with_reference(
            filepath, img, rects, total_examined, self.pred_details
        )

        if len(rects) > 0:
            self.ped_images.add(filepath)
            self._log(f"Pedestrian     => {filepath}")

            unified_rects = self._unify_rects(rects)
            ref_rects_u, matching_rects_u = self._compare_with_reference(
                filepath, img, unified_rects, total_examined,
                self.pred_details_unified
            )

            create_ref_compare = (
                not self.predict_opts.complete_img_ped and
                not self.predict_opts.img_has_no_ped
            )
            ref_compare_img = img.copy() if create_ref_compare else None

            # Draw detections
            h, w = img.shape[:2]
            draw_rects = rects if (
                self.predict_opts.complete_img_ped and h == 96 and w == 48
            ) else unified_rects

            for r in draw_rects:
                x, y, rw, rh = r
                cv2.rectangle(img, (x, y), (x + rw, y + rh),
                              (0, 0, 255), 2)

            # Save result
            pred_dir = os.path.join(
                self.predict_opts.prediction_output_dir,
                self.predict_opts.prediction_subdir
            )
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            result_file = os.path.join(pred_dir, base_name + ".ppm")
            cv2.imwrite(result_file, img)

            if create_ref_compare and ref_compare_img is not None:
                if ref_rects_u:
                    for rr in ref_rects_u:
                        rx, ry, rw, rh = rr
                        cv2.rectangle(ref_compare_img, (rx, ry),
                                      (rx + rw, ry + rh), (255, 255, 0), 3)
                if matching_rects_u:
                    for mr in matching_rects_u:
                        mx, my, mw, mh = mr
                        cv2.rectangle(ref_compare_img, (mx, my),
                                      (mx + mw, my + mh), (0, 0, 255), 2)

                ref_file = os.path.join(
                    pred_dir, base_name + "_ref_compare.ppm"
                )
                cv2.imwrite(ref_file, ref_compare_img)

            # Display if requested (skip cv2.imshow to avoid
            # macOS crash when called from a background thread)
            # Images are saved to the Predictions folder instead.

            return True
        else:
            self._log(f"Non Pedestrian => {filepath}")

            # Still need to create entry in unified dict
            self._compare_with_reference(
                filepath, img, [], total_examined,
                self.pred_details_unified
            )
            return False

    # ── Public API ──

    def run(self):
        """Main prediction entry point.

        Collects test files, runs detection on each, accumulates results.
        """
        test_files = []

        if not self.predict_opts.read_test_imgs_from_file:
            patterns = self.predict_opts.test_img_patterns
            test_files = list_images(
                self.predict_opts.test_img_dir, patterns
            )
        else:
            test_files = read_file_list(
                self.predict_opts.test_img_list_file
            )

        if len(test_files) == 0:
            raise ValueError(
                "No test files found. Check your test directory or file list."
            )

        # Clear accumulators
        self.examined_rects.clear()
        self.pedestrian_rects.clear()
        self.pedestrian_rects_unified.clear()
        self.pred_details.clear()
        self.pred_details_unified.clear()
        self.ped_images.clear()
        self.all_images.clear()
        self.ref_preds.clear()

        # Prepare output directory
        pred_dir = os.path.join(
            self.predict_opts.prediction_output_dir,
            self.predict_opts.prediction_subdir
        )
        create_or_clean_dir(pred_dir)

        if self.predict_opts.read_from_reference:
            self._read_reference_predictions()

        total = len(test_files)
        for idx, filepath in enumerate(test_files):
            self.all_images.append(filepath)
            self._find_pedestrian(filepath)
            self._progress((idx + 1) / total)

        cv2.destroyAllWindows()

        # Note: cv2.destroyAllWindows() is also called in predict_tab._predict_done
        # on the main thread, but this call handles the headless (non-GUI) path.
