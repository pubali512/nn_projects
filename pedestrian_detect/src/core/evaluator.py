"""Evaluation: generate summaries, confusion matrices, per-file details."""

import os
from core.config import SummaryData, TrainOptions, PredictOptions, IMG_HEIGHT, IMG_WIDTH
from utils.table_printer import TableCell, print_table


class Evaluator:
    """Generates prediction summaries, confusion matrices, per-file details.

    Faithful port of the C# WritePredictionSummary / CreateSummary /
    PrintBasicInfo / PrintConfusionMatrices / PrintPerFileDetails logic.
    """

    DELIM = "*"

    def __init__(self, predictor, train_opts: TrainOptions,
                 predict_opts: PredictOptions):
        self.pred = predictor
        self.train_opts = train_opts
        self.predict_opts = predict_opts

    def create_summary(self):
        """Compute image and rectangle summaries from prediction details."""
        self.pred.img_summary.clear()
        self.pred.rect_summary.clear()
        self.pred.rect_summary_unified.clear()

        self._create_img_summary()
        self._create_rect_summary(
            self.pred.rect_summary, self.pred.pred_details
        )
        self._create_rect_summary(
            self.pred.rect_summary_unified, self.pred.pred_details_unified
        )

    def _create_img_summary(self):
        s = self.pred.img_summary
        s.total = len(self.pred.all_images)

        for details in self.pred.pred_details.values():
            if details.img_predicted_ped:
                if details.img_reference_ped:
                    s.true_pos += 1
                else:
                    s.false_pos += 1
            else:
                if details.img_reference_ped:
                    s.false_neg += 1
                else:
                    s.true_neg += 1

    def _create_rect_summary(self, summary: SummaryData, pred_details: dict):
        total_rects = sum(self.pred.examined_rects.values())
        summary.total = total_rects

        for details in pred_details.values():
            summary.true_pos += details.true_pos
            summary.true_neg += details.true_neg
            summary.false_pos += details.false_pos
            summary.false_neg += details.false_neg

    def write_all(self):
        """Generate all summary files."""
        self.create_summary()
        self._write_prediction_summary()
        self._write_confusion_matrices()
        self._write_per_file_details()

    def _write_prediction_summary(self):
        filepath = os.path.join(
            self.predict_opts.prediction_output_dir,
            self.predict_opts.predict_summary
        )
        with open(filepath, "w") as f:
            self._print_basic_info(f)
            f.write("\n\n")
            self._print_img_and_rect_info(f)

    def _write_confusion_matrices(self):
        filepath = os.path.join(
            self.predict_opts.prediction_output_dir,
            self.predict_opts.confusion_matrix
        )
        with open(filepath, "w") as f:
            self._print_confusion_matrices(f)

    def _write_per_file_details(self):
        filepath = os.path.join(
            self.predict_opts.prediction_output_dir,
            self.predict_opts.per_file_details
        )
        with open(filepath, "w") as f:
            self._print_per_file_details(f)

    # ── Table builders ──

    def _delim_row(self, n_cols: int) -> list:
        return [TableCell(self.DELIM) for _ in range(n_cols)]

    def _print_basic_info(self, stream):
        d = self.DELIM
        table = []

        delim_row = self._delim_row(2)
        table.append(delim_row)

        header = TableCell("Training details")
        header.set_as_header()
        table.append([header, TableCell("")])
        table.append(delim_row)

        table.append([
            TableCell("Path to pedestrian images"),
            TableCell(self.train_opts.ped_train_data_path)
        ])
        table.append([
            TableCell("Number of pedestrian images"),
            TableCell(str(self.train_opts.ped_train_data_size))
        ])
        table.append([
            TableCell("Path to non-pedestrian images"),
            TableCell(self.train_opts.non_ped_train_data_path)
        ])
        table.append([
            TableCell("Number of non-pedestrian images"),
            TableCell(str(self.train_opts.non_ped_train_data_size))
        ])
        table.append([
            TableCell("Output directory"),
            TableCell(self.train_opts.training_output_dir)
        ])
        table.append(delim_row)

        if self.train_opts.scaled_image:
            table.append([
                TableCell("Sample selection method for non pedestrians"),
                TableCell("Scale down complete sample image to 18x36")
            ])
        else:
            table.append([
                TableCell("Sample selection method for non pedestrians"),
                TableCell("Cut samples from sample images")
            ])
            table.append([
                TableCell("Nr. of Samples"),
                TableCell(str(self.train_opts.cuts))
            ])
            sample_h = IMG_HEIGHT * self.train_opts.cut_size_factor
            sample_w = IMG_WIDTH * self.train_opts.cut_size_factor
            table.append([
                TableCell("Sample Size (Pixels)"),
                TableCell(f"{sample_w} x {sample_h}")
            ])

        table.append(delim_row)

        if self.train_opts.feature_vec_type == 0:
            table.append([
                TableCell("Feature vector type"),
                TableCell("Greyscale")
            ])
        else:
            table.append([
                TableCell("Feature vector type"),
                TableCell("HOG")
            ])
            bs = self.train_opts.hog_block_size
            table.append([
                TableCell("HOG Block Size"),
                TableCell(f"{bs} x {bs}")
            ])

        table.append([
            TableCell("Number of features"),
            TableCell(str(self.train_opts.feature_vector_size))
        ])

        table.append(delim_row)
        table.append([TableCell(""), TableCell("")])
        table.append(delim_row)

        header = TableCell("Result Summary")
        header.set_as_header()
        table.append([header, TableCell("")])
        table.append(delim_row)

        s = self.pred.img_summary
        table.append([
            TableCell("Total Images"),
            TableCell(str(s.total), False)
        ])
        table.append([
            TableCell("Accuracy"),
            TableCell(f"{s.accuracy():.2%}", False)
        ])
        table.append(delim_row)

        s = self.pred.rect_summary
        table.append([
            TableCell("Total Rectangles"),
            TableCell(str(s.total), False)
        ])
        table.append([
            TableCell("Accuracy"),
            TableCell(f"{s.accuracy():.2%}", False)
        ])
        table.append(delim_row)

        s = self.pred.rect_summary_unified
        table.append([
            TableCell("Total Rectangles (After Unification)"),
            TableCell(str(s.total), False)
        ])
        table.append([
            TableCell("Accuracy"),
            TableCell(f"{s.accuracy():.2%}", False)
        ])
        table.append(delim_row)

        print_table(table, stream)

    def _print_img_and_rect_info(self, stream):
        table = []
        delim_row = self._delim_row(4)

        header = TableCell("Prediction Summary (in Number of Images)")
        header.set_as_header()

        table.append(delim_row)
        table.append([header, TableCell(""), TableCell(""), TableCell("")])
        table.append(delim_row)

        table.append([
            TableCell(""),
            TableCell("Examined"),
            TableCell("Pedestrian"),
            TableCell("Non Pedestrian")
        ])
        table.append(delim_row)

        total_img = len(self.pred.all_images)
        ped_img = len(self.pred.ped_images)
        non_ped_img = total_img - ped_img

        ped_pct = ped_img / total_img if total_img else 0
        non_ped_pct = non_ped_img / total_img if total_img else 0

        table.append([
            TableCell(""),
            TableCell(str(total_img), False),
            TableCell(f"{ped_img} ({ped_pct:.2%})", False),
            TableCell(f"{non_ped_img} ({non_ped_pct:.2%})", False)
        ])
        table.append(delim_row)
        table.append(delim_row)

        header = TableCell("Prediction Summary (in Number of Rectangles)")
        header.set_as_header()
        table.append([header, TableCell(""), TableCell(""), TableCell("")])
        self._add_rect_info(table, self.pred.pedestrian_rects)

        table.append(delim_row)

        header = TableCell(
            "Prediction Summary (in Number of Rectangles) After Unification"
        )
        header.set_as_header()
        table.append([header, TableCell(""), TableCell(""), TableCell("")])
        self._add_rect_info(table, self.pred.pedestrian_rects_unified)

        print_table(table, stream)

    def _add_rect_info(self, table: list, ped_rects: dict):
        delim_row = self._delim_row(4)

        total_rects = sum(self.pred.examined_rects.values())

        table.append(delim_row)
        table.append([
            TableCell("Rectangle Sizes(Pixel)"),
            TableCell("Examined"),
            TableCell("Pedestrian"),
            TableCell("Non Pedestrian")
        ])
        table.append(delim_row)

        ped_total = 0
        non_ped_total = 0

        for rect_str, count in self.pred.examined_rects.items():
            ped_count = ped_rects.get(rect_str, 0)
            non_ped_count = count - ped_count

            t_pct = count / total_rects if total_rects else 0
            p_pct = ped_count / total_rects if total_rects else 0
            np_pct = non_ped_count / total_rects if total_rects else 0

            table.append([
                TableCell(rect_str),
                TableCell(f"{count}({t_pct:.2%})", False),
                TableCell(f"{ped_count}({p_pct:.2%})", False),
                TableCell(f"{non_ped_count}({np_pct:.2%})", False)
            ])

            ped_total += ped_count
            non_ped_total += non_ped_count

        ped_pct = ped_total / total_rects if total_rects else 0
        np_pct = non_ped_total / total_rects if total_rects else 0

        table.append(delim_row)
        table.append([
            TableCell("Total"),
            TableCell(str(total_rects), False),
            TableCell(f"{ped_total} ({ped_pct:.2%})", False),
            TableCell(f"{non_ped_total} ({np_pct:.2%})", False)
        ])
        table.append(delim_row)

    def _print_confusion_matrices(self, stream):
        table = []
        delim_row = self._delim_row(3)

        header = TableCell("Confusion matrix (in Number of Images)")
        header.set_as_header()
        table.append(delim_row)
        table.append([header, TableCell(""), TableCell("")])
        self._add_confusion_matrix(table, self.pred.img_summary)
        table.append(delim_row)

        header = TableCell("Confusion matrix (in Number of Rectangles)")
        header.set_as_header()
        table.append([header, TableCell(""), TableCell("")])
        self._add_confusion_matrix(table, self.pred.rect_summary)
        table.append(delim_row)

        header = TableCell(
            "Confusion matrix (in Number of Rectangles) After Unification"
        )
        header.set_as_header()
        table.append([header, TableCell(""), TableCell("")])
        self._add_confusion_matrix(table, self.pred.rect_summary_unified)

        print_table(table, stream)

    def _add_confusion_matrix(self, table: list, summary: SummaryData):
        delim_row = self._delim_row(3)

        tp = summary.true_pos
        tn = summary.true_neg
        fp = summary.false_pos
        fn = summary.false_neg
        total = summary.total

        tp_pct = tp / total if total else 0
        tn_pct = tn / total if total else 0
        fp_pct = fp / total if total else 0
        fn_pct = fn / total if total else 0

        table.append(delim_row)
        table.append([
            TableCell(""),
            TableCell("Non Pedestrian (Reference)"),
            TableCell("Pedestrian (Reference)")
        ])
        table.append(delim_row)

        table.append([
            TableCell("Non Pedestrian (Predicted)"),
            TableCell(f"{tn}({tn_pct:.2%})", False),
            TableCell(f"{fn}({fn_pct:.2%})", False)
        ])
        table.append(delim_row)

        table.append([
            TableCell("Pedestrian (Predicted)"),
            TableCell(f"{fp}({fp_pct:.2%})", False),
            TableCell(f"{tp}({tp_pct:.2%})", False)
        ])
        table.append(delim_row)

    def _print_per_file_details(self, stream):
        table = []
        delim_row = self._delim_row(2)

        table.append(delim_row)
        table.append([
            TableCell("Classification"),
            TableCell("Image file name")
        ])
        table.append(delim_row)

        for filepath in self.pred.all_images:
            classification = "Pedestrian" if filepath in self.pred.ped_images else "Non Pedestrian"
            table.append([
                TableCell(classification),
                TableCell(filepath)
            ])

        table.append(delim_row)
        print_table(table, stream)
