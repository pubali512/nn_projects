"""Configuration dataclasses for training and prediction options."""

from dataclasses import dataclass, field


# Image constants (matching the Daimler Benchmark 18x36 format)
IMG_WIDTH = 18
IMG_HEIGHT = 36

# Classification labels
PED_CLASS = 1
NON_PED_CLASS = -1


@dataclass
class TrainOptions:
    """All parameters controlling the training phase."""

    # Number of sample images to be used for training
    ped_train_data_size: int = 50
    non_ped_train_data_size: int = 50

    # Where to find the sample images
    ped_train_data_path: str = ""
    non_ped_train_data_path: str = ""

    # Training output directory
    training_output_dir: str = ""

    # Training output files
    arff_file: str = "features.arff"
    ped_mean_pgm: str = "ped_mean.pgm"
    ped_covar_pgm: str = "ped_covar.pgm"
    non_ped_mean_pgm: str = "non_ped_mean.pgm"
    non_ped_covar_pgm: str = "non_ped_covar.pgm"

    ped_train_dump: str = "PedTrainingData"
    non_ped_train_dump: str = "NonPedTrainingData"

    # Feature extraction parameters
    feature_vec_type: int = 1          # 0 = greyscale, 1 = HOG
    hog_block_size: int = 12
    hog_cell_size: int = 6
    feature_vector_size: int = 0       # computed at runtime

    # How to take samples from non-pedestrian images
    scaled_image: bool = False         # True = scale entire image to 18x36
    cuts: int = 1                      # Number of random cuts per image
    cut_size_factor: int = 4           # Cut size = img_size * factor
    total_non_ped_train_data_size: int = 0  # computed at runtime


@dataclass
class PredictOptions:
    """All parameters controlling the prediction phase."""

    test_img_dir: str = ""
    test_img_patterns: str = "*.pgm"
    read_test_imgs_from_file: bool = False
    test_img_list_file: str = ""

    prediction_effort: int = 0         # 0 = Low, 1 = High
    display_predictions: bool = True
    display_delay: int = 1000          # milliseconds

    prediction_output_dir: str = ""
    prediction_subdir: str = "Predictions"

    # Summary output file names
    predict_summary: str = "prediction_summary.txt"
    confusion_matrix: str = "confusion_matrix.txt"
    per_file_details: str = "per_file_details.txt"

    # Reference comparison mode
    complete_img_ped: bool = False
    img_has_no_ped: bool = False
    compare_with_default_hog: bool = True
    read_from_reference: bool = False
    ref_file: str = ""


@dataclass
class FilePredictionDetails:
    """Per-file prediction details for evaluation."""

    file: str = ""
    total_examined_rects: int = 0

    true_pos: int = 0
    true_neg: int = 0
    false_pos: int = 0
    false_neg: int = 0

    img_predicted_ped: bool = False
    img_reference_ped: bool = False


@dataclass
class RefPrediction:
    """Reference prediction data for a single file."""

    file: str = ""
    rects: list = field(default_factory=list)  # list of (x, y, w, h) tuples


@dataclass
class SummaryData:
    """Aggregated prediction summary statistics."""

    total: int = 0
    true_pos: int = 0
    true_neg: int = 0
    false_pos: int = 0
    false_neg: int = 0

    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.true_pos + self.true_neg) / self.total

    def clear(self):
        self.total = 0
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
