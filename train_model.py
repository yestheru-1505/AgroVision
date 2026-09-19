import os
import cv2
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_DIR = "dataset"
MODEL_DIR = "model"

IMAGE_SIZE = (64, 64)

RANDOM_STATE = 42


# Create model directory
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(image_path):
    """
    Convert a crop image into numerical features.

    Features:
    1. HOG-like edge information
    2. Color histograms
    3. Color statistics
    4. Texture information
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    # Resize
    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    # -----------------------------------------------------
    # RGB IMAGE
    # -----------------------------------------------------

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    rgb_float = (
        rgb.astype(np.float32) / 255.0
    )

    # -----------------------------------------------------
    # HSV IMAGE
    # -----------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # -----------------------------------------------------
    # COLOR HISTOGRAM FEATURES
    # -----------------------------------------------------

    histogram_features = []

    for channel in range(3):

        hist = cv2.calcHist(
            [hsv],
            [channel],
            None,
            [32],
            [0, 256]
        )

        hist = cv2.normalize(
            hist,
            hist
        ).flatten()

        histogram_features.extend(hist)

    # -----------------------------------------------------
    # COLOR STATISTICS
    # -----------------------------------------------------

    color_statistics = []

    for channel in range(3):

        channel_data = (
            rgb_float[:, :, channel]
        )

        color_statistics.extend([
            np.mean(channel_data),
            np.std(channel_data),
            np.min(channel_data),
            np.max(channel_data)
        ])

    # -----------------------------------------------------
    # GRAYSCALE
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------------------
    # EDGE FEATURES
    # -----------------------------------------------------

    edges = cv2.Canny(
        gray,
        80,
        160
    )

    edge_density = np.mean(
        edges > 0
    )

    # -----------------------------------------------------
    # GRADIENT FEATURES
    # -----------------------------------------------------

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    gradient_magnitude = np.sqrt(
        sobel_x ** 2 +
        sobel_y ** 2
    )

    gradient_features = [
        np.mean(gradient_magnitude),
        np.std(gradient_magnitude),
        np.max(gradient_magnitude)
    ]

    # -----------------------------------------------------
    # DOWNSIZED PIXEL FEATURES
    # -----------------------------------------------------

    small_image = cv2.resize(
        rgb_float,
        (32, 32)
    )

    pixel_features = (
        small_image.flatten()
    )

    # -----------------------------------------------------
    # COMBINE EVERYTHING
    # -----------------------------------------------------

    features = np.concatenate([

        np.array(histogram_features),

        np.array(color_statistics),

        np.array([
            edge_density
        ]),

        np.array(gradient_features),

        pixel_features

    ])

    return features.astype(
        np.float32
    )


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    if not os.path.exists(DATASET_DIR):

        raise FileNotFoundError(
            "\nDataset folder not found.\n"
            "Create a folder named 'dataset' "
            "beside train_model.py."
        )

    class_names = sorted([

        folder

        for folder in os.listdir(
            DATASET_DIR
        )

        if os.path.isdir(
            os.path.join(
                DATASET_DIR,
                folder
            )
        )

    ])

    if len(class_names) < 2:

        raise ValueError(
            "\nAt least 2 disease classes "
            "are required."
        )

    print("\nClasses detected:")

    for index, name in enumerate(
        class_names
    ):

        print(
            f"{index}: {name}"
        )

    features = []

    labels = []

    # -----------------------------------------------------
    # READ EACH CLASS
    # -----------------------------------------------------

    for class_index, class_name in enumerate(
        class_names
    ):

        class_folder = os.path.join(
            DATASET_DIR,
            class_name
        )

        image_files = [

            file

            for file in os.listdir(
                class_folder
            )

            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                )
            )

        ]

        print(
            f"\n{class_name}: "
            f"{len(image_files)} images"
        )

        if len(image_files) < 2:

            raise ValueError(
                f"\nClass '{class_name}' "
                f"has fewer than 2 images."
            )

        successful = 0

        for filename in image_files:

            image_path = os.path.join(
                class_folder,
                filename
            )

            try:

                feature_vector = (
                    extract_features(
                        image_path
                    )
                )

                features.append(
                    feature_vector
                )

                labels.append(
                    class_index
                )

                successful += 1

            except Exception as error:

                print(
                    f"Skipped {filename}: "
                    f"{error}"
                )

        print(
            f"Successfully loaded: "
            f"{successful}"
        )

    if not features:

        raise ValueError(
            "No valid images were found."
        )

    return (

        np.array(features),

        np.array(labels),

        class_names

    )


# =========================================================
# TRAIN MODEL
# =========================================================

def main():

    print("=" * 65)

    print(
        "        AGROVISION AI MODEL TRAINING"
    )

    print("=" * 65)

    # Load data
    X, y, class_names = load_dataset()

    print(
        f"\nTotal images: {len(X)}"
    )

    print(
        f"Feature count: {X.shape[1]}"
    )

    # -----------------------------------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=RANDOM_STATE,

            stratify=y

        )
    )

    print(
        f"\nTraining images: "
        f"{len(X_train)}"
    )

    print(
        f"Testing images: "
        f"{len(X_test)}"
    )

    # -----------------------------------------------------
    # MACHINE LEARNING MODEL
    # -----------------------------------------------------

    model = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )

    ])

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    print(
        "\nTraining AgroVision AI model..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )

    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        "\n======================================"
    )

    print(
        "MODEL ACCURACY"
    )

    print(
        "======================================"
    )

    print(
        f"{accuracy * 100:.2f}%"
    )

    print(
        "\nCLASSIFICATION REPORT"
    )

    print(
        classification_report(

            y_test,

            predictions,

            target_names=class_names,

            zero_division=0

        )
    )

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "crop_disease_model.pkl"
    )

    class_path = os.path.join(
        MODEL_DIR,
        "class_names.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        class_names,
        class_path
    )

    print(
        "\n======================================"
    )

    print(
        "MODEL SAVED SUCCESSFULLY"
    )

    print(
        "======================================"
    )

    print(
        model_path
    )

    print(
        class_path
    )

    print(
        "\nAgroVision training completed!"
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()