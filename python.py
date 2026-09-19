import os

# Get the folder where train_model1.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
def load_dataset():

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(
            f"\nDataset folder not found:\n{DATASET_DIR}\n\n"
            "Create a 'dataset' folder and put your "
            "class folders inside it."
        )

    class_names = sorted([
        folder
        for folder in os.listdir(DATASET_DIR)
        if os.path.isdir(
            os.path.join(DATASET_DIR, folder)
        )
    ])

    if len(class_names) < 2:
        raise ValueError(
            "\nAt least 2 class folders are required.\n"
            f"Current dataset folder:\n{DATASET_DIR}\n"
        )

    print("\nDataset location:")
    print(DATASET_DIR)

    print("\nClasses detected:")

    for index, name in enumerate(class_names):
        print(f"{index}: {name}")

    features = []
    labels = []

    for class_index, class_name in enumerate(class_names):

        class_folder = os.path.join(
            DATASET_DIR,
            class_name
        )

        image_files = [
            file
            for file in os.listdir(class_folder)
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

        if len(image_files) == 0:
            print(
                f"WARNING: No images found in "
                f"{class_folder}"
            )
            continue

        for filename in image_files:

            image_path = os.path.join(
                class_folder,
                filename
            )

            try:

                feature_vector = extract_features(
                    image_path
                )

                features.append(
                    feature_vector
                )

                labels.append(
                    class_index
                )

            except Exception as error:

                print(
                    f"Skipped {filename}: "
                    f"{error}"
                )

    if len(features) == 0:

        raise ValueError(
            "\nNo valid images were found.\n"
            "Put JPG/PNG images inside your "
            "disease class folders."
        )

    return (
        np.array(features),
        np.array(labels),
        class_names
    )