from flask import (
    Flask,
    request,
    render_template_string
)

import os
import cv2
import joblib
import numpy as np

from werkzeug.utils import secure_filename


# =========================================================
# CONFIGURATION
# =========================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "model"

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "webp"
}

app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER

app.config[
    "MAX_CONTENT_LENGTH"
] = 10 * 1024 * 1024


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "crop_disease_model.pkl"
)

CLASS_PATH = os.path.join(
    MODEL_FOLDER,
    "class_names.pkl"
)


try:

    model = joblib.load(
        MODEL_PATH
    )

    class_names = joblib.load(
        CLASS_PATH
    )

    print(
        "\nAI model loaded successfully."
    )

    print(
        "Classes:",
        class_names
    )

except Exception as error:

    model = None

    class_names = []

    print(
        "\nERROR loading AI model:"
    )

    print(error)


# =========================================================
# FEATURE EXTRACTION
# MUST MATCH TRAINING CODE
# =========================================================

IMAGE_SIZE = (64, 64)


def extract_features(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            "Unable to read uploaded image."
        )

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    rgb_float = (
        rgb.astype(np.float32) / 255.0
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # -----------------------------------------------------
    # HISTOGRAM
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
    # EDGE
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
    # GRADIENT
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

        np.mean(
            gradient_magnitude
        ),

        np.std(
            gradient_magnitude
        ),

        np.max(
            gradient_magnitude
        )

    ]

    # -----------------------------------------------------
    # SMALL IMAGE FEATURES
    # -----------------------------------------------------

    small_image = cv2.resize(
        rgb_float,
        (32, 32)
    )

    pixel_features = (
        small_image.flatten()
    )

    # -----------------------------------------------------
    # FINAL FEATURE VECTOR
    # -----------------------------------------------------

    features = np.concatenate([

        np.array(
            histogram_features
        ),

        np.array(
            color_statistics
        ),

        np.array([
            edge_density
        ]),

        np.array(
            gradient_features
        ),

        pixel_features

    ])

    return features.astype(
        np.float32
    )


# =========================================================
# PREDICTION
# =========================================================

def predict_disease(image_path):

    if model is None:

        return (
            "Model not loaded",
            0.0
        )

    try:

        features = extract_features(
            image_path
        )

        features = features.reshape(
            1,
            -1
        )

        # Prediction
        predicted_class = (
            model.predict(
                features
            )[0]
        )

        disease = class_names[
            int(predicted_class)
        ]

        # Probability
        probabilities = (
            model.predict_proba(
                features
            )[0]
        )

        confidence = float(
            np.max(
                probabilities
            )
        )

        return (
            disease,
            confidence
        )

    except Exception as error:

        print(
            "Prediction error:",
            error
        )

        return (
            "Prediction failed",
            0.0
        )


# =========================================================
# FARMER-FRIENDLY RECOMMENDATION
# =========================================================

def get_advice(disease):

    name = disease.lower()

    if "healthy" in name:

        return (
            "The plant appears healthy. "
            "Continue regular watering, "
            "proper nutrition and regular "
            "crop monitoring."
        )

    if "blight" in name:

        return (
            "Remove severely affected leaves, "
            "keep the leaves as dry as possible "
            "and consult a local agriculture "
            "expert for suitable treatment."
        )

    if "rust" in name:

        return (
            "Remove badly affected leaves, "
            "maintain good air circulation and "
            "consult a local agriculture expert "
            "for suitable treatment."
        )

    if "spot" in name:

        return (
            "Remove affected leaves and "
            "avoid unnecessary overhead watering. "
            "Monitor nearby plants."
        )

    if "mildew" in name:

        return (
            "Improve air circulation around the "
            "plants and reduce prolonged leaf "
            "moisture. Consult an agriculture "
            "expert for treatment."
        )

    return (
        "Monitor the crop carefully and "
        "consult a local agriculture expert "
        "before applying any treatment."
    )


# =========================================================
# HTML
# =========================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>
AgroVision - AI Crop Disease Detection
</title>

<style>

body {

    margin: 0;

    font-family: Arial, sans-serif;

    background: #eef7ea;

    color: #222;

}

header {

    background: #176b2c;

    color: white;

    text-align: center;

    padding: 25px;

}

header h1 {

    margin: 0;

    font-size: 35px;

}

.container {

    max-width: 750px;

    width: 90%;

    margin: 40px auto;

    padding: 35px;

    background: white;

    border-radius: 20px;

    box-shadow:
        0 5px 20px
        rgba(0,0,0,0.12);

    text-align: center;

}

h2 {

    color: #176b2c;

}

input[type=file] {

    margin: 20px;

    padding: 10px;

    max-width: 90%;

}

button {

    border: none;

    border-radius: 8px;

    padding: 13px 22px;

    margin: 7px;

    background: #176b2c;

    color: white;

    font-size: 16px;

    cursor: pointer;

}

button:hover {

    background: #0c4d1e;

}

.result {

    margin-top: 30px;

    padding: 25px;

    background: #e7f5e9;

    border-radius: 15px;

    text-align: left;

}

.result h2 {

    text-align: center;

}

.confidence {

    font-size: 20px;

    font-weight: bold;

}

.warning {

    margin-top: 20px;

    padding: 15px;

    background: #fff3cd;

    border-radius: 10px;

}

.error {

    color: red;

    font-weight: bold;

}

.voice-box {

    text-align: center;

    margin-top: 25px;

}

footer {

    text-align: center;

    color: #555;

    padding: 25px;

}

</style>

</head>


<body>


<header>

<h1>🌱 AgroVision</h1>

<p>
AI-Based Crop Disease Detection System
</p>

</header>


<div class="container">


<h2>
Detect Crop Disease
</h2>


<p>
Upload a crop leaf image and let the
trained AI model analyze it.
</p>


<form
method="POST"
enctype="multipart/form-data"
>


<input
type="file"
name="crop_image"
accept=".jpg,.jpeg,.png,.bmp,.webp"
required
>


<br>


<button type="submit">

🔍 Detect Disease

</button>


</form>


{% if error %}

<p class="error">

{{ error }}

</p>

{% endif %}


{% if disease %}


<div class="result">


<h2>
🌿 AI Detection Result
</h2>


<p>

<strong>
Image:
</strong>

{{ filename }}

</p>


<p>

<strong>
Detected Disease:
</strong>

{{ disease }}

</p>


<p class="confidence">

AI Confidence:
{{ confidence }}%

</p>


<p>

<strong>
Recommended Action:
</strong>

</p>


<p>

{{ advice }}

</p>


<div class="voice-box">


<hr>


<h3>
🔊 Farmer Voice Explanation
</h3>


<button
type="button"
onclick="speakEnglish()">

🔊 English

</button>


<button
type="button"
onclick="speakTelugu()">

🔊 తెలుగు

</button>


</div>


{% if confidence_value < 60 %}

<div class="warning">

⚠️ AI confidence is below 60%.
The result should be treated as
uncertain. Please verify the symptoms
with an agriculture expert.

</div>

{% endif %}


</div>


{% endif %}


</div>


<footer>

AgroVision |
Smart Farming for a Better Future

</footer>


<script>


const disease =
{{ disease|tojson }};

const confidence =
{{ confidence|tojson }};

const advice =
{{ advice|tojson }};


function speakEnglish() {

    window.speechSynthesis.cancel();

    const message =
        "The detected crop disease is "
        + disease
        + ". The AI confidence is "
        + confidence
        + " percent. "
        + "Recommended action: "
        + advice;

    const speech =
        new SpeechSynthesisUtterance(
            message
        );

    speech.lang = "en-IN";

    speech.rate = 0.9;

    speech.pitch = 1.0;

    window.speechSynthesis.speak(
        speech
    );
}


function speakTelugu() {

    window.speechSynthesis.cancel();

    const message =
        "గుర్తించిన పంట వ్యాధి "
        + disease
        + ". AI నమ్మక స్థాయి "
        + confidence
        + " శాతం. "
        + "సూచించిన చర్య: "
        + advice;

    const speech =
        new SpeechSynthesisUtterance(
            message
        );

    speech.lang = "te-IN";

    speech.rate = 0.8;

    speech.pitch = 1.0;

    window.speechSynthesis.speak(
        speech
    );
}


</script>


</body>

</html>

"""


# =========================================================
# FILE CHECK
# =========================================================

def allowed_file(filename):

    return (

        "." in filename

        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS

    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)

def home():

    filename = None
    disease = None
    confidence = None
    confidence_value = 0
    advice = None
    error = None


    if request.method == "POST":

        # ---------------------------------------------
        # CHECK FILE
        # ---------------------------------------------

        if "crop_image" not in request.files:

            error = (
                "Please select a crop image."
            )

        else:

            file = request.files[
                "crop_image"
            ]


            if file.filename == "":

                error = (
                    "No image selected."
                )


            elif not allowed_file(
                file.filename
            ):

                error = (
                    "Please upload JPG, JPEG, "
                    "PNG, BMP or WEBP image."
                )


            else:

                filename = secure_filename(
                    file.filename
                )


                image_path = os.path.join(

                    app.config[
                        "UPLOAD_FOLDER"
                    ],

                    filename

                )


                file.save(
                    image_path
                )


                # ---------------------------------
                # AI PREDICTION
                # ---------------------------------

                (
                    disease,
                    confidence_value
                ) = predict_disease(
                    image_path
                )


                confidence = round(

                    confidence_value * 100,

                    2

                )


                advice = get_advice(
                    disease
                )


    return render_template_string(

        HTML,

        filename=filename,

        disease=disease,

        confidence=confidence,

        confidence_value=(
            confidence_value * 100
        ),

        advice=advice,

        error=error

    )


# =========================================================
# START FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )