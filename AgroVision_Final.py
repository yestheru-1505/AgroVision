from flask import Flask, request, render_template_string, send_from_directory
import os
import cv2
import joblib
import numpy as np
from uuid import uuid4
from werkzeug.utils import secure_filename

# ============================================================
# AgroVision - Python 3.14, Flask, OpenCV, scikit-learn
# No TensorFlow
# Requires model/crop_disease_model.pkl and model/class_names.pkl
# ============================================================

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "model")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_FOLDER, "crop_disease_model.pkl")
CLASS_PATH = os.path.join(MODEL_FOLDER, "class_names.pkl")

# -----------------------------
# Load trained model
# -----------------------------
model = None
class_names = []
try:
    model = joblib.load(MODEL_PATH)
    class_names = joblib.load(CLASS_PATH)
    print("AI model loaded successfully")
    print("Classes:", class_names)
except Exception as exc:
    print("Trained model not loaded.")
    print("Run train_model1.py first.")
    print("Reason:", exc)

# -----------------------------
# Same features used by training
# -----------------------------
IMAGE_SIZE = (64, 64)

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to read image")

    image = cv2.resize(image, IMAGE_SIZE)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_float = rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    histogram_features = []
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histogram_features.extend(hist)

    color_statistics = []
    for channel in range(3):
        data = rgb_float[:, :, channel]
        color_statistics.extend([
            np.mean(data), np.std(data), np.min(data), np.max(data)
        ])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.mean(edges > 0)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_features = [np.mean(magnitude), np.std(magnitude), np.max(magnitude)]

    small_image = cv2.resize(rgb_float, (32, 32)).flatten()

    return np.concatenate([
        np.array(histogram_features),
        np.array(color_statistics),
        np.array([edge_density]),
        np.array(gradient_features),
        small_image
    ]).astype(np.float32)

# -----------------------------
# English + Telugu disease guide
# General management guidance; pesticide use must follow current local advice/label.
# -----------------------------
GUIDES = {
    "healthy": {
        "en": {
            "name": "Healthy Plant",
            "symptoms": "No major disease pattern was identified by the model.",
            "treatment": "Continue regular monitoring, suitable irrigation, balanced nutrition and field sanitation.",
            "prevention": "Use healthy planting material, maintain field hygiene and inspect plants regularly."
        },
        "te": {
            "name": "ఆరోగ్యకరమైన మొక్క",
            "symptoms": "మోడల్ ప్రకారం ముఖ్యమైన వ్యాధి లక్షణాలు గుర్తించబడలేదు.",
            "treatment": "పంటను క్రమంగా పరిశీలించండి. తగిన నీరు, సమతుల పోషకాలు మరియు పొలం పరిశుభ్రత పాటించండి.",
            "prevention": "ఆరోగ్యకరమైన విత్తనాలు లేదా నాట్లను ఉపయోగించి మొక్కలను క్రమం తప్పకుండా పరిశీలించండి."
        }
    },
    "tomato early blight": {
        "en": {
            "name": "Tomato Early Blight",
            "symptoms": "Brown leaf spots, often with concentric-ring patterns, commonly starting on older leaves.",
            "treatment": "Remove severely affected leaves and infected debris, improve sanitation, avoid unnecessary leaf wetness, and follow current locally registered disease-management advice.",
            "prevention": "Use healthy planting material, rotate crops where suitable, and do not leave infected debris in the field."
        },
        "te": {
            "name": "టమాటా ఎర్లీ బ్లైట్",
            "symptoms": "పాత ఆకులపై గోధుమ రంగు మచ్చలు, కొన్నిసార్లు వలయాల్లాంటి ఆకృతి కనిపించవచ్చు.",
            "treatment": "తీవ్రంగా సోకిన ఆకులు మరియు అవశేషాలను తొలగించండి. పొలం పరిశుభ్రత పాటించి, ఆకులు ఎక్కువసేపు తడిగా ఉండకుండా చూడండి. స్థానిక తాజా సూచనలను అనుసరించండి.",
            "prevention": "ఆరోగ్యకరమైన నాట్లను ఉపయోగించండి, సాధ్యమైన చోట పంట మార్పిడి చేయండి మరియు సోకిన అవశేషాలను పొలంలో ఉంచవద్దు."
        }
    },
    "potato late blight": {
        "en": {
            "name": "Potato Late Blight",
            "symptoms": "Water-soaked or dark leaf and stem lesions; the disease can spread quickly in cool, wet conditions and may affect tubers.",
            "treatment": "Remove badly affected foliage, maintain good drainage, avoid storing visibly infected tubers, and follow current local potato late-blight advice.",
            "prevention": "Use healthy seed tubers, monitor during cool/wet weather, maintain sanitation, and avoid injuries to tubers at harvest."
        },
        "te": {
            "name": "బంగాళాదుంప లేట్ బ్లైట్",
            "symptoms": "ఆకులు మరియు కాండంపై నీటితో తడిసినట్లు లేదా నల్లటి మచ్చలు కనిపించవచ్చు. చల్లని, తడి వాతావరణంలో వ్యాధి వేగంగా వ్యాపించవచ్చు.",
            "treatment": "తీవ్రంగా సోకిన ఆకులను తొలగించండి, మంచి నీటి పారుదల కల్పించండి, స్పష్టంగా సోకిన దుంపలను నిల్వ చేయవద్దు మరియు స్థానిక తాజా సూచనలను అనుసరించండి.",
            "prevention": "ఆరోగ్యకరమైన విత్తన దుంపలను ఉపయోగించండి, చల్లని/తడి వాతావరణంలో పర్యవేక్షించండి మరియు కోత సమయంలో దుంపలకు గాయాలు కాకుండా చూడండి."
        }
    },
    "rice leaf blast": {
        "en": {
            "name": "Rice Leaf Blast",
            "symptoms": "Leaf lesions can become spindle or diamond shaped and enlarge under favorable conditions.",
            "treatment": "Monitor the field closely, maintain balanced crop management, improve sanitation and follow the latest local rice-blast advisory.",
            "prevention": "Use healthy seed, suitable varieties where available, balanced nutrient management and regular scouting."
        },
        "te": {
            "name": "వరి లీఫ్ బ్లాస్ట్",
            "symptoms": "ఆకులపై వజ్రం లేదా పొడవైన మచ్చల వంటి గాయాలు ఏర్పడి అనుకూల పరిస్థితుల్లో పెద్దవిగా మారవచ్చు.",
            "treatment": "పొలాన్ని తరచుగా పరిశీలించండి, సమతుల పంట నిర్వహణ పాటించండి, పొలం పరిశుభ్రతను మెరుగుపరచండి మరియు స్థానిక తాజా బ్లాస్ట్ సూచనలను అనుసరించండి.",
            "prevention": "ఆరోగ్యకరమైన విత్తనం, తగిన రకాలు, సమతుల పోషక నిర్వహణ మరియు క్రమం తప్పని పర్యవేక్షణ ఉపయోగకరం."
        }
    },
    "wheat leaf rust": {
        "en": {
            "name": "Wheat Leaf Rust",
            "symptoms": "Rust-colored pustules can appear on leaves and spread under favorable conditions.",
            "treatment": "Monitor the crop frequently and follow current local wheat-rust management advice when disease pressure increases.",
            "prevention": "Use locally recommended resistant or tolerant varieties when available and maintain regular field scouting."
        },
        "te": {
            "name": "గోధుమ లీఫ్ రస్ట్",
            "symptoms": "ఆకులపై తుప్పు రంగు చిన్న పొక్కులు లేదా మచ్చలు కనిపించవచ్చు.",
            "treatment": "పంటను తరచుగా పరిశీలించండి. వ్యాధి పెరిగినప్పుడు స్థానిక తాజా గోధుమ రస్ట్ నియంత్రణ సూచనలను అనుసరించండి.",
            "prevention": "అందుబాటులో ఉన్నప్పుడు స్థానికంగా సిఫారసు చేసిన నిరోధక లేదా సహనశీల రకాలను ఎంచుకోండి."
        }
    },
    "maize leaf spot": {
        "en": {
            "name": "Maize Leaf Spot",
            "symptoms": "Leaf spots can develop and enlarge as infection progresses.",
            "treatment": "Manage severely affected debris, maintain field sanitation, monitor nearby plants and follow current local maize disease advice.",
            "prevention": "Use healthy seed, balanced crop nutrition, crop rotation where appropriate and good field sanitation."
        },
        "te": {
            "name": "మొక్కజొన్న లీఫ్ స్పాట్",
            "symptoms": "ఆకులపై మచ్చలు ఏర్పడి వ్యాధి పెరిగే కొద్దీ అవి పెద్దవిగా మారవచ్చు.",
            "treatment": "తీవ్రంగా సోకిన అవశేషాలను నిర్వహించండి, పొలం పరిశుభ్రత పాటించండి, పక్క మొక్కలను పరిశీలించండి మరియు స్థానిక తాజా సూచనలను అనుసరించండి.",
            "prevention": "ఆరోగ్యకరమైన విత్తనం, సమతుల పోషక నిర్వహణ, అవసరమైన చోట పంట మార్పిడి మరియు పొలం పరిశుభ్రత పాటించండి."
        }
    }
}

GENERIC = {
    "en": {
        "name": "Possible Crop Disease",
        "symptoms": "The model identified the closest class in your training dataset. Confirm the visible symptoms before treatment.",
        "treatment": "Remove severely affected plant parts where appropriate, maintain field sanitation, monitor nearby plants and consult a local agriculture expert before using any pesticide or other treatment.",
        "prevention": "Use healthy planting material, maintain field hygiene and monitor the crop regularly."
    },
    "te": {
        "name": "సంభావ్య పంట వ్యాధి",
        "symptoms": "మీ శిక్షణ డేటాసెట్‌లో మోడల్‌కు దగ్గరగా ఉన్న తరగతిని గుర్తించింది. చికిత్సకు ముందు లక్షణాలను ధృవీకరించండి.",
        "treatment": "అవసరమైతే తీవ్రంగా సోకిన భాగాలను తొలగించండి, పొలం పరిశుభ్రత పాటించండి మరియు పురుగుమందు లేదా ఇతర చికిత్సకు ముందు స్థానిక వ్యవసాయ నిపుణుడిని సంప్రదించండి.",
        "prevention": "ఆరోగ్యకరమైన నాట్లను ఉపయోగించండి, పొలం పరిశుభ్రత పాటించండి మరియు పంటను క్రమం తప్పకుండా పరిశీలించండి."
    }
}


def normalize(label):
    return str(label).lower().replace("_", " ").replace("-", " ").replace("/", " ")


def get_guide(label):
    text = normalize(label)
    for key, guide in GUIDES.items():
        if key in text:
            return guide
    if "healthy" in text:
        return GUIDES["healthy"]
    return GENERIC


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_disease(image_path):
    if model is None:
        raise RuntimeError("Trained model not found. Run train_model1.py first.")

    features = extract_features(image_path).reshape(1, -1)
    predicted_index = int(model.predict(features)[0])

    if predicted_index >= len(class_names):
        raise RuntimeError("Model/classes mismatch. Retrain the model.")

    label = str(class_names[predicted_index])

    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(features)[0]))
    else:
        confidence = 0.0

    return label, confidence

# ============================================================
# UI
# ============================================================
HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgroVision - AI Crop Disease Assistant</title>
<style>
*{box-sizing:border-box} body{margin:0;font-family:Arial,sans-serif;background:#eef7ee;color:#203b24}
header{background:#176b35;color:#fff;padding:25px;text-align:center} header h1{margin:0 0 8px;font-size:34px}
.container{max-width:1000px;width:92%;margin:30px auto;background:#fff;padding:28px;border-radius:18px;box-shadow:0 5px 18px rgba(0,0,0,.12)}
.upload{border:2px dashed #4aa35a;padding:25px;text-align:center;border-radius:14px;background:#f8fff8}
input,select{width:min(90%,420px);padding:12px;margin:10px 0;border:1px solid #aaa;border-radius:8px;font-size:15px}
button{background:#176b35;color:#fff;border:0;border-radius:9px;padding:12px 18px;margin:6px;font-size:15px;cursor:pointer}button:hover{background:#0e4d25}
.result{margin-top:28px;padding:24px;background:#eaf8eb;border-left:6px solid #176b35;border-radius:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}.card{background:#f8fff8;padding:18px;border-radius:12px;border:1px solid #dcecdc}.card h3{color:#176b35;margin-top:0}
.preview{display:block;max-width:320px;max-height:320px;margin:20px auto;border-radius:12px}.confidence{font-size:21px;font-weight:bold}.warn{margin-top:18px;padding:14px;background:#fff3cd;border-radius:10px}.error{color:#b00020;font-weight:bold}.voice{text-align:center;margin-top:22px;padding:18px;background:#f2fbf2;border-radius:12px}.note{font-size:13px;color:#5f6d61;line-height:1.5}footer{text-align:center;padding:20px;color:#666}
@media(max-width:750px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><h1>🌱 AgroVision</h1><p>AI Crop Disease Detection, Treatment & Voice Assistance</p></header>
<div class="container">
<h2>🔍 Crop Disease Detection</h2>
<div class="upload">
<form method="POST" enctype="multipart/form-data">
<h3>Upload Crop Leaf Image</h3>
<input type="file" name="image" accept="image/*" required><br>
<label><b>Select Crop</b></label><br>
<select name="crop">
<option>Tomato</option><option>Potato</option><option>Rice</option><option>Wheat</option><option>Maize</option><option>Other</option>
</select><br>
<button type="submit">🤖 Detect Disease</button>
</form>
</div>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
{% if result %}
<div class="result">
<h2>🌿 AI Detection Result</h2>
{% if image %}<img class="preview" src="/uploads/{{ image }}" alt="Uploaded crop leaf">{% endif %}
<p><b>Crop:</b> {{ crop }}</p>
<p><b>Detected Disease:</b> {{ result }}</p>
<p class="confidence">AI Confidence: {{ confidence }}%</p>
<div class="grid">
<div class="card"><h3>🇬🇧 English</h3><p><b>Disease:</b><br>{{ guide_en.name }}</p><p><b>Symptoms:</b><br>{{ guide_en.symptoms }}</p><p><b>Treatment / Management:</b><br>{{ guide_en.treatment }}</p><p><b>Prevention:</b><br>{{ guide_en.prevention }}</p></div>
<div class="card"><h3>🇮🇳 తెలుగు</h3><p><b>వ్యాధి:</b><br>{{ guide_te.name }}</p><p><b>లక్షణాలు:</b><br>{{ guide_te.symptoms }}</p><p><b>చికిత్స / నిర్వహణ:</b><br>{{ guide_te.treatment }}</p><p><b>నివారణ:</b><br>{{ guide_te.prevention }}</p></div>
</div>
<div class="voice">
<h3>🔊 Voice Assistance</h3>
<button type="button" onclick="speakEnglish()">🔊 Listen in English</button>
<button type="button" onclick="speakTelugu()">🔊 తెలుగులో వినండి</button>
<button type="button" onclick="stopVoice()">⏹ Stop</button>
<p class="note">Telugu voice availability depends on the speech voices installed in your browser/Windows system.</p>
</div>
{% if confidence_value < 60 %}<div class="warn">⚠️ Confidence is below 60%. Treat this as a screening result and verify the symptoms with a qualified agriculture professional.</div>{% endif %}
<p class="note">AgroVision is a decision-support tool based on the trained dataset. Verify a diagnosis before treatment. Follow current local agricultural recommendations and registered product labels.</p>
</div>
<script>
const enName={{ guide_en.name|tojson }},teName={{ guide_te.name|tojson }};
const enSymptoms={{ guide_en.symptoms|tojson }},teSymptoms={{ guide_te.symptoms|tojson }};
const enTreatment={{ guide_en.treatment|tojson }},teTreatment={{ guide_te.treatment|tojson }};
const enPrevention={{ guide_en.prevention|tojson }},tePrevention={{ guide_te.prevention|tojson }};
const confidence={{ confidence|tojson }};
function getVoice(lang){const v=speechSynthesis.getVoices();return v.find(x=>x.lang&&x.lang.toLowerCase().startsWith(lang.toLowerCase()))||null;}
function speak(text,lang){speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.lang=lang;u.rate=lang==='te-IN'?0.8:0.9;const v=getVoice(lang);if(v)u.voice=v;speechSynthesis.speak(u);}
function speakEnglish(){speak('The detected disease is '+enName+'. Confidence is '+confidence+' percent. Symptoms: '+enSymptoms+'. Treatment or management: '+enTreatment+'. Prevention: '+enPrevention,'en-IN');}
function speakTelugu(){speak('గుర్తించిన వ్యాధి: '+teName+'. ఏ ఐ నమ్మక స్థాయి '+confidence+' శాతం. లక్షణాలు: '+teSymptoms+'. చికిత్స లేదా నిర్వహణ: '+teTreatment+'. నివారణ: '+tePrevention,'te-IN');}
function stopVoice(){speechSynthesis.cancel();}
</script>
{% endif %}
</div>
<footer>© 2026 AgroVision | AI for Smart Agriculture</footer>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = confidence = crop = image_name = error = None
    confidence_value = 0
    guide_en = GENERIC["en"]
    guide_te = GENERIC["te"]

    if request.method == "POST":
        crop = request.form.get("crop", "Other")
        image = request.files.get("image")

        if image is None or image.filename == "":
            error = "Please select a crop image."
        elif not allowed_file(image.filename):
            error = "Upload JPG, JPEG, PNG, BMP or WEBP image."
        else:
            safe_name = secure_filename(image.filename)
            if not safe_name:
                error = "Invalid file name."
            else:
                unique_name = f"{uuid4().hex}_{safe_name}"
                image_path = os.path.join(UPLOAD_FOLDER, unique_name)
                try:
                    image.save(image_path)
                    result, confidence_value = predict_disease(image_path)
                    confidence = round(confidence_value * 100, 2)
                    guide = get_guide(result)
                    guide_en, guide_te = guide["en"], guide["te"]
                    image_name = unique_name
                except Exception as exc:
                    error = str(exc)

    return render_template_string(
        HTML,
        result=result,
        confidence=confidence,
        confidence_value=confidence_value * 100,
        crop=crop,
        image=image_name,
        error=error,
        guide_en=guide_en,
        guide_te=guide_te
    )

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
