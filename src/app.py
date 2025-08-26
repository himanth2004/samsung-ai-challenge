from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import re
import torch
import joblib
from model_definition import MouseDynamicsClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Templates moved to ai-challenge/docs
TEMPLATES_DIR = os.path.join(os.path.dirname(BASE_DIR), "docs")

app = Flask(__name__, template_folder=TEMPLATES_DIR)
CORS(app)

try:
    print("Loading PyTorch behavior model...")
    candidate_behavior_paths = [
        os.environ.get("BEHAVIOR_MODEL_PATH"),
        os.path.join(BASE_DIR, "behavior_model.pth"),
        os.path.join(os.getcwd(), "behavior_model.pth"),
        os.path.join(os.path.dirname(BASE_DIR), "behavior_model.pth"),
    ]
    behavior_model_path = next((p for p in candidate_behavior_paths if p and os.path.exists(p)), None)
    print(f"Behavior model candidates: {candidate_behavior_paths}")
    print(f"Behavior model resolved path: {behavior_model_path}")

    if behavior_model_path:
        behavior_model = MouseDynamicsClassifier()
        behavior_model.load_state_dict(torch.load(behavior_model_path, map_location=torch.device("cpu")))
        behavior_model.eval()
        print("✅ PyTorch behavior model loaded successfully.")
    else:
        raise FileNotFoundError("behavior_model.pth not found in candidate paths")
except Exception as e:
    print(f"❌ Error loading PyTorch model: {e}")
    behavior_model = None

typing_pipeline = None
try:
    print("Loading typing pipeline...")
    candidate_pipeline_paths = [
        os.environ.get("TYPING_PIPELINE_PATH"),
        os.path.join(BASE_DIR, "typing_pipeline.pkl"),
        os.path.join(os.getcwd(), "typing_pipeline.pkl"),
        os.path.join(os.path.dirname(BASE_DIR), "typing_pipeline.pkl"),
    ]
    typing_pipeline_path = next((p for p in candidate_pipeline_paths if p and os.path.exists(p)), None)
    print(f"Typing pipeline candidates: {candidate_pipeline_paths}")
    print(f"Typing pipeline resolved path: {typing_pipeline_path}")
    if typing_pipeline_path:
        typing_pipeline = joblib.load(typing_pipeline_path)
        print("✅ Typing pipeline loaded successfully.")
    else:
        raise FileNotFoundError("typing_pipeline.pkl not found in candidate paths")
except Exception as e:
    print(f"❌ Error loading typing pipeline: {e}")
    typing_pipeline = None

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    behavior_features = data.get("behavior")
    print(f"\n--- Prediction Request Received ---")                # Printed in terminal
    print(f"Input data: {behavior_features}")                     # Printed in terminal
    print("Processing with behavior model...")                    # Printed in terminal

    if not behavior_model:
        print("❌ No behavior model loaded.")
        return jsonify({"prediction": "No behavior model loaded."})

    if not behavior_features or not isinstance(behavior_features, list):
        print("❌ Invalid or missing behavior data.")
        return jsonify({"prediction": "Invalid or missing behavior data."})

    try:
        coords = behavior_features
        # Use only the latest mouse coordinate (single point)
        if len(coords) > 1:
            coords = [coords[-1]]
        behavior_tensor = torch.tensor([coords], dtype=torch.float32)
        with torch.no_grad():
            output = behavior_model(behavior_tensor)
            pred_behavior = torch.argmax(output, dim=1).item()
        print(f"✅ Behavior model output: {pred_behavior}")        # Printed in terminal
        return jsonify({"prediction": pred_behavior})
    except Exception as e:
        print(f"❌ Error during prediction: {e}")                  # Printed in terminal
        return jsonify({"prediction": "Error during prediction."})

@app.route("/predict_typing", methods=["POST"])
def predict_typing():
    data = request.json
    text_input = data.get("text")
    print(f"\n--- Typing Prediction Request Received ---")          # Printed in terminal
    print(f"Input text: {text_input}")                            # Printed in terminal
    print("Processing with typing pipeline...")                      # Printed in terminal

    if typing_pipeline is None:
        print("❌ No typing pipeline loaded.")
        return jsonify({"prediction": "No typing pipeline loaded."})

    if not text_input or not isinstance(text_input, str):
        print("❌ Invalid or missing text data.")
        return jsonify({"prediction": "Invalid or missing text data."})

    try:
        # Text preprocessing (same as training)
        replacements = {
            "he's": "he is", "she's": "she is", "it's": "it is", "I'm": "I am",
            "you're": "you are", "we're": "we are", "they're": "they are",
            "he'll": "he will", "she'll": "she will", "it'll": "it will",
            "i'll": "i will", "you'll": "you will", "we'll": "we will",
            "they'll": "they will", "he'd": "he would", "she'd": "she would",
            "it'd": "it would", "i'd": "i would", "you'd": "you would",
            "we'd": "we would", "they'd": "they would", "haven't": "have not",
            "hasn't": "has not", "hadn't": "had not", "don't": "do not",
            "doesn't": "does not", "didn't": "did not", "can't": "cannot",
            "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
            "mightn't": "might not", "mustn't": "must not", "aren't": "are not",
            "isn't": "is not", "wasn't": "was not", "weren't": "were not",
            "im": "i am", "u": "you"
        }
        
        def clean_text(text):
            text = text.lower()
            for k, v in replacements.items():
                text = text.replace(k, v)
            # Remove non-alphabetic characters
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            return text.strip()
        
        # Clean the input text
        cleaned_text = clean_text(text_input)
        print(f"Cleaned text: {cleaned_text}")                     # Printed in terminal
        
        # Use the saved sklearn pipeline directly
        prediction = typing_pipeline.predict([cleaned_text])[0]
        print(f"✅ Typing pipeline output: {prediction}")             # Printed in terminal
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        print(f"❌ Error during typing prediction: {e}")           # Printed in terminal
        import traceback
        traceback.print_exc()  # Print full error traceback
        return jsonify({"prediction": "Error during typing prediction."})



@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "behavior_model_loaded": behavior_model is not None,
        "typing_pipeline_loaded": typing_pipeline is not None,
        "base_dir": BASE_DIR,
    })

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/about.html', methods=["GET"])
def aboutpage():
    return render_template('about.html')

@app.route('/main.html', methods=["GET"])
def mainpage():
    return render_template('main.html')

@app.route('/index.html', methods=["GET"])
def homepage():
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)