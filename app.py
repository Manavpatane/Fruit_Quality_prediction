from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np
import os
import base64
import uuid
from tensorflow.keras.preprocessing import image as keras_image

app = Flask(__name__)

# Folder paths
MODEL_FOLDER = "models"
LABEL_FOLDER = "labels"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed model names
VALID_MODELS = ['banana', 'apple', 'carrot', 'guava', 'lime', 'orange', 'pomegranate', 'potato', 'tomato']

# Prediction logic
def load_and_predict(model_name, uploaded_file):
    try:
        # Paths for model and labels
        model_path = os.path.join(MODEL_FOLDER, f"{model_name}_keras_model.h5")
        label_path = os.path.join(LABEL_FOLDER, f"{model_name}.txt")

        # Load model and class labels
        model = load_model(model_path, compile=False)
        class_names = [line.strip() for line in open(label_path, "r").readlines()]

        # Save uploaded image
        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, filename)

        if isinstance(uploaded_file, np.ndarray):  # From webcam
            cv2.imwrite(save_path, uploaded_file)
        else:  # From file upload
            uploaded_file.save(save_path)

        # Preprocess image
        img = keras_image.load_img(save_path, target_size=(150, 150))
        img_array = keras_image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        index = np.argmax(prediction)
        confidence = float(prediction[0][index]) * 100
        predicted_class = class_names[index]

        return predicted_class, confidence, filename

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return "Error during prediction", 0.0, None

# Home route
@app.route('/')
def index():
    return render_template('index.html', models=VALID_MODELS)

# Prediction route
@app.route('/predict/<model_name>', methods=['GET', 'POST'])
def predict(model_name):
    # Sanitize model name
    model_name = model_name.strip().lower().replace(".", "")

    if model_name not in VALID_MODELS:
        return f"Invalid model name: {model_name}", 400

    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        try:
            webcam_data = request.form.get('webcam_image')

            if webcam_data:
                # Decode base64 image from webcam
                header, encoded = webcam_data.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                img_array = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                prediction, confidence, image_path = load_and_predict(model_name, img)

            elif 'image' in request.files:
                file = request.files['image']
                if file and file.filename != '':
                    prediction, confidence, image_path = load_and_predict(model_name, file)
        except Exception as e:
            print(f"[ERROR] During POST processing: {e}")
            prediction = "Error during prediction"
            confidence = 0.0
            image_path = None

    return render_template('predict.html',
                           model=model_name,
                           prediction=prediction,
                           confidence=confidence,
                           image_path=image_path)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
