from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Set the upload folder for storing images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = load_model('./model/saved_model/final_model.keras')

# Define your class names (update as per your model's classes)
CLASS_NAMES = ['lecture', 'no_lecture']


def prepare_image(image_path):
    # Match the model's expected input size
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize the image
    return img


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            img = prepare_image(file_path)
            prediction = model.predict(img)
            if prediction.size == 0:  # Check if prediction is empty
                return "Error: Model prediction is empty"
            predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
            confidence = np.max(prediction[0])
            return render_template('result.html', image_path=file_path, prediction=predicted_class, confidence=confidence)
    return render_template('index.html')


if __name__ == "__main__":
    # Ensure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
