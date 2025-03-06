from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the dental model
dental_model = load_model('model/dental_model.h5')

# Load the oral diseases model
oral_model = load_model('model/oral_diseases_model.h5')

# Load the label encoders
dental_label_encoder_path = 'xray_data/annotations/label_encoder.npy'
oral_label_encoder_path = 'oral-diseases/annotations/oral_label_encoder.npy'

dental_le = LabelEncoder()
dental_le.classes_ = np.load(dental_label_encoder_path, allow_pickle=True)

oral_le = LabelEncoder()
oral_le.classes_ = np.load(oral_label_encoder_path, allow_pickle=True)

def prepare_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert('RGB').resize(target_size)
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_dental', methods=['POST'])
def upload_dental():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Save the uploaded image
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)
        
        # Prepare the image for dental model
        img_batch = prepare_image(filepath, target_size=(512, 256))
        
        # Make prediction for dental model
        dental_predictions = dental_model.predict(img_batch)
        dental_predicted_class_idx = np.argmax(dental_predictions, axis=1)[0]
        dental_predicted_class = dental_le.inverse_transform([dental_predicted_class_idx])[0]
        dental_confidence = dental_predictions[0][dental_predicted_class_idx] * 100
        
        dental_result = {
            'predicted_class': dental_predicted_class,
            'confidence': round(dental_confidence, 2)
        }
        
        return render_template('index.html', dental_result=dental_result)
    return 'File upload failed'

@app.route('/upload_oral', methods=['POST'])
def upload_oral():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Save the uploaded image
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)
        
        # Prepare the image for oral diseases model
        img_batch = prepare_image(filepath, target_size=(224, 224))
        
        # Make prediction for oral diseases model
        oral_predictions = oral_model.predict(img_batch)
        oral_predicted_class_idx = np.argmax(oral_predictions, axis=1)[0]
        oral_predicted_class = oral_le.inverse_transform([oral_predicted_class_idx])[0]
        oral_confidence = oral_predictions[0][oral_predicted_class_idx] * 100
        
        oral_result = {
            'predicted_class': oral_predicted_class,
            'confidence': round(oral_confidence, 2)
        }
        
        return render_template('index.html', oral_result=oral_result)
    return 'File upload failed'

if __name__ == '__main__':
    app.run(debug=True)
