from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
import json

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

# Configure Gemini API
genai.configure(api_key="AIzaSyDNqtEYmu9moev-6nZTjrBNW4mwoXXP_RA")
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

def prepare_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert('RGB').resize(target_size)
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def get_gemini_diagnosis(xray_results, xray_accuracy, oral_disease_results, oral_disease_accuracy):
    # Default values for patient history and symptoms if not provided
    patient_history = "No specific history provided"
    patient_symptoms = "No specific symptoms reported"
    
    # Format the input for the API
    prompt = (f"Consider you are an expert dentist. Based on the following medical results:\n"
              f"X-ray results: {xray_results} (Accuracy: {xray_accuracy}%)\n"
              f"Oral disease results: {oral_disease_results} (Accuracy: {oral_disease_accuracy}%)\n\n"
              f"You are a licensed dental specialist with 20 years of experience in dental diagnostics and treatment planning. "
              f"Based on the following patient diagnostic results:\n"
              f"- X-ray findings: {xray_results} (Confidence level: {xray_accuracy}%)\n"
              f"- Oral examination results: {oral_disease_results} (Confidence level: {oral_disease_accuracy}%)\n"
              f"- Patient history summary: {patient_history}\n"
              f"- Patient reported symptoms: {patient_symptoms}\n"
              f"Please analyze these results comprehensively and provide:\n"
              f"1. A detailed professional assessment\n"
              f"2. A clear diagnosis with differentiation between primary and secondary issues\n"
              f"3. A personalized treatment plan with prioritized interventions\n"
              f"4. Preventative recommendations\n"
              f"Format your response as a structured JSON object with the following schema:\n"
              f"{{\n"
              f"  \"clinical_findings\": [\n"
              f"    {{\"observation\": \"string\", \"significance\": \"string\", \"confidence\": \"number\"}}\n"
              f"  ],\n"
              f"  \"diagnosis\": [\n"
              f"    {{\"condition\": \"string\", \"severity\": \"string\", \"certainty\": \"string\", \"supporting_evidence\": [\"string\"]}}\n"
              f"  ],\n"
              f"  \"treatment_plan\": [\n"
              f"    {{\"intervention\": \"string\", \"priority\": \"string\", \"timeline\": \"string\", \"expected_outcome\": \"string\"}}\n"
              f"  ],\n"
              f"  \"preventative_measures\": [\n"
              f"    {{\"recommendation\": \"string\", \"rationale\": \"string\"}}\n"
              f"  ],\n"
              f"  \"follow_up\": {{\n"
              f"    \"timeline\": \"string\",\n"
              f"    \"specific_monitoring\": [\"string\"]\n"
              f"  }}\n"
              f"}}\n"
              f"Include at least 3-5 items in each list category. Base your assessment only on scientifically validated dental practices and current clinical guidelines.")

    # Generate content using the Gemini API
    try:
        response = gemini_model.generate_content(prompt)
        # Try to parse the response as JSON
        try:
            # Clean up the response text to ensure it's valid JSON
            # Sometimes AI models add markdown code block syntax
            response_text = response.text
            if "```json" in response_text:
                response_text = response_text.split("```json", 1)[1]
            if "```" in response_text:
                response_text = response_text.split("```", 1)[0]
            
            # Remove any extra characters before/after the JSON content
            response_text = response_text.strip()
            
            # Parse the JSON
            diagnosis_data = json.loads(response_text)
            return diagnosis_data
        except json.JSONDecodeError as e:
            return {"error": f"Could not parse Gemini response as JSON: {str(e)}", "raw_response": response.text}
    except Exception as e:
        return {"error": f"Error calling Gemini API: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'dental_file' not in request.files or 'oral_file' not in request.files:
        return 'Both dental X-ray and oral disease files are required'
    
    dental_file = request.files['dental_file']
    oral_file = request.files['oral_file']
    
    if dental_file.filename == '' or oral_file.filename == '':
        return 'Both files must be selected'
    
    # Create uploads directory if it doesn't exist
    os.makedirs('static/uploads', exist_ok=True)
    
    # Process dental X-ray
    dental_filepath = os.path.join('static/uploads', dental_file.filename)
    dental_file.save(dental_filepath)
    dental_img_batch = prepare_image(dental_filepath, target_size=(512, 256))
    dental_predictions = dental_model.predict(dental_img_batch)
    dental_predicted_class_idx = np.argmax(dental_predictions, axis=1)[0]
    dental_predicted_class = dental_le.inverse_transform([dental_predicted_class_idx])[0]
    dental_confidence = dental_predictions[0][dental_predicted_class_idx] * 100
    
    # Process oral disease
    oral_filepath = os.path.join('static/uploads', oral_file.filename)
    oral_file.save(oral_filepath)
    oral_img_batch = prepare_image(oral_filepath, target_size=(224, 224))
    oral_predictions = oral_model.predict(oral_img_batch)
    oral_predicted_class_idx = np.argmax(oral_predictions, axis=1)[0]
    oral_predicted_class = oral_le.inverse_transform([oral_predicted_class_idx])[0]
    oral_confidence = oral_predictions[0][oral_predicted_class_idx] * 100
    
    # Get comprehensive diagnosis from Gemini
    diagnosis_data = get_gemini_diagnosis(
        dental_predicted_class, 
        round(dental_confidence, 2),
        oral_predicted_class, 
        round(oral_confidence, 2)
    )
    
    # Render results
    return render_template(
        'result.html',
        dental_predicted_class=dental_predicted_class,
        dental_confidence=round(dental_confidence, 2),
        oral_predicted_class=oral_predicted_class,
        oral_confidence=round(oral_confidence, 2),
        diagnosis_data=diagnosis_data,
        dental_image=dental_filepath.replace('static/', ''),
        oral_image=oral_filepath.replace('static/', '')
    )

if __name__ == '__main__':
    app.run(debug=True)