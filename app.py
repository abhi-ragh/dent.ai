from flask import Flask, request, render_template, jsonify, redirect, url_for, session, make_response
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import json
import uuid
import requests
import io
from datetime import datetime
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
import fhirclient.models.patient as p
import fhirclient.models.observation as obs
import fhirclient.models.bundle as bundle
from fhirclient import client
from fhirclient.models.fhirreference import FHIRReference
from fhirclient.models.coding import Coding
from fhirclient.models.codeableconcept import CodeableConcept
from fhirclient.models.fhirdate import FHIRDate
from fhirclient.models.patient import Patient
from fhirclient.models.identifier import Identifier
from fhirclient.models.humanname import HumanName
from fhirclient.models.contactpoint import ContactPoint
from fhirclient.models.fhirdate import FHIRDate
from fhirclient.models.fhirdatetime import FHIRDateTime
import pdfkit
from datetime import datetime, timezone
import csv

PDFKIT_CONFIG = pdfkit.configuration(wkhtmltopdf='/usr/bin/wkhtmltopdf')

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.urandom(24)

# FHIR server configuration using SMART Health IT Sandbox
FHIR_SERVER_URL = "https://r4.smarthealthit.org"
smart_defaults = {
    'app_id': 'dental_diagnostic_app',
    'api_base': FHIR_SERVER_URL
}
smart = client.FHIRClient(settings=smart_defaults)

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

def search_patient(identifier):
    """Search for a patient in FHIR using identifier"""
    search = p.Patient.where(struct={'identifier': identifier})
    search_set = search.perform_resources(smart.server)
    return search_set[0] if search_set else None

def create_new_patient(data):
    """Create a new patient in FHIR server"""
    from fhirclient.models.patient import Patient
    from fhirclient.models.identifier import Identifier
    from fhirclient.models.humanname import HumanName
    from fhirclient.models.contactpoint import ContactPoint
    
    patient = Patient()
    
    # Create Identifier
    mrn = Identifier()
    mrn.system = "http://dental-diagnostic.org/mrn"
    mrn.value = str(uuid.uuid4())
    patient.identifier = [mrn]
    
    # Create HumanName
    name = HumanName()
    name.family = data['lastname']
    name.given = [data['firstname']]
    patient.name = [name]
    
    # Set birth date
    patient.birthDate = FHIRDate(data['birthdate'])
    
    # Set gender
    patient.gender = data['gender']
    
    # Create Telecom (contact points)
    phone = ContactPoint()
    phone.system = 'phone'
    phone.value = data['phone']
    
    email = ContactPoint()
    email.system = 'email'
    email.value = data.get('email', '')
    
    patient.telecom = [phone, email]
    
    # Save patient to FHIR server
    try:
        patient.create(smart.server)
        return patient
    except Exception as e:
        print(f"Error creating patient: {str(e)}")
        raise

def get_patient_history(patient_id):
    """Retrieve patient diagnostic history from FHIR"""
    search_url = f"{FHIR_SERVER_URL}/Observation?subject=Patient/{patient_id}&_sort=-date"
    try:
        response = requests.get(search_url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        bundle_data = response.json()
        history = []
        
        if 'entry' in bundle_data:
            for entry in bundle_data['entry']:
                if 'resource' in entry and entry['resource']['resourceType'] == 'Observation':
                    obs_data = entry['resource']
                    
                    # Extract the diagnosis data from the valueString if it exists
                    if 'valueString' in obs_data:
                        try:
                            diagnosis_data = json.loads(obs_data['valueString'])
                            
                            # Get date with fallback options
                            date = obs_data.get('effectiveDateTime', 
                                   obs_data.get('issued', 'Unknown date'))
                            
                            history_item = {
                                'id': obs_data['id'],
                                'date': date,
                                'code': obs_data.get('code', {}).get('text', 'Unknown observation'),
                                'diagnosis': diagnosis_data
                            }
                            history.append(history_item)
                        except json.JSONDecodeError:
                            print(f"Invalid JSON in observation {obs_data.get('id', 'unknown')}")
                            continue
        
        print(f"Retrieved {len(history)} history items for patient {patient_id}")
        return history
    except Exception as e:
        print(f"Error retrieving patient history: {str(e)}")
        return []

def save_diagnostic_result(patient_id, diagnosis_type, diagnosis_data, remarks=""):
    """Minimal Observation creation to satisfy FHIR server requirements."""
    from fhirclient.models.observation import Observation
    from fhirclient.models.fhirinstant import FHIRInstant
    from fhirclient.models.fhirdatetime import FHIRDateTime
    from fhirclient.models.fhirreference import FHIRReference
    from fhirclient.models.codeableconcept import CodeableConcept
    from fhirclient.models.coding import Coding

    observation = Observation()
    observation.status = "final"
    
    observation.subject = FHIRReference({"reference": f"Patient/{patient_id}"})
    
    observation.code = CodeableConcept({
        "coding": [{
            "system": "http://dental-diagnostic.org/observation-codes",
            "code": diagnosis_type,
            "display": "Dental X-Ray Diagnosis" if diagnosis_type == "xray_diagnosis" else "Oral Disease Diagnosis"
        }],
        "text": "AI Diagnostic Assessment"
    })
    
    # Store current timestamp in UTC
    current_time = datetime.now(timezone.utc).isoformat()
    observation.effectiveDateTime = FHIRDateTime(current_time)
    observation.issued = FHIRInstant(current_time)
    
    # Store the actual diagnosis data as JSON string
    observation.valueString = json.dumps(diagnosis_data)
    
    # Add remarks as a note if provided
    if remarks:
        observation.note = [{"text": remarks}]
    
    try:
        result = observation.create(smart.server)
        return result
    except Exception as e:
        print(f"Error saving diagnostic result: {str(e)}")
        return None

def get_gemini_diagnosis(diagnosis_type, condition_results, condition_accuracy, patient_history=None, remarks=None):
    """Get comprehensive diagnosis from Gemini API"""
    # Format patient history if available
    history_str = "No previous diagnostic history available"
    if patient_history and len(patient_history) > 0:
        history_items = []
        for i, item in enumerate(patient_history[:3]):  # Limit to 3 most recent
            date = item.get('date', 'Unknown date')
            diagnosis = item.get('diagnosis', {})
            if 'diagnosis' in diagnosis and len(diagnosis['diagnosis']) > 0:
                conditions = [d['condition'] for d in diagnosis['diagnosis']]
                history_items.append(f"- {date}: {', '.join(conditions)}")
        
        if history_items:
            history_str = "Previous diagnostic history:\n" + "\n".join(history_items)
    
    # Format practitioner remarks if available
    remarks_str = "No specific remarks provided"
    if remarks and remarks.strip():
        remarks_str = f"Practitioner remarks: {remarks}"
    
    # Set diagnosis type specific information
    if diagnosis_type == "xray_diagnosis":
        analysis_type = "X-ray analysis"
        diagnosis_context = f"X-ray results: {condition_results} (Accuracy: {condition_accuracy}%)"
    else:  # oral_diagnosis
        analysis_type = "Oral cavity examination"
        diagnosis_context = f"Oral disease results: {condition_results} (Accuracy: {condition_accuracy}%)"
    
    # Format the input for the API
    prompt = (f"Consider you are an expert dentist. Based on the following medical results:\n"
              f"{diagnosis_context}\n\n"
              f"Additional context:\n"
              f"{history_str}\n"
              f"{remarks_str}\n\n"
              f"You are a licensed dental specialist with 20 years of experience in dental diagnostics and treatment planning. "
              f"Based on the {analysis_type} showing: {condition_results} (Confidence level: {condition_accuracy}%)\n"
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
            response_text = response.text
            if "```json" in response_text:
                response_text = response_text.split("```json", 1)[1]
            if "```" in response_text:
                response_text = response_text.split("```", 1)[0]
            
            response_text = response_text.strip()
            diagnosis_data = json.loads(response_text)
            return diagnosis_data
        except json.JSONDecodeError as e:
            return {"error": f"Could not parse Gemini response as JSON: {str(e)}", "raw_response": response.text}
    except Exception as e:
        return {"error": f"Error calling Gemini API: {str(e)}"}

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/app')
def app_index():
    session.clear()
    return render_template('index.html')

@app.route('/app/patient_search', methods=['POST'])
def patient_search():
    if request.method == 'POST':
        search_term = request.form['search_term']
        
        # Search by ID, phone, or name using SMART sandbox
        search_url = f"{FHIR_SERVER_URL}/Patient?_format=json"
        
        # Try to search by identifier first
        response = requests.get(f"{search_url}&identifier={search_term}")
        
        if response.status_code == 200:
            bundle_data = response.json()
            
            if 'entry' in bundle_data and len(bundle_data['entry']) > 0:
                patients = []
                for entry in bundle_data['entry']:
                    if 'resource' in entry and entry['resource']['resourceType'] == 'Patient':
                        patient_data = entry['resource']
                        name = ""
                        if 'name' in patient_data and len(patient_data['name']) > 0:
                            name_data = patient_data['name'][0]
                            family = name_data.get('family', '')
                            given = ' '.join(name_data.get('given', []))
                            name = f"{given} {family}".strip()
                        birthdate = patient_data.get('birthDate', '')
                        identifier = ""
                        if 'identifier' in patient_data and len(patient_data['identifier']) > 0:
                            identifier = patient_data['identifier'][0].get('value', '')
                        patients.append({
                            'id': patient_data.get('id', ''),
                            'name': name,
                            'birthdate': birthdate,
                            'identifier': identifier
                        })
                return render_template('patient_results.html', patients=patients, search_term=search_term)
        
        # If not found by identifier, search by name
        response = requests.get(f"{search_url}&name={search_term}")
        if response.status_code == 200:
            bundle_data = response.json()
            if 'entry' in bundle_data and len(bundle_data['entry']) > 0:
                patients = []
                for entry in bundle_data['entry']:
                    if 'resource' in entry and entry['resource']['resourceType'] == 'Patient':
                        patient_data = entry['resource']
                        name = ""
                        if 'name' in patient_data and len(patient_data['name']) > 0:
                            name_data = patient_data['name'][0]
                            family = name_data.get('family', '')
                            given = ' '.join(name_data.get('given', []))
                            name = f"{given} {family}".strip()
                        birthdate = patient_data.get('birthDate', '')
                        identifier = ""
                        if 'identifier' in patient_data and len(patient_data['identifier']) > 0:
                            identifier = patient_data['identifier'][0].get('value', '')
                        patients.append({
                            'id': patient_data.get('id', ''),
                            'name': name,
                            'birthdate': birthdate,
                            'identifier': identifier
                        })
                return render_template('patient_results.html', patients=patients, search_term=search_term)
        
        # No results found
        return render_template('patient_results.html', patients=[], search_term=search_term)

@app.route('/app/patient/new', methods=['GET', 'POST'])
def new_patient():
    if request.method == 'POST':
        patient_data = {
            'firstname': request.form['firstname'],
            'lastname': request.form['lastname'],
            'birthdate': request.form['birthdate'],
            'gender': request.form['gender'],
            'phone': request.form['phone'],
            'email': request.form['email']
        }
        
        patient = create_new_patient(patient_data)
        session['patient_id'] = patient.id
        session['patient_name'] = f"{patient_data['firstname']} {patient_data['lastname']}"
        return redirect(url_for('diagnosis_options'))
    
    return render_template('new_patient.html')

@app.route('/app/patient/<patient_id>/select', methods=['GET'])
def select_patient(patient_id):
    patient_url = f"{FHIR_SERVER_URL}/Patient/{patient_id}"
    response = requests.get(patient_url)
    
    if response.status_code == 200:
        patient_data = response.json()
        name = "Unknown Patient"
        if 'name' in patient_data and len(patient_data['name']) > 0:
            name_data = patient_data['name'][0]
            family = name_data.get('family', '')
            given = ' '.join(name_data.get('given', []))
            name = f"{given} {family}".strip()
        session['patient_id'] = patient_id
        session['patient_name'] = name
        patient_history = get_patient_history(patient_id)
        session['patient_history'] = patient_history
        return redirect(url_for('diagnosis_options'))
    
    return redirect(url_for('app_index'))

@app.route('/app/diagnosis/options', methods=['GET'])
def diagnosis_options():
    if 'patient_id' not in session:
        return redirect(url_for('app_index'))
    patient_id = session['patient_id']
    patient_name = session['patient_name']
    return render_template('diagnosis_options.html', patient_id=patient_id, patient_name=patient_name)

@app.route('/app/diagnosis/xray', methods=['GET', 'POST'])
def xray_diagnosis():
    if 'patient_id' not in session:
        return redirect(url_for('app_index'))
    
    patient_id = session['patient_id']
    patient_name = session['patient_name']
    patient_history = session.get('patient_history', [])
    
    if request.method == 'POST':
        if 'xray_file' not in request.files:
            return render_template('xray_diagnosis.html', error="X-ray file is required", patient_name=patient_name)
        
        xray_file = request.files['xray_file']
        doctor_remarks = request.form.get('doctor_remarks', '')
        
        if xray_file.filename == '':
            return render_template('xray_diagnosis.html', error="No file selected", patient_name=patient_name)
        
        os.makedirs('static/uploads', exist_ok=True)
        xray_filepath = os.path.join('static/uploads', f"xray_{uuid.uuid4()}_{xray_file.filename}")
        xray_file.save(xray_filepath)
        xray_img_batch = prepare_image(xray_filepath, target_size=(512, 256))
        xray_predictions = dental_model.predict(xray_img_batch)
        xray_predicted_class_idx = np.argmax(xray_predictions, axis=1)[0]
        xray_predicted_class = dental_le.inverse_transform([xray_predicted_class_idx])[0]
        xray_confidence = xray_predictions[0][xray_predicted_class_idx] * 100
        
        diagnosis_data = get_gemini_diagnosis(
            "xray_diagnosis", 
            xray_predicted_class, 
            round(xray_confidence, 2),
            patient_history,
            doctor_remarks
        )
        
        save_diagnostic_result(patient_id, "xray_diagnosis", diagnosis_data, doctor_remarks)
        
        updated_history = get_patient_history(patient_id)
        session['patient_history'] = updated_history

        session['diagnosis_data'] = diagnosis_data
        session['diagnosis_type'] = "X-ray Analysis"
        session['condition_results'] = xray_predicted_class
        session['condition_accuracy'] = float(round(xray_confidence, 2))
        session['image_path'] = xray_filepath.replace('static/', '')
        session['doctor_remarks'] = doctor_remarks
        
        return redirect(url_for('diagnosis_result'))
    
    return render_template('xray_diagnosis.html', patient_name=patient_name)

@app.route('/app/diagnosis/oral', methods=['GET', 'POST'])
def oral_diagnosis():
    if 'patient_id' not in session:
        return redirect(url_for('app_index'))
    
    patient_id = session['patient_id']
    patient_name = session['patient_name']
    patient_history = session.get('patient_history', [])
    
    if request.method == 'POST':
        if 'oral_file' not in request.files:
            return render_template('oral_diagnosis.html', error="Oral cavity image is required", patient_name=patient_name)
        
        oral_file = request.files['oral_file']
        doctor_remarks = request.form.get('doctor_remarks', '')
        
        if oral_file.filename == '':
            return render_template('oral_diagnosis.html', error="No file selected", patient_name=patient_name)
        
        os.makedirs('static/uploads', exist_ok=True)
        oral_filepath = os.path.join('static/uploads', f"oral_{uuid.uuid4()}_{oral_file.filename}")
        oral_file.save(oral_filepath)
        oral_img_batch = prepare_image(oral_filepath, target_size=(224, 224))
        oral_predictions = oral_model.predict(oral_img_batch)
        oral_predicted_class_idx = np.argmax(oral_predictions, axis=1)[0]
        oral_predicted_class = oral_le.inverse_transform([oral_predicted_class_idx])[0]
        oral_confidence = oral_predictions[0][oral_predicted_class_idx] * 100
        
        diagnosis_data = get_gemini_diagnosis(
            "oral_diagnosis", 
            oral_predicted_class, 
            round(oral_confidence, 2),
            patient_history,
            doctor_remarks
        )
        
        save_diagnostic_result(patient_id, "oral_diagnosis", diagnosis_data, doctor_remarks)

        updated_history = get_patient_history(patient_id)
        session['patient_history'] = updated_history
        
        session['diagnosis_data'] = diagnosis_data
        session['diagnosis_type'] = "Oral Disease Analysis"
        session['condition_results'] = oral_predicted_class
        session['condition_accuracy'] = float(round(oral_confidence, 2))
        session['image_path'] = oral_filepath.replace('static/', '')
        session['doctor_remarks'] = doctor_remarks
        
        return redirect(url_for('diagnosis_result'))
    
    return render_template('oral_diagnosis.html', patient_name=patient_name)

@app.route('/app/diagnosis/result', methods=['GET'])
def diagnosis_result():
    if 'patient_id' not in session or 'diagnosis_data' not in session:
        return redirect(url_for('app_index'))
    
    patient_id = session['patient_id']
    patient_name = session['patient_name']
    diagnosis_data = session['diagnosis_data']
    diagnosis_type = session['diagnosis_type']
    condition_results = session['condition_results']
    condition_accuracy = session['condition_accuracy']
    image_path = session['image_path']
    doctor_remarks = session.get('doctor_remarks', '')
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    return render_template(
        'diagnosis_result.html',
        patient_id=patient_id,
        patient_name=patient_name,
        diagnosis_type=diagnosis_type,
        condition_results=condition_results,
        condition_accuracy=condition_accuracy,
        diagnosis_data=diagnosis_data,
        image_path=image_path,
        doctor_remarks=doctor_remarks,
        date=date
    )


@app.route('/app/diagnosis/result/pdf', methods=['GET'])
def generate_pdf():
    if 'patient_id' not in session or 'diagnosis_data' not in session:
        return redirect(url_for('app_index'))
    
    patient_id = session['patient_id']
    patient_name = session['patient_name']
    diagnosis_data = session['diagnosis_data']
    diagnosis_type = session['diagnosis_type']
    condition_results = session['condition_results']
    condition_accuracy = session['condition_accuracy']
    image_path = session['image_path']
    doctor_remarks = session.get('doctor_remarks', '')
    
    html = render_template(
        'pdf_template.html',
        patient_id=patient_id,
        patient_name=patient_name,
        diagnosis_type=diagnosis_type,
        condition_results=condition_results,
        condition_accuracy=condition_accuracy,
        diagnosis_data=diagnosis_data,
        image_path=image_path,
        doctor_remarks=doctor_remarks,
        date=datetime.now().strftime("%Y-%m-%d %H:%M")
    )
    
    try:
        pdf = pdfkit.from_string(html, False)
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=dental_diagnosis_{patient_id}_{datetime.now().strftime("%Y%m%d")}.pdf'
        return response
    except Exception as e:
        return f"Error generating PDF: {str(e)}"

@app.route('/app/patient/history', methods=['GET'])
def patient_history():
    if 'patient_id' not in session:
        return redirect(url_for('app_index'))
    
    patient_id = session['patient_id']
    patient_name = session['patient_name']
    history = get_patient_history(patient_id)
    
    return render_template('patient_history.html', patient_id=patient_id, patient_name=patient_name, history=history)

@app.route('/app/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    diagnosis_type = data.get('diagnosis_type', '').lower()

    # Choose the right CSV file based on diagnosis type
    if 'oral' in diagnosis_type:
        feedback_file = 'oral_feedback.csv'
    else:
        feedback_file = 'xrays_feedback.csv'

    # Check if file exists, if not write header row
    file_exists = os.path.exists(feedback_file)
    with open(feedback_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['patient_id', 'diagnosis_type', 'feedback', 'timestamp'])
        writer.writerow([
            data.get('patient_id'),
            data.get('diagnosis_type'),
            data.get('feedback'),
            data.get('timestamp')
        ])

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)