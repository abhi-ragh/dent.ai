import os
import numpy as np
import tensorflow as tf
import csv

def create_dummy_model(filepath, input_shape, num_classes):
    """Creates and saves a simple dummy Keras model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(4, (3, 3), activation='relu'), # A simple conv layer
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax') # Output matches classes
    ])
    model.save(filepath)
    print(f"✅ Created dummy model: {filepath}")

def create_dummy_label_encoder(filepath, classes):
    """Creates and saves a dummy NumPy label encoder file."""
    np.save(filepath, np.array(classes, dtype=object))
    print(f"✅ Created dummy label encoder: {filepath}")

def create_dummy_csv(filepath, header):
    """Creates a CSV file with a header if it doesn't exist."""
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"✅ Created dummy CSV file: {filepath}")
    else:
        print(f"⏩ CSV file already exists: {filepath}")

def main():
    """Main function to create all required directories and files."""
    print("--- Starting Dummy File Creation ---")

    # 1. Define directory paths
    dirs_to_create = [
        'model',
        'xray_data/annotations',
        'oral-diseases/annotations',
        'static/uploads', # For image uploads
        'templates' # For Flask HTML templates
    ]

    # Create all directories
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        print(f"📁 Directory ensured: {d}")

    # 2. Create Dummy Keras Models
    # Note: The input shapes are common placeholders.
    # The number of classes matches the dummy label encoders below.
    create_dummy_model('model/dental_model.h5', input_shape=(256, 512, 3), num_classes=3)
    create_dummy_model('model/oral_diseases_model.h5', input_shape=(224, 224, 3), num_classes=4)

    # 3. Create Dummy Label Encoders
    dental_classes = ['Healthy', 'Caries', 'Periodontitis']
    oral_classes = ['Aphthous Ulcer', 'Leukoplakia', 'Gingivitis', 'Healthy']
    create_dummy_label_encoder('xray_data/annotations/label_encoder.npy', dental_classes)
    create_dummy_label_encoder('oral-diseases/annotations/oral_label_encoder.npy', oral_classes)
    
    # 4. Create Dummy CSV files for feedback
    xray_header = ['patient_id', 'diagnosis_type', 'feedback', 'timestamp']
    oral_header = ['patient_id', 'diagnosis_type', 'feedback', 'timestamp']
    create_dummy_csv('xrays_feedback.csv', xray_header)
    create_dummy_csv('oral_feedback.csv', oral_header)

    # 5. Create Dummy HTML Template Files to prevent TemplateNotFound errors
    templates = [
        'landing.html', 'index.html', 'patient_results.html', 'new_patient.html',
        'diagnosis_options.html', 'xray_diagnosis.html', 'oral_diagnosis.html',
        'diagnosis_result.html', 'patient_history.html', 'pdf_template.html'
    ]
    for t in templates:
        path = os.path.join('templates', t)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write(f"<!DOCTYPE html><html><head><title>{t}</title></head><body><h1>Placeholder for {t}</h1></body></html>")
            print(f"✅ Created dummy template: {path}")

    print("\n--- All dummy files and directories created successfully! ---")

if __name__ == '__main__':
    main()
