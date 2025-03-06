import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np

# Define paths
dataset_path = 'oral-diseases'
annotations_path = 'oral-diseases/annotations/oral_annotations.csv'
feedback_path = 'oral-feedback.csv'  # New feedback CSV for oral images

# Create or load annotations CSV
if not os.path.exists(annotations_path):
    # Initialize lists
    filenames = []
    classes = []
    
    # Iterate through directories
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    filenames.append(os.path.join(class_name, img_name))
                    classes.append(class_name)
    
    # Create DataFrame
    data = pd.DataFrame({
        'filename': filenames,
        'class': classes
    })
    
    # Save to CSV
    data.to_csv(annotations_path, index=False)
else:
    # Load existing annotations
    data = pd.read_csv(annotations_path)

# --- Integrate Feedback Data ---
# Check if feedback CSV exists and append positively rated examples
if os.path.exists(feedback_path):
    feedback_df = pd.read_csv(feedback_path)
    # Assuming feedback CSV has columns: 'image_filename', 'predicted_label', 'feedback'
    positive_feedback = feedback_df[feedback_df['feedback'] == 'up']
    if not positive_feedback.empty:
        feedback_data = pd.DataFrame({
            'filename': positive_feedback['image_filename'],
            'class': positive_feedback['predicted_label']
        })
        # Remove duplicates if necessary
        data = pd.concat([data, feedback_data], ignore_index=True)
# --- End Feedback Integration ---

# Encode class labels
le = LabelEncoder()
data['class_encoded'] = le.fit_transform(data['class'])

# Split data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['class_encoded'])

# Define image dimensions
img_width, img_height = 224, 224  # Using a smaller size for efficiency

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_data,
    directory=dataset_path,
    x_col='filename',
    y_col='class',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    val_data,
    directory=dataset_path,
    x_col='filename',
    y_col='class',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

# Define model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# Save the model and label encoder
model.save('model/oral_diseases_model.h5')
np.save('oral-diseases/annotations/oral_label_encoder.npy', le.classes_)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
