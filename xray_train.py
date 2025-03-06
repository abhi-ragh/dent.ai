import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

# Load annotations
annotations_path = 'xray_data/annotations/train_annotations.csv'
data = pd.read_csv(annotations_path)

# Encode class labels
le = LabelEncoder()
data['class_encoded'] = le.fit_transform(data['class'])

# Split data into training and validation sets (but we'll ignore validation)
train_data, _ = train_test_split(data, test_size=0.2, random_state=42, stratify=data['class_encoded'])

# Define image dimensions
img_width, img_height = 512, 256

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_data,
    directory='xray_data/train',
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

# Train the model without validation
model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=len(train_generator)
)

# Save the model
model.save('model/dental_model.h5')

# Save the label encoder
np.save('xray_data/annotations/label_encoder.npy', le.classes_)
