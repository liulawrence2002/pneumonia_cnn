# This section imports necessary libraries for file operations (os), data handling (numpy, pandas), model building and training (tensorflow and keras), image processing (ImageDataGenerator, PIL), and evaluation (LabelEncoder, classification_report).
# See explaination of the following tensorflow and scikt-learn libs at the bottom of the code
#   Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, ReLU, Input,
#   Model, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, ImageDataGenerator, Adam, train_test_split, LabelEncoder, and classification_report
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, ReLU, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from PIL import Image
import cv2


# Directory containing the images
DIR = "images"
file_path = 'Data/PatientsTrainingData.xlsx'  # Path to the spreadsheet containing labels
spreadsheet = pd.read_excel(file_path)

# Function to preprocess images and convert grayscale to RGB
def preprocess_data(spreadsheet, dir_path, num_records=5200, target_size=(224, 224)):
    images = []
    labels = []

    for index, row in spreadsheet.iterrows():
        if index >= num_records:
            break

        xray_file = row['Patient X-Ray File']
        label = row['Pneumonia']

        # Load and preprocess the X-ray image
        image_path = os.path.join(dir_path, xray_file)
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = pad_image_to_target(image, target_size)  # Pad and resize image to target size
            image = np.stack((image,) * 3, axis=-1) / 255.0  # Convert grayscale to RGB by duplicating channels
            images.append(image)
            labels.append(label)
        else:
            print(f"Image file {image_path} not found.")

    return np.array(images), np.array(labels)

# Function to pad and resize the image
def pad_image_to_target(image, target_size):
    target_width, target_height = target_size
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    new_image = Image.new('L', target_size)
    new_image.paste(resized_image, ((target_width - new_width) // 2, (target_height - new_height) // 2))
    return new_image

# Preprocess the data
images, labels = preprocess_data(spreadsheet, DIR, num_records=5200, target_size=(224, 224))

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(np.unique(encoded_labels))

# Split the data into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    images, encoded_labels, test_size=0.15, random_state=42, stratify=encoded_labels
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Build the model based on Table 1 from the paper
def build_model(input_shape=(224, 224, 3), num_classes=3):  # Change num_classes to 3
    inputs = Input(shape=input_shape)

    # First Convolutional Block
    x = Conv2D(32, (6, 6), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = Conv2D(32, (6, 6), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = MaxPooling2D((2, 2))(x)

    # Second Convolutional Block
    x = Conv2D(64, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = Conv2D(64, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = MaxPooling2D((2, 2))(x)

    # Third Convolutional Block
    x = Conv2D(128, (4, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = Conv2D(128, (4, 4), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = MaxPooling2D((3, 3))(x)

    # Fourth Convolutional Block
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = MaxPooling2D((2, 2))(x)

    # Fifth Convolutional Block
    x = Conv2D(1024, (2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = Conv2D(512, (2, 2), padding='same')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = MaxPooling2D((2, 2))(x)

    # Flatten and Fully Connected Layers
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Dense(1024)(x)
    x = Dense(512)(x)
    x = Dense(512)(x)
    x = Dense(256)(x)
    x = Dense(64)(x)

    # Output Layer for 3 classes
    outputs = Dense(num_classes, activation='softmax')(x)  # Set to 3 classes
    model = Model(inputs, outputs)
    return model

# Compile the model
model = build_model()
model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks for training
checkpoint = ModelCheckpoint('best_pneumonia_model.keras', save_best_only=True, monitor='val_accuracy')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=5, cooldown=5)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Load the best model
best_model = tf.keras.models.load_model('best_pneumonia_model.keras')
predictions = best_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Classification report
target_names = label_encoder.inverse_transform(np.unique(encoded_labels)).astype(str)
print(classification_report(y_test, predicted_classes, target_names=target_names))