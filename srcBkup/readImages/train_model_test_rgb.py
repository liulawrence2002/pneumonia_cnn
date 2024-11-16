import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

# Function to load and preprocess test images
def preprocess_test_images(image_paths, target_size=(224, 224)):
    images = []
    for image_path in image_paths:
        if os.path.exists(image_path):
            # Read and preprocess the image
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = resize_and_pad(image, target_size)   # Resize and pad to target size
            # image = np.array(image) / 255.0              # Normalize to [0, 1]
            # image_rgb = np.stack((image,) * 3, axis=-1)  # Convert grayscale to RGB
            image_rgb = np.stack((image,) * 3, axis=-1) / 255.0  # Convert grayscale to RGB by duplicating channels

            images.append(image_rgb)
        else:
            print(f"Image file {image_path} not found.")
    return np.array(images)

# Function to resize and pad images while maintaining aspect ratio
def resize_and_pad(image, target_size):
    target_width, target_height = target_size
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    new_image = Image.new('L', target_size)
    new_image.paste(resized_image, ((target_width - new_width) // 2, (target_height - new_height) // 2))
    return new_image

# Load the spreadsheet with true labels
spreadsheet_path = 'Data/PatientsTestingData.xlsx'
spreadsheet = pd.read_excel(spreadsheet_path)

# Extract file paths and true labels
test_image_paths = spreadsheet['Patient X-Ray File'].tolist()
true_labels = spreadsheet['Pneumonia'].values  

# Load the trained model
model = load_model('best_pneumonia_model.keras')

# Preprocess test images to match training conditions
test_images = preprocess_test_images(test_image_paths)

# Make predictions on the RGB images
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# Print out predictions for each image
for i, image_path in enumerate(test_image_paths):
    print(f"Image: {image_path}, Predicted Class: {predicted_classes[i]}")

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_labels)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Print classification report
# print("\nClassification Report:")
# print(classification_report(true_labels, predicted_classes))

