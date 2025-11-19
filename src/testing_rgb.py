import cv2
import os 
import numpy as np 
import pandas as pd 
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report 

# function to load and preprocess test images 

def preprocess_test_images(image_paths):
    images = []
    for image_path in image_paths:
        if os.path.exists(image_path):
           image = cv2.imread(image_path)
           image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
           image = cv2.resize (image , (448,448))
           image_array = np.array(image) / 255.00 # this will normalize the image to a value between 0 and 1 (this will simpify the calculations)
           images.append(image_array)
        else:
            print(f"image file {image_path} not found.")
    return np.array(images)
spreadsheet_path = 'Data/PatientsTestingData.xlsx'
spreadsheet = pd.read_excel(spreadsheet_path)

# Extract files paths from the spreadsheet 
test_image_paths = spreadsheet['Patient X-Ray File'].tolist()
test_images = preprocess_test_images(test_image_paths)

# load the trained model 

model = load_model('best_pneumonia_modelrgb.keras')
#model = load_model('87_85.keras') # this uses the image size of 224,224,3
#model = load_model('90_03.keras') # this uses the image size of 448,448,3

# make predictions 

predictions = model.predict(test_images)

# get the predicted classes (index of the highest probability)

predicted_classes = np.argmax(predictions , axis = 1 )

# print out the predictions 

for i , image_path in enumerate(test_image_paths):
    print(f"\Image: {image_path} , Predicted Class: {predicted_classes[i]}")
true_labels = spreadsheet['Pneumonia'].values
print(f"\ntrue outcome {spreadsheet['Pneumonia']}")

#calculate the accuracy 

accuracy = np.mean(predicted_classes == true_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
