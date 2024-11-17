import os 
import numpy as np 
import pandas as pd 
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report 

# function to load and preprocess test images 
def preprocess_test_images (image_paths , target_size = (448,448)):
    images = []
    for image_path in image_paths: 
        if os.path.exists(image_path):
            # if the path exists continue and preprocess the image otherwise print the image does not exist 
            image = Image.open(image_path).convert('L') # this is a function that converts the image into greyscale 
            image = resize_and_pad(image, target_size) # this is a function that will resize the images to the defined image size of 448x448 
            image_array = np.array(image) / 255.00 # greyscale defaults to a number between 0 and 255 so by dividing by 255 the value is normalized
            images.append(image_array)
        else: 
            print(f"Image file {image_path} not found.")
    return np.array(images)

# This is the function to convert all the images to 448x448 while still retaining the orginal aspect ratio

def resize_and_pad(image, target_size):
    target_width , target_height = target_size 
    width, height = image.size 
    scale = min(target_width / width , target_height / height)
    new_width = int(width * scale)
    new_height = int(height*scale)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    new_image = Image.new('L', target_size)
    new_image.paste(resized_image, ((target_width - new_width) // 2 , (target_height - new_height) // 2))
    return new_image

spreadsheet_path = 'Data/PatientsTestingData.xlsx'
spreadsheet = pd.read_excel(spreadsheet_path)

# Extract files paths from the spreadsheet 
test_image_paths = spreadsheet['Patient X-Ray File'].tolist()
test_images = preprocess_test_images(test_image_paths)

# Reshape the images to add a channel dimension for the greyscale images 

test_images = test_images.reshape(-1 , 448 , 448 , 1)

# load the trained model 

model = load_model('best_pneumonia_model.keras')

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
