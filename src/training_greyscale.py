import os # this is the libary to locate local files from the computer 
import imageio # this is the libary that is used to read images 
import numpy as np # this is the libary for numerical operations 
import pandas as pd # this is the libary for data manipulation 
import tensorflow as tf # this is the libary for deep learning (machine learning/ neural networks)
from tensorflow.keras.models import Model , Sequential # these are models that will be called on later in the code
from tensorflow.keras.layers import Dense, Flatten , Conv2D, MaxPooling2D , Input , Dropout , BatchNormalization, Activation # these are the input , hidden and output layers within the CNN model 
from tensorflow.keras.optimizers import adam # this is the optimization function 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint , ReduceLRonPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from sklearn.model_selection import train_test_split # this is the library that is used to split data into a training and testing subset 
from sklearn.preprocessing import LabelEncoder # this is the function to encode labels 
from sklearn.metrics import classification_report # this is the function that is used to generate a classification report 
from PIL import Image # this is the libary that is used to handle image operations 

# DIR is the directory that contains all the images 

DIR ="images"

# Load the spreadsheet from the local file titled "Data/PatientsTrainingData.xlsx"

file_path = 'Data/PatientsTrainingData.xlsx'
spreadsheet = pd.read_excel(file_path) # this function reads the excel file into the pandas dataframe 

# Function to preprocess the images 

def preprocess_data(spreadsheet, dir_path, num_records=5200 , target_size= (440,440)):
    images = []
    pneumonias = []
    
    for index, row in spreadsheet.iterrows():
        if index >= num_records:
            break

        xray_file = row['Patient X-Ray File']
        pneumonia = row['Pneumonia']

        # Load the X-ray Image 

        image_path = os.path.join(dir_path , xray_file)
        if os.path.exists(image_path):
            # Read and preprocess the Data
            image = Image.open(image_path).convert('L') # this is the function that opens the image and coverts the image to greyscale 
            image = pad_image_to_target(image, target_size) # this is to pad the image to the target size 
            image_array = np.array(image) / 255.00 # this will normalize the image to a value between 0 and 1 (this will simpify the calculations)
            images.append(image_array)
            pneumonias.append(pneumonia)
        else:
            print(f"image file {image_path} not found.")
    return np.array(images) , np.array(pneumonias)

# this is the function to make the image into the desired size 448x448 

def pad_image_to_target(image, target_size):
    target_width , target_height = target_size 
    width, height = image.size 
    scale = min(target_width / width , target_height / height)
    new_width = int(width * scale)
    new_height = int(height*scale)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    new_image = Image.new('L', target_size)
    new_image.paste(resized_image, ((target_width - new_width) // 2 , (target_height - new_height) // 2))
    return new_image

# Preprocess the Data

images , pneumonias = preprocess_data(spreadsheet, DIR , num_records= 5200 , target_size=(448,448))
images=images.reshape(-1 , 448 , 448 , 1) # reshape the images to add a channel dimension for the greyscale images 

# Encode the Labels 


