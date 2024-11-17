
import os # this is the libary to locate local files from the computer 
import imageio # this is the libary that is used to read images 
import numpy as np # this is the libary for numerical operations 
import pandas as pd # this is the libary for data manipulation 
import tensorflow as tf # this is the libary for deep learning (machine learning/ neural networks)
from tensorflow.keras.models import Model , Sequential # these are models that will be called on later in the code
from tensorflow.keras.layers import Dense, Flatten , Conv2D, MaxPooling2D , Input , Dropout , BatchNormalization, Activation # these are the input , hidden and output layers within the CNN model 
from tensorflow.keras.optimizers import Adam # this is the optimization function 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint , ReduceLROnPlateau
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

def preprocess_data(spreadsheet, dir_path, num_records=5200 , target_size= (256,256)):
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

# this is the function to make the image into the desired size 256x256 

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

images , pneumonias = preprocess_data(spreadsheet, DIR , num_records= 5200 , target_size=(256,256))
images=images.reshape(-1 , 256 , 256 , 1) # reshape the images to add a channel dimension for the greyscale images 

# Encode the Labels 
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(pneumonias) # this is a function that converts categorical labels into numerical values 
num_classes = len(np.unique(encoded_labels)) # this will ge tth enumber of unique classes 

# split the data unto training, validation, and testing sets 

X_train_val, X_test, y_train_val , y_test = train_test_split(images, encoded_labels, test_size = 0.15 , random_state = 42 , stratify = encoded_labels)
X_train, X_val , y_train , y_val = train_test_split ( X_train_val , y_train_val , test_size = 0.176 , random_state = 42 , stratify = y_train_val )

# Data augmentation 
# In order to train this CNN, novel images with slight modifcations are generated from the existing images to be used for training 
# The slight changes include rotation, horizontal shift , vertical shift , and zoom 

datagen = ImageDataGenerator (
    rotation_range = 10, # this will randomly rotate the images between 0 and 10 degrees 
    width_shift_range = 0.1, # this will randomly translate the images horizontally 
    height_shift_range = 0.1, # this will randomly translate the images vertically 
    zoom_range = 0.1, # this will randomaly zoom into the images 
    horizontal_flip = True, # this will flip the images horizontally 
    vertical_flip = False # this will flip the images vertically 
)
datagen.fit(X_train) # this will compute the data augmentation onto the the training set 

# define the model 
image_input = Input(shape= (256 , 256 , 1) , name = 'image') # this is the image layer 

# first convolutional layer with 32 filers with filter size of 1x1 

x = Conv2D(32 , (3 , 3))(image_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2 , 2))(x)
x = Dropout(0.2)(x)

# second convolutional layer with 64 filers with filter size of 1x1 

x = Conv2D(64 , (3 , 3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2 , 2))(x)
x = Dropout(0.2)(x)

x = Conv2D(128 , (2 , 2))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2 , 2) , strides = 2)(x)
x = Dropout(0.4)(x)



# Flatten and fully connect the layers 

x = Flatten()(x)
x = Dense(512)(x)
x = BatchNormalization()(x)
x = Dense(128)(x)
output = Dense(num_classes , activation = 'softmax')(x)

# create the model 

model = Model(inputs = image_input , outputs = output)
model.summary()

# compile the model 

model.compile(optimizer=Adam(learning_rate = 1e-4) , loss= 'sparse_categorical_crossentropy' , metrics = ['accuracy'])

# Callbacks for training -- Setting up the necessary callback functions 
# ModelCheckpoint will save the best model based on validation accuracy 
# Early Stopping will stop the training early if validation accuracy stops improving 
# ReduceLRonPlateau will reduce the leaning rate if validation accuracy plateaus 

checkpoint = ModelCheckpoint('best_pneumonia_modelv2.keras' , save_best_only = True , monitor = 'val_accuracy')
early_stopping = EarlyStopping( monitor = 'val_accuracy' , patience = 15 , restore_best_weights = True )
reduce_lr = ReduceLROnPlateau( monitor = 'val_accuracy' , factor = 0.3 , patience = 2 , cooldown = 5 )

# Train the model 

history = model.fit(
    datagen.flow(X_train , y_train , batch_size = 32), 
    epochs = 150 , 
    validation_data = (X_val , y_val) , 
    callbacks = [ checkpoint , early_stopping , reduce_lr ]
)

# Evaluate the model on the test set -- trains the model using the augmented data , validation data and callback functions 

test_loss , test_acc , = model.evaluate( X_test , y_test , verbose = 2)
print(f"\nTest accuracy : {test_acc}")

# Load the best Model 

best_model = tf.keras.models.load.model('best_pneumonia_modelv2.keras')
predictions = best_model.predict(X_test)
predicted_classes = np.argmax(predictions , axis = 1 )

# classification report 

target_names = label_encoder.inverse_trainsform(np.unique(encoded_labels)).astype(str)
print(classification_report(y_test , predicted_classes , target_names = target_names))
