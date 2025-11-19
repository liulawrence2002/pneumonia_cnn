import cv2 # this is the libary that is used to convert the image into rgb
import os # this is the libary to locate local files from the computer 
import imageio # this is the libary that is used to read images 
import numpy as np # this is the libary for numerical operations 
import pandas as pd # this is the libary for data manipulation 
import tensorflow as tf # this is the libary for deep learning (machine learning/ neural networks)
from tensorflow.keras.models import Model , Sequential # these are models that will be called on later in the code
from tensorflow.keras.layers import Dense, Flatten , Conv2D, MaxPooling2D , Input , Dropout , BatchNormalization, Activation , GlobalAveragePooling2D, Add, Concatenate # these are the input , hidden and output layers within the CNN model 
from tensorflow.keras.optimizers import Adam # this is the optimization function 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint , ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.regularizers import l2
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
def preprocess_data(spreadsheet, dir_path, num_records=5200 , target_size= (224,224)):
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
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
            image = cv2.resize (image , (224,224))
            image_array = np.array(image) / 255.00 # this will normalize the image to a value between 0 and 1 (this will simpify the calculations)
            images.append(image_array)
            pneumonias.append(pneumonia)
        else:
            print(f"image file {image_path} not found.")
    return np.array(images) , np.array(pneumonias)
# Preprocess the data
images, labels = preprocess_data(spreadsheet, DIR, num_records=5200, target_size=(224, 224))
# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(np.unique(encoded_labels))
# Split the data into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    images, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)
# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True
    #fill_mode='nearest',
   # brightness_range=[0.8,1.2],
   # shear_range=0.1
)
datagen.fit(X_train)

def build_modelv2(input_shape= (224,224,3) , num_classes =3 ):
     inputs = Input(shape = input_shape)
    
     # this is the first convolutional block
     conv1 = Conv2D(32 , (5,5) , strides = 1 , padding ='same' )(inputs)
     bn1 = BatchNormalization()(conv1)
     a1 = Activation('relu')(bn1)
     conv2 = Conv2D(32, (5,5) , strides =1 , padding ='same')(a1) 
     bn2 = BatchNormalization()(conv2)
     res1 = Add()([bn1, bn2])
     a2 = Activation('relu')(res1)
     mpool1 = MaxPooling2D(2,2)(a2)
    # this is the second convolutional block
     conv3 = Conv2D(64 , (4,4) , strides =1 , padding = 'same')(mpool1)
     bn3 = BatchNormalization()(conv3)
     a3 = Activation('relu')(bn3)
     conv4 = Conv2D(64, (4,4) , strides = 1 , padding ='same')(a3) 
     bn4 = BatchNormalization()(conv4)
     res2 = Add()([bn3, bn4])
     a4 = Activation('relu')(res2)
     mpool2 = MaxPooling2D(2,2)(a4)
    # third convolutional block 
     conv5 = Conv2D(128,(3,3) , strides =1 , padding ='same')(mpool2)
     bn5 = BatchNormalization()(conv5)
     a5 = Activation('relu')(bn5)
     conv6 = Conv2D(128,(3,3) , strides =1 , padding ='same')(a5)
     bn6 = BatchNormalization()(conv6)
     res3 = Add()[(bn5 , bn6)]
     a6 = Activation('relu')(res3)
     mpool3 = MaxPooling2D(3,3)(a6)
     # For the final layer a parallel path will be used 
    # first path 
     conv7 = Conv2D(256 , (3,3) , strides = 1, padding ='same')(mpool3)
     bn7 = BatchNormalization()(conv7)
     a7 = Activation('relu')(bn7)
     # second path 
     conv8 = Conv2D(256 , (1,1), strides = 1 , padding ='same')(mpool3)
     bn8 = BatchNormalization()(conv8)
     a8 = Activation('relu')(bn8)
     #merge the two paths into one using concat keras function 
     Merge = Concatenate()([a7, a8])
     mpool4 = MaxPooling2D((2,2))(Merge)

     flat = GlobalAveragePooling2D()(mpool4)
     dense1 = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(flat)
     drop1 = Dropout(0.5)(dense1)
     dense2 = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(drop1)
     drop2 = Dropout(0.3)(dense2)
    
     outputs = Dense(num_classes, activation='softmax')(drop2)
     model = Model(inputs, outputs)
     return model

'''
# Build the model based on Table 1 from the paper
def build_model(input_shape=(224, 224, 3), num_classes=3):  # Change num_classes to 3
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (5,5) , strides = 1 , padding = 'same' )(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (4,4) , strides = 1 , padding ='same' , kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2) , strides =  2 , padding = 'same')(x)
    x= Conv2D(64 , (2,2) , strides = 1 , padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (2,2) , strides =1 , padding ='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2) , strides = 2 , padding ='same')(x)
    x = Conv2D(128 , (1,1) , strides =1 , padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (1,1) , strides =1 , padding ='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3) , strides = 3 , padding ='same')(x)
    x = Conv2D(256 , (1,1) , strides =1 , padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (1,1) , strides =1 , padding ='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3) , strides = 3 , padding ='same')(x)
    x = Conv2D(1024 , (1,1) , strides =1 , padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (1,1) , strides =1 , padding ='same')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2) , strides = 2 , padding ='same')(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Dense(1024)(x)
    x = Dense(512)(x)
    x = Dense(512)(x)
    x = Dense(256)(x)
    x = Dense(64)(x)
    outputs = Dense(num_classes, activation='softmax')(x)  # Set to 3 classes
    model = Model(inputs, outputs)
    return model
'''
# Compile the model
model = build_modelv2()
model.summary()
model.compile(optimizer=Adam(learning_rate=1e-4 , clipnorm =1.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Callbacks for training
checkpoint = ModelCheckpoint('best_pneumonia_modelrgbv2.keras', save_best_only=True, monitor='val_accuracy')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=1e-7, cooldown=3)
# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=150,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping, reduce_lr]
)
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")
# Load the best model
best_model = tf.keras.models.load_model('best_pneumonia_modelrgbv2.keras')
predictions = best_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
# Classification report
target_names = label_encoder.inverse_transform(np.unique(encoded_labels)).astype(str)
print(classification_report(y_test, predicted_classes, target_names=target_names))


