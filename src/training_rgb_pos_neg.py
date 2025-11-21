import cv2 # this is the libary that is used to convert the image into rgb
import os # this is the libary to locate local files from the computer
import imageio # this is the libary that is used to read images
import numpy as np # this is the libary for numerical operations
import pandas as pd # this is the libary for data manipulation
import tensorflow as tf # this is the libary for deep learning (machine learning/ neural networks)
from tensorflow.keras.models import Model , Sequential # these are models that will be called on later in the code
from tensorflow.keras.layers import Dense, Flatten , Conv2D, MaxPooling2D , Input , Dropout , BatchNormalization, Activation # these are the input , hidden and output layers within the CNN model
from tensorflow.keras.optimizers import Adam # this is the optimization function
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint , ReduceLROnPlateau, TensorBoard # TensorBoard for visualization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split # this is the library that is used to split data into a training and testing subset
from sklearn.preprocessing import LabelEncoder # this is the function to encode labels
from sklearn.metrics import classification_report # this is the function that is used to generate a classification report
from PIL import Image # this is the libary that is used to handle image operations
from datetime import datetime # this is used to create unique log directories for TensorBoard 
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
        if pneumonia ==2:
            pneumonia =1
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
if label_encoder ==2:
    label_encoder =1
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
)
datagen.fit(X_train)
# Build the model based on Table 1 from the paper
def build_model(input_shape=(224, 224, 3), num_classes=3):  # Change num_classes to 3
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3,3) , strides = 1 , padding = 'same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3,3) , strides = 1 , padding ='same')(x)
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
    x = Dropout(0.45)(x)
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

    outputs = Dense(num_classes, activation='softmax')(x)  # Set to 2 
    model = Model(inputs, outputs)
    return model
# Compile the model
model = build_model()
model.summary()
model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create unique TensorBoard log directory with timestamp
log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
print(f"\nTensorBoard logs will be saved to: {log_dir}")
print("To view TensorBoard, run: tensorboard --logdir=logs/fit")

# Callbacks for training
checkpoint = ModelCheckpoint('best_pneumonia_modelrgbposneg.keras', save_best_only=True, monitor='val_accuracy')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, cooldown=5)

# TensorBoard callback for real-time visualization
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,  # Log weight histograms every epoch
    write_graph=True,  # Visualize the model graph
    write_images=False,  # Don't log image data (can be large)
    update_freq='epoch',  # Log metrics after each epoch
    profile_batch=0,  # Disable profiling to save resources
    embeddings_freq=0
)
# Train the model
print("\n" + "="*60)
print("Starting model training...")
print("="*60)
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard_callback]
)
print("\n" + "="*60)
print("Training completed!")
print("="*60)
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")
# Load the best model
best_model = tf.keras.models.load_model('best_pneumonia_modelrgbposneg.keras')
predictions = best_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
# Classification report
target_names = label_encoder.inverse_transform(np.unique(encoded_labels)).astype(str)
print(classification_report(y_test, predicted_classes, target_names=target_names))


