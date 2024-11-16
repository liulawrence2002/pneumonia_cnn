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
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = MaxPooling2D((2, 2))(x)

    # Second Convolutional Block
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = MaxPooling2D((2, 2))(x)

    # Third Convolutional Block
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = Conv2D(128, (3, 3), padding='same')(x)
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
    x = MaxPooling2D((3, 3))(x)

    # Fifth Convolutional Block
    x = Conv2D(1024, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)        
    x = Conv2D(512, (3, 3), padding='same')(x)
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


'''
1. Conv2D (Convolutional 2D Layer)
Definition: This layer performs a 2D convolution operation on input data (e.g., images), applying a set of learnable filters or kernels to extract features from the data.
Purpose: It helps detect patterns like edges, textures, and more complex structures in the data as the filters slide over the input.
Use Case: Primarily used in Convolutional Neural Networks (CNNs) for image classification and recognition tasks.
2. MaxPooling2D
Definition: This layer performs max pooling, a form of downsampling where it takes the maximum value from a set of values in a pooling window (usually 2x2 or 3x3) and discards the rest.
Purpose: Reduces the spatial dimensions of the input, thereby reducing computation and helping with invariance to small translations of the input.
Use Case: Often used after a Conv2D layer to reduce the dimensionality of feature maps.
3. Flatten
Definition: The Flatten layer is used to convert a multi-dimensional input (e.g., a 2D feature map) into a 1D vector.
Purpose: This is necessary to feed the data into fully connected (Dense) layers after feature extraction through convolution and pooling.
Use Case: Typically used before a Dense layer in CNNs.
4. Dense (Fully Connected Layer)
Definition: A Dense layer connects every neuron in the layer to every neuron in the previous layer, forming a fully connected neural network.
Purpose: It performs a weighted sum of the inputs followed by an activation function, helping the model learn non-linear relationships.
Use Case: Used in classification or regression tasks, especially towards the end of CNNs or other deep neural networks.
5. Dropout
Definition: Dropout is a regularization technique where randomly selected neurons are "dropped" (set to zero) during training, preventing overfitting by reducing reliance on any specific neuron.
Purpose: Helps to prevent overfitting by forcing the model to generalize better.
Use Case: Typically used in Dense layers, especially in deep networks.
6. BatchNormalization
Definition: This technique normalizes the input of each layer so that it has a mean of 0 and a standard deviation of 1, applied across a mini-batch of data.
Purpose: Improves convergence during training, stabilizes learning, and can lead to faster training.
Use Case: Often used after Conv2D and Dense layers to maintain stability during training.
7. ReLU (Rectified Linear Unit)
Definition: ReLU is an activation function that outputs the input directly if it is positive; otherwise, it outputs zero.
Purpose: Introduces non-linearity into the network and helps with faster convergence compared to older activation functions like sigmoid or tanh.
Use Case: Commonly used in CNNs and other neural networks.
8. Input
Definition: The Input layer is where the data is fed into the model. It specifies the shape and type of the input data.
Purpose: Defines the input shape for the network, without any transformation.
Use Case: Used as the starting point for defining neural networks.
9. Model
Definition: In Keras (a high-level neural network API), the Model is a container that represents the architecture of the neural network.
Purpose: It defines the layers, how they connect, and the overall flow of data through the network.
Use Case: Used to define the architecture of a deep learning model (e.g., Sequential or Functional API models).
10. EarlyStopping
Definition: This is a callback that stops training the model when a monitored metric (like validation loss) stops improving for a set number of epochs.
Purpose: Prevents overfitting by halting training when further improvements are unlikely.
Use Case: Typically used when training deep neural networks to avoid overfitting and save time.
11. ModelCheckpoint
Definition: This callback saves the model or weights at regular intervals during training (usually when the validation loss improves).
Purpose: Helps to keep the best version of the model during training.
Use Case: Commonly used to save the model with the best validation performance during training.
12. ReduceLROnPlateau
Definition: This callback reduces the learning rate when the validation performance plateaus, helping the model to converge more effectively.
Purpose: Allows the model to fine-tune at lower learning rates when it reaches a plateau.
Use Case: Used to adjust the learning rate dynamically during training to improve convergence.
13. ImageDataGenerator
Definition: This class is used for real-time data augmentation (e.g., random rotations, flips) and data loading during training.
Purpose: Enhances the training dataset by applying transformations on the fly, improving generalization.
Use Case: Commonly used with image data to generate more varied data from the original dataset.
14. Adam (Optimizer)
Definition: Adam (short for Adaptive Moment Estimation) is an optimization algorithm that computes adaptive learning rates for each parameter using both the first and second moments of the gradients.
Purpose: Efficiently adjusts the learning rates during training, improving convergence speed.
Use Case: One of the most popular optimizers for training deep learning models.
15. train_test_split
Definition: A function from scikit-learn that splits a dataset into training and testing subsets.
Purpose: Ensures that a portion of the data is used for training and a separate portion is used for evaluation.
Use Case: Typically used to evaluate how well a model generalizes to unseen data.
16. LabelEncoder
Definition: A tool from scikit-learn to convert categorical labels into numeric values.
Purpose: Makes it easier to work with categorical data in machine learning models by converting labels into integers.
Use Case: Used in classification tasks when the target variable is categorical.
17. classification_report
Definition: A function from scikit-learn that computes and outputs a report showing key classification metrics like precision, recall, F1-score, and support.
Purpose: Provides a summary of the performance of a classifier on a given dataset.
Use Case: Used to evaluate the quality of classification models, especially for imbalanced datasets.
These components are fundamental for building, training, and evaluating deep learning models, particularly in the context of image classification tasks.
'''