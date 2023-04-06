import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import to_categorical

# dataset location
dataset_dir = './asl_dataset'

# define list of classes
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# define the size of the images
img_width = 64
img_height = 64

def read_dataset():
    data = []
    labels = []
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            img = Image.open(img_path).convert('RGB') # convert to RGB for 3 channels
            img = img.resize((img_width, img_height))
            img_array = np.array(img) / 255.0
            data.append(img_array)
            labels.append(classes.index(cls))
    data = np.array(data)
    labels = np.array(labels)
    labels = to_categorical(labels) # convert labels to categorical
    return data, labels

# split the dataset into training and testing sets
def split_dataset(data, labels):
    num_samples = len(data)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    train_indices = indices[:int(0.8*num_samples)]
    test_indices = indices[int(0.8*num_samples):]
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    return train_data, train_labels, test_data, test_labels

# call the functions to read and preprocess the the dataset
data, labels = read_dataset()
train_data, train_labels, test_data, test_labels = split_dataset(data, labels)

# define the input shape
input_shape = (img_width, img_height, 3)

# define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# print the model summary
model.summary()

# compile the model by specifying the optimizer, loss function and metrics to be used during training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model by calling the fit method on the training data and specifying the number of epochs and batch size
history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))

# evaluate the trained model
test_loss, test_acc = model.evaluate(test_data, test_labels)

model.save('asl_model1.h5')