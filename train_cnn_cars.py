# Example and explanations from: https://www.learnopencv.com/keras-tutorial-transfer-learning-using-pre-trained-models/

from keras.applications import vgg16
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
import numpy as np
import keras
import os
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import load_img

# Load pretrained network without the last two fully connected layers
vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

img_width, img_height = 224, 224
train_dir = 'Data/train'
validation_dir = 'Data/validation'
nTrain = 800   # Number of training samples
nVal = 200  # Number of validation samples
batch_size = 20


datagen = ImageDataGenerator(rescale=1./255)    # Rescale all image values to be between 0 and 1.

train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=(nTrain,2))

# Automatically load all images from given directory and create batches of images and labels
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),    # Resize all images to this size
    batch_size=batch_size,
    class_mode='categorical',   # Determines the type of label arrays that are returned: "categorical" will be 2D one-hot encoded labels
    shuffle=True)    # Randomly place images in batches

# Pass training images through the network and reshape the output tensor into a vector
i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    train_features[i * batch_size : (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nTrain:
        break
train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))

# Automatically load validation data
validation_features = np.zeros(shape=(nVal, 7, 7, 512))
validation_labels = np.zeros(shape=(nVal,2))

validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Pass validation images through the network and reshape the output tensor into a vector
i = 0
for inputs_batch, labels_batch in validation_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nVal:
        break

validation_features = np.reshape(validation_features, (nVal, 7 * 7 * 512))


################################################
# Creating own layers and trainng the network

from keras import models
from keras import layers
from keras import optimizers

print("Training the last classifier layer...")

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features,
                    train_labels,
                    epochs=20,
                    batch_size=batch_size,
                    validation_data=(validation_features,validation_labels))

# Validate
fnames = validation_generator.filenames

ground_truth = validation_generator.classes

label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())

predictions = model.predict_classes(validation_features)
prob = model.predict(validation_features)

errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),nVal))

# Save models
vgg_conv.save('model1.h5')  # Save VGG16 models and weights
model.save("model2.h5")     # Save my trained top layer

# Predict image
# Read image and pre-process it for image classification
original = load_img("car1.png", target_size=(224, 224))
numpy_image = img_to_array(original)
image_batch = np.expand_dims(numpy_image, axis=0)

# Put through VGG16 network and reshape
predicted_features = vgg_conv.predict(image_batch)
to_classification = np.reshape(predicted_features, (1, 7 * 7 * 512))

# Predict by applying the vector to our little classification network
predictions = model.predict(to_classification)

print(predictions)
print(idx2label)
