from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np
import cv2

# Load both networks
print("[INFO] loading networks...")
model1 = load_model("model1.h5")
model2 = load_model("model2.h5")
print("[INFO] Networks are loaded!")

# Read image and pre-process it for image classification
original = load_img("1.png", target_size=(224, 224))
numpy_image = img_to_array(original)
image_batch = np.expand_dims(numpy_image, axis=0)

# Classify the input image
print("Classification is started...")
# Put through VGG16 network and reshape
predicted_features = model1.predict(image_batch)
to_classification = np.reshape(predicted_features, (1, 7 * 7 * 512))

# Predict by applying the vector to our little classification network
predictions = model2.predict(to_classification)

print(predictions)
