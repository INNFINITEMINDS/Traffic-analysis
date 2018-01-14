from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np
import cv2
from keras.utils import plot_model

AREA = 100.0
MAG_THRESH = 0.6
cap = cv2.VideoCapture("Videos/2017_06_23_1430 Falen Cigaren mod byen-480p.mp4")

def getFrame():
    ret, colorframe = cap.read()
    grayframe = cv2.cvtColor(colorframe,cv2.COLOR_BGR2GRAY)
    return colorframe,grayframe

prvsColor, prvsGray = getFrame()
nextColor, nextGray = getFrame()


# Classify the image to see if it is a car or not
def classify_image(image, model1, model2):
    # Read image and pre-process it for image classification
    resized_image = cv2.resize(image, (224, 224))
    #original = load_img("car1.png", target_size=(224, 224))
    numpy_image = img_to_array(resized_image)
    image_batch = np.expand_dims(numpy_image, axis=0)

    # Classify the input image
    print "Classification is started..."
    # Put through VGG16 network and reshape
    predicted_features = model1.predict(image_batch)
    to_classification = np.reshape(predicted_features, (1, 7 * 7 * 512))

    # Predict by applying the vector to our little classification network
    predictions = model2.predict(to_classification)
    print "Classification succeed!"
    print predictions

    car = False
    if predictions.flatten()[0] > predictions.flatten()[1]: # Test if the image is a car
        car = True
        print "The image contains a car!"
    else:
        car = False
        print "The image does not contain a car!"

    return car


def findMovingObjects(model1, model2):
    global prvsGray
    global prvsColor
    global nextColor
    global nextGray
    exitProgram = False
    hsv = np.zeros_like(prvsColor)
    hsv[:,:,1] = 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Optical flow
    flow = cv2.calcOpticalFlowFarneback(prvsGray,nextGray, None, 0.5, 3, 10, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])    # Convert vectors to magnitude and angle

    # Magnitude threshold
    mag[mag < MAG_THRESH] = 0.0

    hsv[:,:,0] = ang*180/np.pi/2
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)

    # Threshold and find contours
    ret,thresh = cv2.threshold(gray,0,255,0)
    imout, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    finalContours = list()
    subImage = nextColor.copy()
    markedImage = nextColor.copy()

    # Only draw contours with a area bigger than a threshold
    for cnt in contours:
        if cv2.contourArea(cnt) > AREA:
            finalContours.append(cnt)   # Only keep the contours with large areas
            x,y,w,h = cv2.boundingRect(cnt) # x,y is top left coordinate, w,h is width and height
            subImage = nextColor[y:y+h,x:x+w]    # Extract car image

            # Classify the subimage and put a rectangle and text on the image
            classify_image(subImage, model1, model2)    # Classify image
            cv2.rectangle(markedImage,(x,y),(x+w,y+h),(0,255,0),1)  # Draw green rectangles
            cv2.putText(markedImage,'Car',(x,y), font, 0.5,(0,255,0),2)

    cv2.imshow('Marked objects',markedImage)

    return subImage


if __name__ == '__main__':
    global prvsGray
    global prvsColor
    global nextColor
    global nextGray

    # Load both neural networks
    print "Loading networks..."
    model1 = load_model("Classification/Car_classification/model1.h5")
    model2 = load_model("Classification/Car_classification/model2.h5")
    print "Networks are loaded!"

    while(1):
        nextColor, nextGray = getFrame()    # Get next video frame

        videoImage = cv2.resize(nextColor, (0,0), fx=0.53, fy=0.53)
        cv2.imshow('Video',videoImage)
        cv2.moveWindow("Video", 0, 50)

        image = findMovingObjects(model1, model2)

        # Handle user inputs
        k = cv2.waitKey(0) # 50hz frame-rate
        if k == 27:     #ESC key: exit program
            break

        # Save previous video frame
        prvsGray = nextGray
        prvsColor = nextColor

    cap.release()
    cv2.destroyAllWindows()
