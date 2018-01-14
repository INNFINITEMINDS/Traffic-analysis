import cv2
import numpy as np
import os
import re

cap = cv2.VideoCapture("Videos/new/2017_06_23_1430_Falen_Cigaren mod_byen.mp4")
AREA = 100.0
MAG_THRESH = 0.6
anno_file_path = "Annotations/"
SPEED = 100  # Rewind speed

def getFrame():
    ret, colorframe = cap.read()
    grayframe = cv2.cvtColor(colorframe,cv2.COLOR_BGR2GRAY)
    return colorframe,grayframe

# Rewind or fast forward video
def rewindVideo(direction):
    numberOfFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    newFrame = currentFrame

    # Determine new frame
    if direction == "forward":
        newFrame = currentFrame + SPEED
        if newFrame > numberOfFrames:
            newFrame = numberOfFrames
    elif direction == "backward":
        newFrame = currentFrame - SPEED
        if newFrame < 0:
            newFrame = 0
    print("Frame: " + str(newFrame) + " of: " + str(numberOfFrames))

    cap.set(cv2.CAP_PROP_POS_FRAMES, newFrame)  # Set new frame

# Search the specific directory and return the highest filenumber
def getNextFileNumber(folderPath):
    files = os.listdir(folderPath)
    biggestNum = 0
    for str in files:   # loop through files
        number = re.findall(r'\b\d+\b', str)   # Regular expression to extract numbers from a string
        num = int(number[0])
        if num > biggestNum:    # find highest number
            biggestNum = num
    return biggestNum

# Show image and choose where to save it
def saveImage(image):
    pause = True
    exitProgram = False
    selection = 1

    # Save car image based on key press
    k = cv2.waitKey(0)
    if k == 27:  #ESC key
        exitProgram = True
        return pause, exitProgram
    if k == 32: # Space-bar key
        pause = not pause
        return pause, exitProgram

    selection = k - ord('0')    # Convert ASCII to int
    if not selection in range(1,11):    # Select between 1 and 9 otherwise selection becomes 0
        selection = 0
    print(selection)

    # Save image as a file
    filePath = anno_file_path + str(selection) + "/"
    number = getNextFileNumber(filePath)    # Get the next file number
    number = number + 1
    cv2.imwrite(filePath + str(number)+".png",image)

    return pause, exitProgram

prvsColor, prvsGray = getFrame()
nextColor, nextGray = getFrame()

def findMovingObjects(pause):
    global prvsGray
    global prvsColor
    global nextColor
    global nextGray
    exitProgram = False
    hsv = np.zeros_like(prvsColor)
    hsv[:,:,1] = 255

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
            cv2.rectangle(markedImage,(x,y),(x+w,y+h),(0,255,0),2)  # Draw green rectangles

    markedImage = cv2.resize(markedImage, (0,0), fx=0.53, fy=0.53)
    #cv2.namedWindow('Marked objects', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Marked objects',markedImage)
    cv2.moveWindow("Marked objects", 680, 50)

    if pause == True:
        for cnt in finalContours:
            x,y,w,h = cv2.boundingRect(cnt) # x,y is top left coordinate, w,h is width and height
            subImage = nextColor[y:y+h,x:x+w]    # Extract car image
            #cv2.namedWindow('Object', cv2.WINDOW_NORMAL)
            cv2.imshow('Object', subImage)
            #cv2.resizeWindow('Object', 600,600)
            cv2.moveWindow("Object", 500, 550)
            pause, exitProgram = saveImage(subImage)
            if not (pause == True and exitProgram == False):
                return pause, exitProgram

    return pause, exitProgram

if __name__ == '__main__':

    pause = True
    rewind_dir = "forward"
    exitProgram = False
    global prvsGray
    global prvsColor
    global nextColor
    global nextGray

    while(1):
        nextColor, nextGray = getFrame()    # Get next video frame

        videoImage = cv2.resize(nextColor, (0,0), fx=0.53, fy=0.53)
        cv2.imshow('Video',videoImage)   # Display original video, if pause = False, the video runs as fast as possible
        cv2.moveWindow("Video", 0, 50)

        if pause == True:  # Use this if you tracking is not wanted when video is running
            pause, exitProgram = findMovingObjects(pause)

        # Handle user inputs
        k = cv2.waitKey(50) # 50hz frame-rate
        if k == 27 or exitProgram == True:  #ESC key
            break
        if k == 32: # Space-bar key
            pause = not pause
        if k == 83: # Right arrow key: fast forward video
            rewindVideo("forward")
        if k == 81: # Left arrow key: rewind video
            rewindVideo("backward")

        # Save previous video frame
        prvsGray = nextGray
        prvsColor = nextColor

    cap.release()
    cv2.destroyAllWindows()
