#To run: python ContourAndFitShapes_OrientationPrediction.py -v nameOfVideo

import numpy as np
import pandas as pd
import math
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
import time
#import h5py
#import glob

# dataRe = h5py.File('dataRetrieved.hdf5', libver='latest')  <--- Used to save array of all orientations as a dataset file. Useful for possible machine learning or other pattern recognition techniques in the future

accelerometerDataTemp = pd.read_excel('croppedOD.xlsx',sheet_name='Test1') #save data obtained from accelerometer
accelerometerData = accelerometerDataTemp.as_matrix()
AccelerometerAngles = []
AccelerometerTimes = []

WarningPossible180Accelerometer = False
WarningCountAccelerometer = 0
ConfirmedPassed180Accelerometer = False
lastBigValueAccelerometer = 0

PredictedAngles = []
TimesOfPredictedAngles = []

WarningPossible180 = False
WarningCount = 0
ConfirmedPassed180 = False
lastBigValue = 0

argprs = argparse.ArgumentParser()
argprs.add_argument("-v", "--video", required=True, help="video file input")
arguments = vars(argprs.parse_args())  # parse video file
stream = cv2.VideoCapture(arguments["video"])  # If "live" video footage is required, use cap = cv2.VideoCapture(0)
# fourcc= cv2.VideoWriter_fourcc(*'H264') <--- Used to save videos, delete if not needed
# out = cv2.VideoWriter('CFSop.mp4',fourcc,20.0,(640,480))
while True:

    (grabbed, uneditedFrame) = stream.read()

    if not grabbed:
        break  # break while loop if no more frames available

    grayImage = cv2.cvtColor(uneditedFrame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    blurredGrayImage = cv2.GaussianBlur(grayImage, (155, 155), 0)  # smooth image

    ret, threshImage = cv2.threshold(blurredGrayImage, 10, 255, cv2.THRESH_BINARY_INV) #segmenting image based on intensity values, low intensity= white
    threshImage = cv2.erode(threshImage, None, iterations=2)
    threshImage = cv2.dilate(threshImage, None, iterations=2)

    contourArray = cv2.findContours(threshImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #grouping based on results from threshold image
    contourArray = contourArray[0] if imutils.is_cv2() else contourArray[1]
    if len(contourArray) != 0:  # checks if a contour was succesfully found before proceeding
        maxContour = max(contourArray, key=cv2.contourArea, default=0)  # key points of contour
        if len(maxContour) >= 5:  # needs to find more than 5 key points to fit an ellipse
            (x, y), (MA, ma), ellipseAngle = cv2.fitEllipse(maxContour)  # fit ellipse applied to the contour
            ellipse = cv2.fitEllipse(maxContour)  # delete this if you drawing the ellipse is not needed
            cv2.ellipse(uneditedFrame, ellipse, (255, 255, 0), 2)  # delete if drawing the contour is not needed
            print("Estimated Angle of Orientation: %f" % ellipseAngle)  # Predicted angle obtained from ellipse

            # PredictedAngles.append(ellipseAngle) <---UNCOMMENT THIS FOR 180 DEGREE PREDICTION
            # TimesOfPredictedAngles.append(time.time()) <--- UNCOMMENT FOR 180 DEGREES

            # COMMENT THE REST OUT FOR 180 DEGREES INSTEAD OF 360 DEGREE PREDICTION. THIS WAS THOROUGHLY TESTED AND BASED ON THE GRAPHICAL OUTPUT I BELIEVE IT WORKS BUT I DID NOT COMPARE WITH VALUES OF ACCELEROMETER SO IM UNSURE OF THE ITS ACCURACY.
            # IT DETECTS IF 2 CONTINUOUS VALUES HAVE JUMPED BY MORE THAN 100 DEGREES. IF THEY HAVE IT ADDS 180 TO THE PREDICTED ANGLE.

            if ConfirmedPassed180:  # more than 180 AKA 181-360 degrees
                ellipseAngle += 180
                PredictedAngles.append(ellipseAngle)
            else:                  # 180 or less AKA 0-180
                PredictedAngles.append(ellipseAngle)
            TimesOfPredictedAngles.append(time.time())

            if len(PredictedAngles) > 1:
                differenceInAngles = PredictedAngles[(len(PredictedAngles)-2)]-ellipseAngle  # calculates differnce between last 2 angles
                if differenceInAngles >= 100:  # checks for an extreme change in angles
                    WarningPossible180 = True  # possible need to add 180
                    lastBigValue = PredictedAngles[(len(PredictedAngles)-2)]  # saves the big value for comparison with future angles
            differenceInAnglesAndLastBigValue = lastBigValue-ellipseAngle

            if WarningPossible180 and differenceInAnglesAndLastBigValue >= 100:
                WarningCount += 1
                if WarningCount >= 2 and not ConfirmedPassed180:  # this checks whether we need to add 180 or stop adding 180
                    ConfirmedPassed180 = True  # if the continuous values have a difference greater than 100 to the big value, assume that there is a need to add 180
                    for temp in range(1, 3):  # remove the 2 values used to check against the big value and correct them by adding 180
                        angleToChange = PredictedAngles.pop(-temp)
                        angleToChange = angleToChange + 180
                        PredictedAngles.append(angleToChange)
                    WarningCount = 0  # reset check
                    differenceInAnglesAndLastBigValue = 0  # reset check
                    WarningPossible180 = False  # reset check

                elif WarningCount >= 2 and ConfirmedPassed180:  # stop adding 180
                    ConfirmedPassed180 = False
                    for temp in range(1, 3):
                        angleToChange = PredictedAngles.pop(-temp)
                        angleToChange = angleToChange - 180
                        PredictedAngles.append(angleToChange)
                    WarningCount = 0
                    differenceInAnglesAndLastBigValue = 0
                    WarningPossible180 = False

            if WarningPossible180 and differenceInAnglesAndLastBigValue < 100:  # False alarm, second value didnt exihibit big change in value
                WarningPossible180 = False
                WarningCount = 0
                differenceInAnglesAndLastBigValue = 0
    # out.write(uneditedFrame)
    cv2.imshow("Frame", uneditedFrame)
    cv2.waitKey(15)

stream.release()
#out.release()

#EVERYTHING BELOW IS USED TO PLOT THE DATA AND CALCULATE THE RATE OF ERROR. AT THE MOMENT IT IS COMPARING 360 DEGREES WITH 180 DEGREES FROM ACCELEROMETER DUE TO COMPLICATIONS WITH THE ACCELEROMETER DATA. COMMENT OUT THE PART OF MENTIONED ABOVE FOR 180 TO 180 COMPARISON
for x in range(len(accelerometerData)):
    TempAccelerometerAngle = math.degrees(np.arctan((accelerometerData[x][2])/(np.power(accelerometerData[x][1],2)+np.power(accelerometerData[x][3],2))))  # accelerometer raw data to angle
    AccelerometerAngles.append(TempAccelerometerAngle)
    AccelerometerTimes.append(accelerometerData[x][0])

movingAverageOfAccelerometerAngles = np.convolve(AccelerometerAngles, np.ones((10,))/10, mode='valid')  # Moving average
movingAverageOfAccelerometerTime = np.convolve(AccelerometerTimes, np.ones((10,))/10, mode='valid')

firstAngleOfAccelerometer = movingAverageOfAccelerometerAngles[0]
firstTimeOfAccelerometer = movingAverageOfAccelerometerTime[0]

movingAverageOfAccelerometerAnglesFrom0 = list(map(lambda everyAngle: everyAngle-firstAngleOfAccelerometer, movingAverageOfAccelerometerAngles)) # start from angle 0
movingAverageOfAccelerometerTimeFrom0 = list(map(lambda everyTime: (everyTime-firstTimeOfAccelerometer)/1000, movingAverageOfAccelerometerTime))


movingAverageOfPredictedAngles = np.convolve(PredictedAngles, np.ones((30,))/30, mode='valid')
movingAverageOfPredictedTimes = np.convolve(TimesOfPredictedAngles, np.ones((30,))/30, mode='valid')

firstPredictedAngle = movingAverageOfPredictedAngles[0]
firstPredictedTime = movingAverageOfPredictedTimes[0]

movingAverageOfPredictedAnglesFrom0 = list(map(lambda everyAngle: everyAngle-firstPredictedAngle, movingAverageOfPredictedAngles))
movingAverageOfPredictedTimesFrom0 = list(map(lambda everyTime: (everyTime-firstPredictedTime), movingAverageOfPredictedTimes))

plt.scatter(movingAverageOfPredictedTimesFrom0, movingAverageOfPredictedAnglesFrom0, label='Fit Ellipse Orientation')
plt.scatter(movingAverageOfAccelerometerTimeFrom0, movingAverageOfAccelerometerAnglesFrom0, label='Accelerometer Orientation')
plt.xlabel('Time in seconds')
plt.ylabel('Orientation in degrees')
plt.title('Fit Ellipse Orientation VS Accelerometer Orientation')
plt.legend()
plt.show()

np.save('movingAverageOfAccelerometerAnglesFrom0', movingAverageOfAccelerometerAnglesFrom0) #save for use as np array, useful for rate of error
np.save('movingAverageOfPredictedAnglesFrom0', movingAverageOfPredictedAnglesFrom0)

# dataRe.create_dataset('dataset_1', data=PredictedAngles)
# dataRe.close()
cv2.destroyAllWindows()

