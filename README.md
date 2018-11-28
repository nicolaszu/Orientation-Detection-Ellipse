# Orientation Detection Ellipse Python

Python3 + OpenCV script for detecting orientation inside a tube-like structure. It was originally designed for an endoscopic robot and showed promising results when compared to results from an accelerometer.

Version 2 is currently being developed. 

## Usage
To run:

```bash
python OrientationDetection.py -v nameOfVideo
```

## Explanation
#### Step 1: Parse the Video
```python
argprs = argparse.ArgumentParser()
argprs.add_argument("-v", "--video", required=True, help="video file input")
arguments = vars(argprs.parse_args())
stream = cv2.VideoCapture(arguments["video"])
```
#### Step 2: Preprocessing Video Frames
 
```python
grayImage = cv2.cvtColor(uneditedFrame, cv2.COLOR_BGR2GRAY)
blurredGrayImage = cv2.GaussianBlur(grayImage, (155, 155), 0)
ret, threshImage = cv2.threshold(blurredGrayImage, 10, 255, cv2.THRESH_BINARY_INV) 
threshImage = cv2.erode(threshImage, None, iterations=2)
threshImage = cv2.dilate(threshImage, None, iterations=2)
```
#### Step 3: Segment Video Frames:
 
```python
contourArray = cv2.findContours(threshImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contourArray = contourArray[0] if imutils.is_cv2() else contourArray[1]

```

#### Step 4: Finding Largest Contour of Video Frames
 
```python
  maxContour = max(contourArray, key=cv2.contourArea, default=0)
        if len(maxContour) >= 5:
```

#### Step 5: Apply Fit Ellipse Algorithm to Deduce the Orientation
```python
  (x, y), (MA, ma), ellipseAngle = cv2.fitEllipse(maxContour)  
   ellipse = cv2.fitEllipse(maxContour)  
   print("Estimated Angle of Orientation: %f" % ellipseAngle)
```

#### Additonal steps:
The script checks for extreme changes in Orientation. It detects if 2 continuous values have jumped by more than 100 degrees. If they have it adds 180 to the predicted angle.

Additionally, the script calculates moving average in order to accurately compare to accelerometer data. 
