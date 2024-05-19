import cv2
import cv2.aruco as aruco
import sys
import numpy as np

def nothing(x):
    pass

video_id = 2

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)

# invert color
# get blue channel
# threshold
# etc.

# https://stackoverflow.com/questions/7904055/detect-hand-using-opencv

while True:
    # Create a window
    cv2.namedWindow('frame')

    # Capture a frame from the webcam
    ret, frame = cap.read()

    # make the contrast bigger and adjust for aditional brightness 
        # see: https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html
        # see: https://stackoverflow.com/a/56909036
    alpha = 1.8 # contrast (1.0 - 3.0)
    beta = -30 # brightness control (0-100)
    contrast = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    # ?? -> kind make threshold worse


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # maximize brightness of paper
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0] # maximize brightness of paper
    # changed from: https://stackoverflow.com/a/72264323
    # ??: try HSV (https://stackoverflow.com/q/22588146, https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html,
        # https://stackoverflow.com/a/64258373, https://stackoverflow.com/a/72264323)

    
    # bluring the image for a smother image # !!
    blur = cv2.GaussianBlur(l_channel,(7,7),0)

    # use automatic thresholding: see https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html (Otsu's Binarization)
    t_min = 50 # exclude extremes
    t_max = 200 
    ret, thresh = cv2.threshold(blur, t_min, t_max, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # MORPH
    kernel_size=(3, 3)
    kernel = np.ones(kernel_size, dtype=np.float64)

    dilation = cv2.dilate(thresh, kernel)
    erotion = cv2.erode(thresh, kernel)
    closing = cv2.erode(dilation, kernel)

    # detect edges (see: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
    c_min = 200
    c_max = 200
    canny = cv2.Canny(closing, c_min, c_max)

    # detect contours 
    # (see: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
        # "object to be found should be white and background should be black"
    # (see: https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # draw bounding box (see: https://stackoverflow.com/a/23411041)
    # for c in contours:
        # get points furtherst to right/left
        # rect = cv2.boundingRect(c)
        # if rect[2] < 100 or rect[3] < 100: continue
        # x,y,w,h = rect
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.putText(frame,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))
    
    # get two righ/left most points

    # see: https://pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
    c = max(contours, key=cv2.contourArea)

    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])

    # determine second most extreme points (see: https://stackoverflow.com/a/60935067)      

    # draw the outline of the object, then draw each of the
    # extreme points, where the left-most is red, right-most
    # is green, top-most is blue, and bottom-most is teal
    # cv2.drawContours(frame, [c], -1, (0, 255, 255), 2)
    
    cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
    cv2.circle(frame, extRight, 8, (0, 255, 0), -1)

    
    
    output = frame
    # Detect major color = white -> get value, use that for trheshold
        # LAB = L brightness, A = red-green, b = blue-yellow
        # HSL = hue, saturation, lightness

    
    # contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    



    cv2.imshow('frame', output)
    # cv2.imshow('frame', canny)

    # Wait for a key press and check if it's the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
