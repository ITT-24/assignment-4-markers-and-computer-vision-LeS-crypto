import cv2
import cv2.aruco as aruco
import sys
import numpy as np


video_id = 2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)

# invert color
# get blue channel
# threshold
# etc.

class Field():
    def __init__(self):
        self.prev_transform = []

    def transform_game_field(self, ids:list[list[int]], corners, frame):
        """
        Parse the coordinates of the markers to determine the arrangement of the
        markers on the board and transformes them to fit the webcam.
        (The markers should always be in the same spot, but this makes for
        a more robust detection.)
        Keeps the the orientation until all expected markers have been detected.
        Returns the transformed image
        """
        marker_ids = [0, 1, 2, 3] # expected markers
        ids = ids.flatten()
        ids.sort() 
        # id's should always be in the same place, but arent detected like that
        
        if len(ids) == 4 and (marker_ids == ids).all():
            # print("found all markers")
            c = corners

            # store ids of the playfield in the right order
            # 0: top_left, 1: top_right, 2: bot_right, 3: bot_left
            box = np.zeros(4, dtype="int8") 

            # get the top-left corner of each marker for comparison
            m_0 = c[ids[0]][0][0] # x/y coordinates of top-left corner
            m_1 = c[ids[1]][0][0]
            m_2 = c[ids[2]][0][0]
            m_3 = c[ids[3]][0][0]
            ms = [m_0, m_1, m_2, m_3]

            # get to top_l/bot_r corners using min/max
            idx_top_l = np.argmin(np.sum(ms, axis=1))
            idx_bot_r = np.argmax(np.sum(ms, axis=1))
            # print("?", ms, "\n-->", idx_top_l, idx_bot_r)
            box[0] = idx_top_l
            box[2] = idx_bot_r

            if idx_top_l > idx_bot_r:
                ids = np.delete(ids, idx_top_l)
                ids = np.delete(ids, idx_bot_r)
            else: 
                ids = np.delete(ids, idx_bot_r)
                ids = np.delete(ids, idx_top_l)
            # print("rest", ids)

            # compare the second two points:
            # get top_r (x> & y<) & get bot_l (x< & y>)
            if (ms[ids[0]][0] > ms[ids[1]][0]) and (ms[ids[0]][1] < ms[ids[1]][1]):
                idx_top_r = ids[0]
                idx_bot_l = ids[1]
            else:
                idx_top_r = ids[1]
                idx_bot_l = ids[0]

            box[1] = idx_top_r
            box[3] = idx_bot_l
            # print("BOX:", box)
            
            # get coordinates of outside corners and arrange
            top_l = [c[box[0]][0][0][0], c[box[0]][0][0][1]]
            top_r = [c[box[1]][0][1][0], c[box[1]][0][1][1]]
            bot_l = [c[box[2]][0][2][0], c[box[2]][0][2][1]] # id:2
            bot_r = [c[box[3]][0][3][0], c[box[3]][0][3][1]]
            self.prev_transform = np.float32(np.array([top_l, top_r, bot_r, bot_l]))

            """
            m0= °[0] ----- [1]° =m1
                  |         |
            m3= .[3] ----- [2]. =m2
            """

            # keep the transformation if not all markers have been reliably found
        
        if len(self.prev_transform) > 0:
            old_points = self.prev_transform
            old_points = self.prev_transform
            # arrange the new points according to the diff coord system
            height = frame.shape[0]
            width = frame.shape[1]
            # new_points = np.float32(np.array([ [0, 0], [width, 0], [width, height], [0, height] ])) 
            new_points = np.float32(np.array([[width, 0], [0, 0],
                                            [width, height], [0, height]]))


            matrix = cv2.getPerspectiveTransform(old_points, new_points )
            frame = cv2.warpPerspective(frame, matrix, (width, height))


        return frame
field = Field()

# https://stackoverflow.com/questions/7904055/detect-hand-using-opencv

while True:
    # Create a window
    cv2.namedWindow('frame')

    # Capture a frame from the webcam
    ret, frame = cap.read()

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect ArUco markers in the frame
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    if ids is not None:
        # Draw lines along the sides of the marker
        aruco.drawDetectedMarkers(frame, corners, ids)
        frame = field.transform_game_field(ids, corners, frame)
    # frame = cv2.imread("test.png")

    # make the contrast bigger and adjust for aditional brightness 
        # see: https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html
        # see: https://stackoverflow.com/a/56909036
    alpha = 3.0 # contrast (1.0 - 3.0)
    beta = -100# brightness control (0-100)
    contrast = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    # contrast out all "light" noise
    # make the contrast bigger, but dont really reduce brightness 
    #   (washes out too light areas)
    
    # maximize brightness of paper
    lab = cv2.cvtColor(contrast, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0] # maximize brightness of paper
    # changed from: https://stackoverflow.com/a/72264323
    # ??: try HSV (https://stackoverflow.com/q/22588146, https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html,
        # https://stackoverflow.com/a/64258373, https://stackoverflow.com/a/72264323)

    # bluring the image for a smother image # !!
    blur = cv2.GaussianBlur(l_channel,(3,3),0) # to minimize the background noise

    # output = blur

    # use automatic thresholding: see https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html (Otsu's Binarization)
    t_min = 0 # exclude extremes
    t_max = 255
    ret, thresh = cv2.threshold(blur, t_min, t_max, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
    # use adaptive threshold to extract the contours
    # ret, thresh = cv2.threshold(thresh, t_min, t_max, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # MORPH -> to better "connect" the contours
    kernel_size=(7, 7)
    kernel = np.ones(kernel_size, dtype=np.float64)

    dilation = cv2.dilate(thresh, kernel) # not useful
    erotion = cv2.erode(thresh, kernel)
    closing = cv2.erode(dilation, kernel)
    opening = cv2.dilate(erotion, (1, 1))

    # detect edges (see: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
    c_min = 100
    c_max = 200
    canny = cv2.Canny(erotion, c_min, c_max)

    # output = erotion

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
    # print("max-contour", c.sum()) #> 5000?
    # NOTE: check if contours is empty
    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # determine second most extreme points (see: https://stackoverflow.com/a/60935067)      

    # draw the outline of the object, then draw each of the
    # extreme points, where the left-most is red, right-most
    # is green, top-most is blue, and bottom-most is teal
    # cv2.drawContours(frame, [c], -1, (0, 255, 255), 2)
    
    cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
    cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
    cv2.circle(frame, extTop, 8, (255, 0, 0), -1)
    cv2.circle(frame, extBot, 8, (255, 255, 0), -1)

    
    
    output = frame # NOTE: HERE
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
