import cv2
from cv2 import aruco
import numpy as np
import pyglet
from PIL import Image
import sys

"""
- read out webcam image
- extract region btw. AruCo markers (on board)
    - & transform to rectangle with same resolution as webcam (any)
- display image in pyglet app
- create game
    - game mechanics include hand gestures (to destoy targets or move things)
    - i.e. detect hand/fingers (e.g. contrast/color-diff)

- [ ] - (2P) The region of interest is detected, extracted, transformed, and displayed.
- [ ] - (4P) Objects (such as fingers) in the region of interest are tracked reliably and interaction with game objects works.
- [ ] - (2P) Game mechanics work and (kind of) make sense.
- [ ] - (1P) Performance is ok.
- [ ] - (1P) The program does not crash   
"""

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)


video_id = 0
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# ARUCO - STUFF
aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params =  aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# Define the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()

# converts OpenCV image to PIL image and then to pyglet texture
# https://gist.github.com/nkymut/1cb40ea6ae4de0cf9ded7332f1ca0d55
def cv2glet(img,fmt):
    '''Assumes image is in BGR color space. Returns a pyimg object'''
    if fmt == 'GRAY':
      rows, cols = img.shape
      channels = 1
    else:
      rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels*cols
    pyimg = pyglet.image.ImageData(width=cols, 
                                   height=rows, 
                                   fmt=fmt, 
                                   data=raw_img, 
                                   pitch=top_to_bottom_flag*bytes_per_row)
    window.set_size(width=cols, height=rows) #??
    return pyimg


class Playfield():
    def __init__(self):
        self.has_transformed = False
        self.marker_ids = [0, 1, 2, 3]
        self.prev_transform = []

    # def mark_corners(self, ids:list[list[int]], corners, img, frame)

    def transform_game_field_old(self, ids:list[list[int]], corners, img, frame):
        # ? if has id:0 & 2 -> can cal
        # print(corners)

        c = corners
        if len(ids) == 4:
            ids = ids.flatten()
            ids.sort()
            print(ids)

            if set(self.marker_ids) == set(ids): # check that only the actual markers got detected
                print("True")

                top_l = [c[ids[0]][0][0][0], c[ids[0]][0][0][1]] # id:0
                top_r = [c[ids[1]][0][1][0], c[ids[1]][0][1][1]] # id:1
                bot_r = [c[ids[2]][0][2][0], c[ids[2]][0][2][1]] # id:2
                bot_l = [c[ids[3]][0][3][0], c[ids[3]][0][3][1]] # id:3
                print(f"0: {top_l}, 1: {top_r}, 2: {bot_r}, 3: {bot_l}")

                # old_points = np.float32(np.array([ top_l, top_r, bot_r, bot_l ]))
                old_points = np.float32(np.array([top_r, top_l, bot_l, bot_r]))
                # [0, 1, 2, 3 ]
                new_points = np.float32(np.array([ [0, 0], [WINDOW_WIDTH, 0], [WINDOW_WIDTH, WINDOW_HEIGHT], [0, WINDOW_HEIGHT] ]))

                matrix = cv2.getPerspectiveTransform(old_points, new_points )
                frame = cv2.warpPerspective(frame, matrix, (img.width, img.height))
                img = cv2glet(frame, 'BGR')
        #     self.has_transformed = True
        #     # can calculate 4th point

            # TODO: sort by top_left corner [x][0][0][0/1]
                # 0 = sum x/y = smallest & 3= sum x/y = biggest
                # 1 = left x < 2 left x
                # then get the appropriate outer corners
            
            
            # for i in range(0,len(ids)):
            #     print(np.min(c[i][0][0][0]))

        # for i in len(0, c.)
# 
        # # top_l = smallest [x][0][0][0/1]
        # # top_r = smalles  [x][0][0][0] + biggest num for [x][0][0][1]
        # c = corners
        # if len(ids) == 4: # >= 3: -> TODO
        #     # ids are not in the same order, we need 0, 1, 2, 3
        #     top_l = [c[0][0][0][0], c[0][0][0][1]] # id:0
        #     top_r = [c[1][0][0][0], c[1][0][0][1]] # id:1
        #     bot_r = [c[2][0][0][0], c[2][0][0][1]] # id:2
        #     bot_l = [c[3][0][0][0], c[3][0][0][1]] # id:3
        #     print(f"0: {top_l}, 1: {top_r}, 2: {bot_r}, 3: {bot_l}")

        #     # old_points = np.float32(np.array([ top_l, top_r, bot_r, bot_l ]))
        #     old_points = np.float32(np.array([top_r, top_l, bot_r, bot_l]))
        #     # [0, 1, 2, 3 ]
        #     new_points = np.float32(np.array([ [0, 0], [WINDOW_WIDTH, 0], [WINDOW_WIDTH, WINDOW_HEIGHT], [0, WINDOW_HEIGHT] ]))

        #     matrix = cv2.getPerspectiveTransform(old_points, new_points )
        #     frame = cv2.warpPerspective(frame, matrix, (img.width, img.height))
        #     img = cv2glet(frame, 'BGR')
        #     self.has_transformed = True
        #     # can calculate 4th point
        
        return img

        # x = [0][0][corner][0]
        # y = [0][0][corner][1]

        # top_l = [c[0][0][0][0], c[0][0][0][1]]
        # top_r = [c[1][0][1][0], c[1][0][1][1]]
        # bot_r = [c[2][0][2][0], c[2][0][2][1]]
        # bot_l = [c[3][0][3][0], c[3][0][3][1]]

    # TODO: make better, i.e. for 3 markers
    def transform_game_field(self, ids:list[list[int]], corners, img, frame):
        marker_ids = [3, 0, 2, 1] # HACK
        ids = ids.flatten()
        
        # Keeps the previous transformation until markers have been reliably recalculated and found
        
        if len(ids) == 4 and (marker_ids == ids).all():
            print("found all markers")
            c = corners

            top_l = [c[0][0][0][0], c[0][0][0][1]] # id:0
            top_r = [c[1][0][1][0], c[1][0][1][1]] # id:1
            bot_l = [c[2][0][2][0], c[2][0][2][1]] # id:2
            bot_r = [c[3][0][3][0], c[3][0][3][1]] # id:3
            # ?? get the outer corner

            self.prev_transform = np.float32(np.array([top_r, bot_r, bot_l, top_l]))
            # rect.x = top_l[0] # top left corner
            # rect.y = top_l[1] # bc. flipped coord system 
            # rect2.x = bot_l[0] # top left corner
            # rect2.y = bot_l[1] # bc. flipped coord system 
            # rect3.x = bot_r[0] # top left corner
            # rect3.y = bot_r[1] # bc. flipped coord system 
            # rect4.x = top_r[0] # top left corner
            # rect4.y = top_r[1] # bc. flipped coord system 

        if len(self.prev_transform) > 0: # keep transformation
            # old_points = np.float32(np.array([top_r, top_l, bot_l, bot_r]))
            old_points = self.prev_transform
            new_points = np.float32(np.array([ [0, 0], [WINDOW_WIDTH, 0], [WINDOW_WIDTH, WINDOW_HEIGHT], [0, WINDOW_HEIGHT] ]))
            # new_points = np.float32(np.array([[img.width, 0], [0, 0], 
            #                                   [0, img.height], [img.width, img.height] ]))

            matrix = cv2.getPerspectiveTransform(old_points, new_points )
            frame = cv2.warpPerspective(frame, matrix, (img.width, img.height))
            img = cv2glet(frame, 'BGR')
            self.has_transformed = True
            pass

        # get top_left corner of every marker
            #  1 top_left = min sum of corner x/y
            #  3 bot_right = max sum of corner x/y
            #  0 top_right = x > top_left, y ~ bot_right + x > bot_left
            #  2 bot_left = x ~ bot_right, y  > top_right + x > top_right
            # if 1 missing, just use 0_x, 2_y or use self.prev_transform
        return img

""" ??
    // CAM
        m1= °[1] ----- [0]° =m4
              |         |
        m2= .[2] ----- [3]. =m3

    // IRL
        m0= °[0] ----- [1]° =m1
              |         |
        m3= .[3] ----- [2]. =m2
    corner = corners[0][marker][corner][x/y]
    coordinte => [0] = x, [1] = y 
"""

# ----- DRAW - EVENT ----- #

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)
shape_batch = pyglet.graphics.Batch()
rect = pyglet.shapes.Rectangle(100, 100, 10, 10, (255, 0, 0), batch=shape_batch)
rect2 = pyglet.shapes.Rectangle(100, 100, 10, 10, (0, 255, 0), batch=shape_batch)
rect3 = pyglet.shapes.Rectangle(100, 100, 10, 10, (0, 0, 255), batch=shape_batch)
rect4 = pyglet.shapes.Rectangle(100, 100, 10, 10, (255, 0, 255), batch=shape_batch)
# img = pyglet.image.load("../image_extraction/sample_image.jpg")
pf = Playfield()

@window.event
def on_draw():
    window.clear()
    ret, frame = cap.read()
    # frame = cv2.flip(frame, 1) # flip the frame (only do at the end)
    # alpha = 1.5
    # beta = 0
    # frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    # brightness and contrast (https://stackoverflow.com/a/58211607)

    img = cv2glet(frame, 'BGR')


    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the frame
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    if ids is not None:
        # if len(ids) == 4:
            # rect.x = corners[0][0][0][0] # top left corner
            # rect.y = WINDOW_HEIGHT - corners[0][0][0][1] # bc. flipped coord system 
            # rect2.x = corners[1][0][0][0] # top left corner
            # rect2.y = WINDOW_HEIGHT - corners[1][0][0][1] # bc. flipped coord system 
            # rect3.x = corners[2][0][0][0] # top left corner
            # rect3.y = WINDOW_HEIGHT - corners[2][0][0][1] # bc. flipped coord system 
            # rect4.x = corners[3][0][0][0] # top left corner
            # rect4.y = WINDOW_HEIGHT - corners[3][0][0][1] # bc. flipped coord system 
        # print(ids, "\n")
        img = pf.transform_game_field(ids, corners, img, frame) 
        # TODO: check orientation -> top corner needs to be outside
        # or rather, get the most outside+lowest/highest corner

    img.blit(0, 0, 0) # = Background

    # Foreground drawing from here: ↓
    shape_batch.draw()

""" NOTE:
? do the transformation calculation only every 30seconds (less shake)
? save the transformed matrix just use the "old" one
"""


# ----- RUN ----- #

if __name__ == "__main__":
    pyglet.app.run()