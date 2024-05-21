import cv2
from cv2 import aruco
import numpy as np
import pyglet
from pyglet import clock
from PIL import Image
import sys
import random

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

IDEA: draw with palm/pointer finger 
IDEA: eat enemies with hand (just point at it)
IDEA: create sliders to adjust detection
"""

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
shape_batch = pyglet.graphics.Batch()

OFFSET = 20
score = pyglet.text.Label(text="Score: 0", x=OFFSET, y=WINDOW_HEIGHT-OFFSET,
                          color=(0, 0, 0, 255), batch=shape_batch)

MAX_ENEMY_COUNT = 5
HAND_RADIUS = 50
HAND_COLOR = (0, 100, 255, 80)
ENEMY_SIZE = 20
SPEED = 10
ENEMY_COLOR = (255, 0, 0)

# INIT VIDEO FEED
video_id = 0
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# INIT ARUCO MARKERS
aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params =  aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

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
    """
    Takes a paper with 4 AruCo markers and transforms the paper to 
    fit the video frame
    """

    def __init__(self):
        self.has_transformed = False
        self.marker_ids = [0, 1, 2, 3]
        self.prev_transform = []

    def transform_game_field(self, ids:list[list[int]], corners, img, frame):
        """
        Parse the coordinates of the markers to determine the arrangement of the
        markers on the board and transformes them to fit the webcam.
        (The markers should always be in the same spot, but this makes for
        a more robust detection.)
        Keeps the the orientation until all expected markers have been detected.
        Returns the transformed image
        """
        
        marker_ids = self.marker_ids # expected markers
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

            if idx_top_l > idx_bot_r: # to prevent out of bounds
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
            # arrange the new points according to the diff coord system
            new_points = np.float32(np.array([[img.width, 0], [0, 0],
                                            [img.width, img.height], [0, img.height]]))

            matrix = cv2.getPerspectiveTransform(old_points, new_points )
            frame = cv2.warpPerspective(frame, matrix, (img.width, img.height))
            img = cv2glet(frame, 'BGR') # parse into pyglet image
            self.has_transformed = True

        return img, frame


class Game():
    """Bundels the game stuff"""

    def __init__(self):
        self.enemies = []
        self.finger = pyglet.shapes.Circle(0, 0, HAND_RADIUS,
                                         color=HAND_COLOR, batch=shape_batch)
        self.score = 0

    # TODO:
        # Detect hand -> then detect pointer and thumb
        # Draw circle at those positions to indicate + use them for detection
    def detect_hand(self, frame):
        """
        Detect the hand and it's area to use for collision
        See also: hand_detect.py (testing ground for cv2 detection)
        """
        # print("now detecting hand")

        # make the contrast bigger to "wash out" lighter areas
        # see: https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html
        # see: https://stackoverflow.com/a/56909036
        alpha = 3.0 # contrast (1.0 - 3.0)
        beta = -100 # brightness control (0-100)
        contrast = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # maximize brightness of paper
        lab = cv2.cvtColor(contrast, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0] # maximize brightness of paper
        # changed from: https://stackoverflow.com/a/72264323

        # blur for a smother image
        blur = cv2.GaussianBlur(l_channel,(3,3),0)

        # use automatic thresholding: see https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html (Otsu's Binarization)
        t_min = 0 # exclude extremes
        t_max = 255
        ret, thresh = cv2.threshold(blur, t_min, t_max, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
        # use adaptive threshold to then better extract the contours
        
        # MORPH -> to better "connect" the contours
        kernel_size = (7, 7)
        kernel_small = (1, 1)
        kernel = np.ones(kernel_size, dtype=np.float64)

        erotion = cv2.erode(thresh, kernel)
        opening = cv2.dilate(erotion, kernel_small)

        # detect edges (see: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
        c_min = 100
        c_max = 200
        canny = cv2.Canny(erotion, c_min, c_max)

        # detect contours 
        # (see: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
        # (see: https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/)
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # get two righ/left most points
        # see: https://pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
        if contours != (): # max() iterable element is empty
            c = max(contours, key=cv2.contourArea)

            # determine the most extreme points along the contour
            extLeft  = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop   = tuple(c[c[:, :, 1].argmin()][0])
            extBot   = tuple(c[c[:, :, 1].argmax()][0])
            
            # determine from which direction the hand is coming 
            # the higher (y) point should be the pointer finger
            if extRight[0] > extLeft [0]:
                x = extRight[0]
                y = extRight[1]
            else:
                x = extLeft[0]
                y = extLeft[1]

            # rect = cv2.boundingRect(c)
            # x,y,w,h = rect
            # x = x + (w/2)
            # y = y + (h/2) 
        else:
            x = self.finger.x 
            y = self.finger.y 

        # move the marker for the pointer-finger
        # only if difference is big (so less jitter)
        # if np.abs(x-self.finger.x) > 10 and abs(y-self.finger.y) > 10:
        self.finger.x = x
        self.finger.y = WINDOW_HEIGHT - y
        # self.finger.radius = (w+h)/2


    def create_enemy(self, delta_time):
        if len(self.enemies) != MAX_ENEMY_COUNT:
            y = WINDOW_HEIGHT - random.randint(0, WINDOW_HEIGHT)
            enemy = pyglet.shapes.Rectangle(0, y, ENEMY_SIZE, ENEMY_SIZE,
                                         color=ENEMY_COLOR, batch=shape_batch)
            self.enemies.append(enemy)            

    def update_enemies(self, delta_time):
        for enemy in self.enemies:
            self.check_collision(enemy)
            enemy.x += SPEED

    def measure_distance(self, x1, y1, x2, y2):
        """via: pyglet_click.py AS-demo"""
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distance

    def check_collision(self, enemy):
        """Check if the enemies collied with the hand or with the window border"""
        # check out of bounds
        if enemy.x > WINDOW_WIDTH:
            self.enemies.remove(enemy)

        # check collision with "hand"
        f_x = self.finger.x
        f_y = self.finger.y

        distance = self.measure_distance(enemy.x, enemy.y, f_x, f_y)
        if distance < self.finger.radius:
            self.update_score()
            self.enemies.remove(enemy)

    def update_score(self):
        self.score += 1
        score.text=f"Score: {self.score}"


# ----- DRAW - EVENT ----- #

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)
pf = Playfield()
game = Game()

@window.event
def on_draw():
    window.clear()
    ret, frame = cap.read()
    img = cv2glet(frame, 'BGR')

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the frame
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    if ids is not None:
        img, frame = pf.transform_game_field(ids, corners, img, frame) 

    img.blit(0, 0, 0) # = Background

    # GAME ↓ 
    if pf.has_transformed: # start only after transforming playfield
        game.detect_hand(frame)

        # Foreground drawing from here: ↓
        shape_batch.draw()

# UPDATE ENEMIES
clock.schedule_interval(game.create_enemy, 2)
clock.schedule_interval(game.update_enemies, 0.1)


# ----- RUN ----- #

if __name__ == "__main__":
    pyglet.app.run()