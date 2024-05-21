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

OFFSET = 10
score = pyglet.text.Label(text="Score: 0", x=OFFSET, y=WINDOW_HEIGHT-OFFSET, batch=shape_batch)

MAX_ENEMY_COUNT = 5


video_id = 0
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# ARUCO - STUFF
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
    def __init__(self):
        self.has_transformed = False
        self.marker_ids = [0, 1, 2, 3]
        self.prev_transform = []

    # def mark_corners(self, ids:list[list[int]], corners, img, frame)

    # def transform_game_field_old(self, ids:list[list[int]], corners, img, frame):
    #     # ? if has id:0 & 2 -> can cal
    #     # print(corners)

    #     c = corners
    #     if len(ids) == 4:
    #         ids = ids.flatten()
    #         ids.sort()
    #         print(ids)

    #         if set(self.marker_ids) == set(ids): # check that only the actual markers got detected
    #             print("True")

    #             top_l = [c[ids[0]][0][0][0], c[ids[0]][0][0][1]] # id:0
    #             top_r = [c[ids[1]][0][1][0], c[ids[1]][0][1][1]] # id:1
    #             bot_r = [c[ids[2]][0][2][0], c[ids[2]][0][2][1]] # id:2
    #             bot_l = [c[ids[3]][0][3][0], c[ids[3]][0][3][1]] # id:3
    #             print(f"0: {top_l}, 1: {top_r}, 2: {bot_r}, 3: {bot_l}")

    #             # old_points = np.float32(np.array([ top_l, top_r, bot_r, bot_l ]))
    #             old_points = np.float32(np.array([top_r, top_l, bot_l, bot_r]))
    #             # [0, 1, 2, 3 ]
    #             new_points = np.float32(np.array([ [0, 0], [WINDOW_WIDTH, 0], [WINDOW_WIDTH, WINDOW_HEIGHT], [0, WINDOW_HEIGHT] ]))

    #             matrix = cv2.getPerspectiveTransform(old_points, new_points )
    #             frame = cv2.warpPerspective(frame, matrix, (img.width, img.height))
    #             img = cv2glet(frame, 'BGR')
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
        
        # return img

        # x = [0][0][corner][0]
        # y = [0][0][corner][1]

        # top_l = [c[0][0][0][0], c[0][0][0][1]]
        # top_r = [c[1][0][1][0], c[1][0][1][1]]
        # bot_r = [c[2][0][2][0], c[2][0][2][1]]
        # bot_l = [c[3][0][3][0], c[3][0][3][1]]

    # TODO: make better, i.e. for 3 markers
    def transform_game_field(self, ids:list[list[int]], corners, img, frame):
        # marker_ids = [3, 2, 0, 1] # HACK -> this is how they should be detected
        # marker_ids = [1, 0, 3, 2]
        marker_ids = [0, 1, 2, 3]
        ids = ids.flatten()
        # print(ids)
        ids.sort()

        # id's should always be in the same place, but arent detected like that
        
        # Keeps the previous transformation until markers have been reliably recalculated and found
        
        # TODO: just get max/min coordinates
        if len(ids) == 4 and (marker_ids == ids).all():
            # ids.sort() # TODO: test
            print("found all markers")
            markers = corners
            c = corners

            m_0 = markers[ids[0]][0][0] # x/y coordinates of top-left corner
            m_1 = markers[ids[1]][0][0]
            m_2 = markers[ids[2]][0][0]
            m_3 = markers[ids[3]][0][0]

            # TEST
            # store the orientation and corner coordinates of the playfield
            box = np.zeros(4, dtype="int8")
            # 0: top_left, 1: top_right, 2: bot_right, 3: bot_left

            # ms = [markers[0], markers[1], markers[2], markers[3]] # reshape?
            ms = [m_0, m_1, m_2, m_3]
            # get to top_l/bot_r corners using min/max
            idx_top_l = np.argmin(np.sum(ms, axis=1))
            idx_bot_r = np.argmax(np.sum(ms, axis=1))
            print("?", ms, "\n-->", idx_top_l, idx_bot_r)

            # box[0] = ms[idx_top_l]
            # box[2] = ms[idx_bot_r]
            box[0] = idx_top_l
            box[2] = idx_bot_r

            #ms.pop(idx_top_l)
            #ms.pop(idx_bot_r-1)
            # if idx_top_l > idx_bot_r:
            #     ms.pop(idx_top_l)
            #     ms.pop(idx_bot_r)    
            # else: 
            #     ms.pop(idx_bot_r)
            #     ms.pop(idx_top_l)
            ids = np.delete(ids, idx_top_l)
            ids = np.delete(ids, idx_bot_r)

            print("rest", ids)

            # get top_r (x> & y<) & get bot_l (x< & y>)
            if (ms[ids[0]][0] > ms[ids[1]][0]) and (ms[ids[0]][1] < ms[ids[1]][1]):
                idx_top_r = ids[0]
                idx_bot_l = ids[1]
            else:
                idx_top_r = ids[1]
                idx_bot_l = ids[0]

            box[1] = idx_top_r
            box[3] = idx_bot_l

            print("BOX:", box)
            
            # get coordinates of outside corners and arrange
            # 0: top_left, 1: top_right, 2: bot_right, 3: bot_left
            """
            m0= °[0] ----- [1]° =m1
                  |         |
            m3= .[3] ----- [2]. =m2
            """

            top_l = [c[box[0]][0][0][0], c[box[0]][0][0][1]]
            top_r = [c[box[1]][0][1][0], c[box[1]][0][1][1]]
            bot_l = [c[box[2]][0][2][0], c[box[2]][0][2][1]] # id:2
            bot_r = [c[box[3]][0][3][0], c[box[3]][0][3][1]]
            self.prev_transform = np.float32(np.array([top_l, top_r, bot_r, bot_l]))
            # Arrange corners in right orientation           
            # self.prev_transform = np.float32(np.array([ bot_l, bot_r, top_r, top_l]))

            
            # self.prev_transform = np.float32(np.array([ bot_l, bot_r, top_r, top_l]))

            
            # works for outside corners (with  marker_ids = [0, 1, 2, 3])
            # NOTE: sometimes markers get detected wrong and it doesn't transform right
            # top_l = [c[ids[0]][0][3][0], c[ids[0]][0][3][1]] # id:0
            # top_r = [c[ids[1]][0][2][0], c[ids[1]][0][2][1]] # id:1
            # bot_l = [c[ids[2]][0][0][0], c[ids[2]][0][0][1]] # id:2
            # bot_r = [c[ids[3]][0][1][0], c[ids[3]][0][1][1]]            
            # self.prev_transform = np.float32(np.array([ bot_l, bot_r, top_r, top_l]))

            # works
            # top_l = [c[0][0][0][0], c[0][0][0][1]] # id:0
            # top_r = [c[1][0][1][0], c[1][0][1][1]] # id:1
            # bot_l = [c[2][0][2][0], c[2][0][2][1]] # id:2
            # bot_r = [c[3][0][3][0], c[3][0][3][1]] # id:3

            # also works (is not outer corners)
            # top_l = [c[0][0][0][0], c[0][0][0][1]] # id:0
            # top_r = [c[1][0][1][0], c[1][0][1][1]] # id:1
            # bot_l = [c[2][0][3][0], c[2][0][3][1]] # id:2
            # bot_r = [c[3][0][2][0], c[3][0][2][1]] # id:3
            # self.prev_transform = np.float32(np.array([top_r, top_l, bot_l, bot_r]))

        if len(self.prev_transform) > 0: # keep transformation
            # old_points = np.float32(np.array([top_r, top_l, bot_l, bot_r]))
            old_points = self.prev_transform
            # new_points = np.float32(np.array([ [0, 0], [WINDOW_WIDTH, 0], [WINDOW_WIDTH, WINDOW_HEIGHT], [0, WINDOW_HEIGHT] ]))
            new_points = np.float32(np.array([[img.width, 0], [0, 0],
                                            [img.width, img.height], [0, img.height]]))
            # bc. of different coordinate systems

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

# exclude extremes
T_LOW = 50 
T_HIGH = 200
FINGER_RADIUS = 5
FINGER_COLOR = (0, 255, 0)
ENEMY_SIZE = 10

class Game():
    """Bundels the game stuff"""

    def __init__(self):
        self.enemies = []
        self.finger = pyglet.shapes.Circle(0, 0, RADIUS,
                                         color=FINGER_COLOR, batch=shape_batch)
        self.score = 0

    # TODO:
        # Detect hand -> then detect pointer and thumb
        # Draw circle at those positions to indicate + use them for detection
    def detect_hand(self, frame):
        """Detect the hand and it's area for collision"""
        # print("now detecting hand")

            # maximize brightness of paper
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0] # maximize brightness of paper
        # changed from: https://stackoverflow.com/a/72264323

        # bluring the image for a smother image # !!
        blur = cv2.GaussianBlur(l_channel,(7,7),0)

        # use automatic thresholding: see https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html (Otsu's Binarization)
        _, thresh = cv2.threshold(blur, T_LOW, T_HIGH, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # MORPH
        kernel_size=(3, 3)
        kernel = np.ones(kernel_size, dtype=np.float64)

        dilation = cv2.dilate(thresh, kernel)
        closing = cv2.erode(dilation, kernel)
        canny = cv2.Canny(closing, 200, 200)

         # detect contours 
        # (see: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
        # (see: https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/)
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # get two righ/left most points

        # see: https://pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
        c = max(contours, key=cv2.contourArea)

        # determine the most extreme points along the contour
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        
        # determine from which direction the hand is coming 
        # the higher (y) point should be the pointer finger
        if extRight[1] > extLeft [1]:
            x = extRight[0]
            y = extRight[1]
        else:
            x = extLeft[0]
            y = extLeft[1]

        # move the marker for the pointer-finger
        self.finger.x = WINDOW_WIDTH-x
        self.finger.y = y


    def create_enemy(self, delta_time):
        """Create a handfull of enemies to interact with"""
        if len(self.enemies) != MAX_ENEMY_COUNT:
            y = WINDOW_HEIGHT - random.randint(0, WINDOW_HEIGHT)
            enemy = pyglet.shapes.Rectangle(0, y, ENEMY_SIZE, ENEMY_SIZE,
                                         color=ENEMY_COLOR, batch=shape_batch)
            self.enemies.append(enemy)            

    def update_enemies(self, delta_time):
        for enemy in self.enemies:
            self.check_collision(enemy)
            enemy.x += SPEED


    def check_collision(self, enemy):
        # check out of bounds
        if enemy.x > WINDOW_WIDTH:
            self.enemies.remove(enemy)
            # print("oob", len(self.enemies))

        # TODO: check hand collision
        # check if an emeny collides with the bounding box of the finger marker
        f_x = self.finger.x
        f_y = self.finger.y

        if enemy.x > f_x - RADIUS and enemy.x < f_x + RADIUS:
            if enemy.y > f_y - RADIUS and enemy.y < f_y + RADIUS:
                self.update_score()
                self.enemies.remove(enemy)

    def update_score(self):
        self.score += 1
        score.text=f"Score: {self.score}"


SPEED = 10
RADIUS = 10
ENEMY_COLOR = (255, 0, 0)


# ----- DRAW - EVENT ----- #

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)
rect = pyglet.shapes.Rectangle(100, 100, 10, 10, (255, 0, 0), batch=shape_batch)
# rect2 = pyglet.shapes.Rectangle(100, 100, 10, 10, (0, 255, 0), batch=shape_batch)
# rect3 = pyglet.shapes.Rectangle(100, 100, 10, 10, (0, 0, 255), batch=shape_batch)
# rect4 = pyglet.shapes.Rectangle(100, 100, 10, 10, (255, 0, 255), batch=shape_batch)
# img = pyglet.image.load("../image_extraction/sample_image.jpg")
pf = Playfield()
game = Game()

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
        # print(corners)
        # print(ids.flatten)
        # r ={ int(ids[0]): corners[0].reshape(4,2)}
        # print(r)

        rect.x = corners[0][0][0][0] # top left corner
        rect.y = WINDOW_HEIGHT - corners[0][0][0][1]
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

    # DO GAME HERE
    if pf.has_transformed:
        # start only after transforming playfield
        game.detect_hand(frame)

        # Foreground drawing from here: ↓
        shape_batch.draw()


clock.schedule_interval(game.create_enemy, 2)
clock.schedule_interval(game.update_enemies, 0.1)

""" NOTE:
? do the transformation calculation only every 30seconds (less shake)
? save the transformed matrix just use the "old" one
"""


# ----- RUN ----- #

if __name__ == "__main__":
    pyglet.app.run()