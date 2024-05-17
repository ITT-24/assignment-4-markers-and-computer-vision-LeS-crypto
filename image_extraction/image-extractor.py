import cv2
import argparse
import numpy as np
from datetime import datetime

"""
- load & display image (with opencv)
- click image to select 4 points
    - selected region is extracted and warped to a rectangle
    - display result
- ESC = discard changes + start over (= old image)
- S   = saves the image (in result view)
- CMD Line Params:
    - Path to input file
    - Path to output destination
    - resolution of image result

- [x] - (1P) The image is successfully loaded and displayed.
- [x] - (1P) Selecting the corner points works and there is visual feedback for the user.
- [x] - (2P) Perspective transformation to the target resolution works.
- [x] - (1P) Command line parameters and shortcuts work.
"""
# TODO: test

WINDOW_NAME = "Image Extractor"

def parse_cmd_input():
    """
    Read and parse the command line parameters.
    See: https://docs.python.org/3/library/argparse.html
    """

    # init path variables
    source = None
    destination = None
    res = None

    parser = argparse.ArgumentParser( 
        prog="image-extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Exctracts and warps a selection from an image to get a straightend result.",
        epilog="""----- Keyboard Shortcuts -----\n
        ESC \t - Reset the image to its original version
        s \t - save the result in the specified path
        q \t - quit the application (without saving)
        """
    )

    parser.add_argument('-s', '--source', type=str, metavar='',
                        default="sample_image.jpg",
                        required=False,
                        action="store",
                        help="""The path to the image file you want to use. 
                            Defaults to the included 'sample_image.jpg' in the current folder""")
    parser.add_argument('-d', '--destination', type=str, metavar='',
                        default=None, # ?? current directory
                        required=False,
                        action="store",
                        help="""The path to the directory you want to save your result image in.
                            Defaults to a current directory.\n
                            For example: ../image_results""")
    parser.add_argument('-r', '--resolution', type=int, metavar='',
                        required=False,
                        action="store",
                        nargs=2,
                        help="""Set a resolution to store the image in this order: width height.
                            Defaults to the same resolution as the source image.""")
    
    args = parser.parse_args()
    print(args)

    if args.source:
        source = args.source
        print("get image:", source)
        
    if args.destination:
        destination = args.destination
        print("save result to directory:", destination)
    
    if args.resolution:
        res = [args.resolution[0], args.resolution[1]]

    return source, destination, res


class Extractor():
    """Bundles the methods to used to display, transform and save the image"""

    def __init__(self, source:str, destination:str, resolution:list[int]|None) -> None:
        self.source = source
        self.destination = destination
        self.resolution = resolution
        self.clicks = []

        # read and display image
        self.img = cv2.imread(source)
        cv2.namedWindow(WINDOW_NAME)
        
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        pass

    def show_window(self):
        cv2.imshow(WINDOW_NAME, self.img)

    def mouse_callback(self, event, x, y, flags, params):
        # see: opencv_click.py
        global img

        if event == cv2.EVENT_LBUTTONDOWN:
            img = cv2.circle(self.img, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow(WINDOW_NAME, img)
            self.clicks.append([x, y])

            # check if image can be transformed
            if len(self.clicks) == 4:
               self.undistort_selection()

    def undistort_selection(self): # TODO
        """Transforms the image to the specified resolution (default = same res as source)"""

        img = cv2.imread(source) # read img again/ or copy to remove point markers
        # img = self.img.copy()

        if self.resolution == None:
            self.resolution = [self.img.shape[0], self.img.shape[1]]
            print("saveing with original resolution", self.resolution)


        # TRANSFORMATION - see: transformation.ipynb
        old_points = np.float32(np.array(self.clicks))
        # np.float32(np.array([[200, 200], [1700, 400], [1300, 1000], [ 400, 900]]))
        res = self.resolution
        height = res[0]
        width = res[1]
        new_points = np.float32(np.array([ [0, 0], [width, 0], [width, height], [0, height] ]))
        # destination = np.float32(np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]))

        matrix = cv2.getPerspectiveTransform(old_points, new_points )
        self.img = cv2.warpPerspective(img, matrix, (width, height))

        self.clicks = [] # reset click, if user wants further undistortion

        """ Visual guide
            0, 0        1 ---- 2    width, 0
                        |      |
            0, height   4 ---- 3    width, height
        """

    def reset_image(self):
        """Rereads the image into the application"""

        self.img = cv2.imread(source)
        self.clicks = [] # reset
        print("resetting image")


    def save_image(self):
        """
        Save the processed image to the specified directory. 
        New filename includes a simple identifier to not override things
        """
        
        identifier = f"{datetime.now().minute}-{datetime.now().second}" # to not override imgs
        filename = f"extr_{identifier}_{self.source}"
        filepath = None
        if self.destination == None:
            filepath = filename
        else: 
            filepath= f"{self.destination}/{filename}"
        print("saveing image to", filepath)
        has_saved = cv2.imwrite(filepath, self.img)
        print(has_saved)



# ----- RUN ----- #

if __name__ == "__main__":

    source, destination, res = parse_cmd_input()
    extractor = Extractor(source, destination, res)

    # PROGRAMM LOOP
    while(True):
        extractor.show_window()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            extractor.save_image()
        elif key == 27: # ESC
            extractor.reset_image()
        # close the window with "window-x-button" (mouse)
        elif cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            # see: https://stackoverflow.com/a/63256721
            break
    cv2.destroyAllWindows()