# 01 - image-extractor.py
- without specifiying parameters the application uses the included `sample_image.jpg`, transforms the selection into the resolution of the original image, and then saves it to the same folder.
- all three parameters can be set with commandline parameters
  - `-s` the image file with path
  - `-d` the destination folder (extracted images gets named automatically)
  - `-r` resolution
- if a parameter is not set it will default as clarified above
- for further information use the `-h` command or see below:

```bash
py .\image-extractor.py  # default/exemplatory use
py .\image-extractor.py -h # see help
py .\image-extractor.py -s path/to_image.jpg -d path/to/destination/dir -r 400 200

```

# 02 - AR-game.py
- start with parameter for webcam, i.e. `AR-game.py 0`
- detects the four markers (IDs: 0, 1, 2, 3) and transforms the playfield, then starts the game. (Orientation shouldn't matter)
- Detects the contour of the hand and marks the highest point
  - detects the pointer finger, if you're pointing (👈☝️)
  - displays a crosshair where the collision will be detecting
  - testing for collision was done in: `hand_detect.py`, bc. it's easier to see/use in cv2 than pyglet
- Destroy the enemies (rectangles) coming toward you, by overlapping the crosshair with the enemies.
- Quit with `q`

### Sprite Sources
- `crosshair-red.png`: By Aris Katsaris - Own work, CC0, https://commons.wikimedia.org/w/index.php?curid=14674694

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/pR29BhE5)
