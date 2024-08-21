
import numpy as np
from PIL import Image
from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    crop            = False
    count           = False
    test_interval   = 100

    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()