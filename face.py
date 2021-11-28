import cv2
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
import urllib.request
import os
import numpy as np
import PIL.Image

class Face:
    classifier = None
    @classmethod
    def get_face_bboxes(cls, pixels):
        HAAR = "haarcascade_frontalface_default.xml"
        if not os.path.isfile(HAAR):
            HAAR_URL="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(HAAR_URL, "haarcascade_frontalface_default.xml")
            assert(os.path.isfile(HAAR)), "Failed to download %s" % HAAR_URL
        if not cls.classifier:
            cls.classifier = CascadeClassifier(HAAR)
        bboxes = cls.classifier.detectMultiScale(pixels)
        return bboxes

    @classmethod
    def get_face_bboxes_from_pil_image(cls, pil_image):
        opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return cls.get_face_bboxes(opencvImage)
    @classmethod
    def get_face_bboxes_from_file(cls, image_file):
        pixels = imread(image_file)
        return cls.get_face_bboxes(pixels)

def main():
    test_file = "images/278.jpg"
    pixels = imread(test_file)
    # print(pixels)
    bboxes = Face.get_face_bboxes(pixels)
    print(bboxes)
    for box in bboxes:
        x, y, width, height = box
        x2, y2 = x + width, y + height
        rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 1)

    # show the image
    imshow('faces detection', pixels)
    # keep the window open until we press a key
    waitKey(0)
    # close the window
    destroyAllWindows()

if __name__ == "__main__":
    main()