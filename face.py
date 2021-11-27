import cv2
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
import urllib
import os

classifier = None

class Face:

    @staticmethod
    def get_faces_bboxes(image_file):
        HAAR = "haarcascade_frontalface_default.xml"
        global classifier
        if not os.path.isfile(HAAR):
            HAAR_URL="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.URLopener().retrieve(HAAR, "haarcascade_frontalface_default.xml")
            assert(os.path.isfile(HAAR)), "Failed to download %s" % HAAR_URL
            classifier = CascadeClassifier(HAAR)
        assert(classifier)
        bboxes = classifier.detectMultiScale(HAAR)
        # return [(b[0], b[1], b[0]+b[2], b[1]+b[3]) for b in bboxes]
        return bboxes

        # draw a rectangle over the pixels
        rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 1)
    # show the image
    imshow(pixels)
    # keep the window open until we press a key
    waitKey(0)
    # close the window
    destroyAllWindows()


def main():
    pass

if __name__ == "__main__":
    main()