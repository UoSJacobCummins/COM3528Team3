import sys
sys.path.append('..')
from sensors import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import entropy_percept as ep
from plots_util import *

class PerceptionController:
    def __init__( self, robot ):
        
        self.images = None
        self.audio = 0

    def step( self, im, h, t, au ):
        self.images = im
        

    def plots( self ):
        if self.images is None:
            return

        im = cv2.cvtColor(self.images[0], cv2.COLOR_BGR2RGB)

        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 50
        params.maxThreshold = 200

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 20

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.1

        # color
        params.filterByColor = True
        params.blobColor = 255

        detector = cv2.SimpleBlobDetector_create( params)

        # Detect blobs.
        keypoints = detector.detect(im)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(1)

        

        
