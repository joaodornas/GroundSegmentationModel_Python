
import cv2 as cv

import globals

#/////////////////////////////////////////////////////////////////////////
#//SHOW RESULTS
#/////////////////////////////////////////////////////////////////////////

def plot():

    #red = [0,0,255]

    # Change one pixel
    #image[10,5]=red

    img = cv.imread(globals.imageRunning)

    cv.imshow('Window',img)

    cv.waitKey(0)