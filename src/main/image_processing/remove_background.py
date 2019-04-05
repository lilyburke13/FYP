import cv2
import numpy as np
import os
from plantcv import plantcv as pcv


DATA_DIR=''

for image in os.listdir(DATA_DIR):
    # Read Image
    image = cv2.imread(DATA_DIR + "/" + image)

    # Prompt the user to select the region of interest
    roi = cv2.selectROI(image)

    # Collect the cropped image
    cropped_img = image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

    # Convert RGB to HSV and extract saturation channel
    # The HSV value can be changed to be h, s, or v depending on the colour of the flower
    saturation_img = pcv.rgb2gray_hsv(cropped_img, 'h')

    # Threshold the saturation img
    # Depending on the output of the saturation image, the value can be light or dark
    # Light or dark is what defines what part of the image its to be removed
    saturation_thresh = pcv.threshold.binary(saturation_img, 85, 255, 'light')

    # Apply median blur
    saturation_mblur = pcv.median_blur(saturation_thresh, 5)

    # Convert RGB to LAB and extract blue channel
    # Like the HSV function this can be l, a, or b depending on the colour of the flower
    blue_channel_img = pcv.rgb2gray_lab(cropped_img, 'l')
    blue_channel_cnt = pcv.threshold.binary(blue_channel_img, 160, 255, 'light')

    # Join thresholded saturated and blue channel imgs
    blue_saturated = pcv.logical_or(saturation_mblur, blue_channel_cnt)

    # Apply black colour mask
    masked = pcv.apply_mask(cropped_img, blue_saturated, 'black')

    # Save image
    cv2.imwrite('tmp/'+image, masked)
