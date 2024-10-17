import os
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

court_img_path = os.path.join(parent_dir, 'output.jpg')
court_img = cv2.imread(court_img_path)


def get_hsv_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  
        pixel_value = court_img[y, x]

        pixel_hsv = cv2.cvtColor(np.uint8([[pixel_value]]), cv2.COLOR_BGR2HSV)[0][0]

        print(f"Clicked Pixel BGR: {pixel_value}, HSV: {pixel_hsv}")

        lower_bound = np.array([pixel_hsv[0] - 10, pixel_hsv[1] - 40, pixel_hsv[2] - 40])
        upper_bound = np.array([pixel_hsv[0] + 10, pixel_hsv[1] + 40, pixel_hsv[2] + 40])

        print(f"HSV Lower Bound: {lower_bound}")
        print(f"HSV Upper Bound: {upper_bound}")

        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

        cv2.imshow("Mask", mask)

hsv_img = cv2.cvtColor(court_img, cv2.COLOR_BGR2HSV)

cv2.imshow('Court Image', court_img)
cv2.setMouseCallback('Court Image', get_hsv_value)

cv2.waitKey(0)
cv2.destroyAllWindows()