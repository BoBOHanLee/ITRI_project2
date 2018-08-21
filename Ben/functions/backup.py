import cv2
import numpy as np
import pandas as pd

def Inhence_and_threshod(img):
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(14, 14))
    cl = clahe.apply(img)
    # lowpass filter
    gaussian = cv2.GaussianBlur(cl, (5, 5), 1)
    # THRESH
    __, th = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY++ cv2.THRESH_OTSU)

    return th

def inner_fill(th):
    # 接滿
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    mor = cv2.dilate(th, kernel)
    # floodfill external's area
    image1, contours, hierarchy = cv2.findContours(
        mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mor, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    # erode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mor = cv2.erode(mor, kernel)
    #opening
    mor = cv2.morphologyEx(mor, cv2.MORPH_OPEN, kernel)

    return mor

