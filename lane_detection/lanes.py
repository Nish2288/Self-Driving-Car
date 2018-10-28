# ************* Simple lane finder ********************

import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_coordinates(line_coordinates, image):
    slope,intercept = line_coordinates
    y1 = image.shape[0]
    y2 = int(y1*3/5)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return(np.array([x1, y1, x2, y2]))


def avg_slope_intercept(image, lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        for x1, y1, x2, y2 in line:
            parameters=np.polyfit((x1, x2), (y1, y2), 1)
            slope=parameters[0]
            intercept=parameters[1]
            if slope<0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average=np.average(left_fit, axis=0)
    right_fit_average=np.average(right_fit, axis=0)

    left_line=get_coordinates(left_fit_average, image)
    right_line=get_coordinates(right_fit_average, image)
    return(np.array([left_line, right_line]))    


def display_lines(image, lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2=line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image


cap=cv2.VideoCapture('lane_detection/road.mp4')
while(cap.isOpened()):
    _, image=cap.read()
    #image=cv2.imread('road_image.jpg')
    lane_image=np.copy(image)
    gray=cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray, (5, 5), 0)
    canny=cv2.Canny(blur, 50, 150)

    mask=np.zeros_like(canny)
    height=image.shape[0]
    poly=np.array([[(200,height), (1100,height), (550,250)]])
    cv2.fillPoly(mask, poly, 255)

    masked_image=cv2.bitwise_and(mask,canny)

    lines=cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    avg_line=avg_slope_intercept(lane_image,lines)

    line_image=display_lines(lane_image,avg_line)



    combo_image=cv2.addWeighted(lane_image,0.8,line_image,1,1)
    cv2.imshow('Result',combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()