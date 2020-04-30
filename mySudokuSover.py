
import cv2 as cv
import argparse
from matplotlib import pyplot as plt
import numpy as np
import itertools
import sys
from skimage import morphology
from skimage.util import invert
import math

def scaleImg(img):
    width = int(img.shape[1]  - img.shape[1]%32)
    height = int(img.shape[0] - img.shape[0]%32)
    dim = (width, height)
    img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return img

def getPoints(l):
    r, theta = l[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*r
    y0 = b*r
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    return (x1,y1), (x2,y2)

def getIntersection(pair_of_lines):
    line1, line2 = pair_of_lines
    s1 = np.array(line1[0])
    e1 = np.array(line1[1])
    s2 = np.array(line2[0])
    e2 = np.array(line2[1])
    if(s1[0]==e1[0] and s2[0]==e2[0]): #both vertical lines
        return False
    elif(s1[0]!=e1[0] and s2[0]!=e2[0]): #none of them are vertical lines
        a1 = (s1[1] - e1[1]) / (s1[0] - e1[0])
        b1 = s1[1] - (a1 * s1[0])
        a2 = (s2[1] - e2[1]) / (s2[0] - e2[0])
        b2 = s2[1] - (a2 * s2[0])
        if abs(a1 - a2) < sys.float_info.epsilon: #paralel
            return False
        x = (b2 - b1) / (a1 - a2)
        y = a1 * x + b1
    elif(s1[0]==e1[0]): #line1 is vertical line
        x = s1[0]
        a2 = (s2[1] - e2[1]) / (s2[0] - e2[0])
        b2 = s2[1] - (a2 * s2[0])
        y = a2 * x + b2
    elif(s2[0]==e2[0]): #line1 is vertical line
        x = s2[0]
        a1 = (s1[1] - e1[1]) / (s1[0] - e1[0])
        b1 = s1[1] - (a1 * s1[0])
        y = a1 * x + b1
    return (int(x), int(y))

def square_distance(a,b):
    sqrD = (a[0] - b[0])**2 + (a[1] - b[1])**2
    return  sqrD


if __name__ == "__main__":
    img_name = 'img.JPG'
    img = cv.imread(img_name)
    img = scaleImg(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gaus = cv.GaussianBlur(gray, (5,5), 0)
    thresh = cv.adaptiveThreshold(gaus, 255, 1, 1, 11, 2)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    #get contour with largest area
    largest_area = max([cv.contourArea(c) for c in contours])
    outside_contour = [c for c in contours if cv.contourArea(c) == largest_area]

    #create mask with contour
    mask = np.zeros((img.shape),np.uint8)
    cv.drawContours(mask,outside_contour,0,(255,255,255),-1)

    #mask my sudoku grid
    sudoku_masked = np.zeros_like(img)
    sudoku_masked[mask==255] = img[mask==255]



    blank = np.ones_like(img)*255

    #detect 4 edges of the grid in the mask and draw them to the blank
    edges = cv.Canny(mask,5,250,apertureSize = 3)
    lines = cv.HoughLines(edges,1,np.pi/180,200)
    for l in lines:
        p1,p2 = getPoints(l)
        cv.line(blank, p1, p2, (0,0,0),2)
    cv.imshow(str(len(lines))+"lines detected from  mask and added to blank", blank)
    cv.waitKey(0)

    #getting corners of the grid (mask)
    lines_as_coord = [getPoints(l) for l in lines]
    corners = []
    pairs_of_lines =  itertools.combinations(lines_as_coord, 2)
    p_l = list(pairs_of_lines)[0]
    print(p_l[0][0],p_l[1][0])
    for pair_of_lines in itertools.combinations(lines_as_coord, 2):
        intersection =  getIntersection(pair_of_lines)
        if not intersection: continue
        corners.append(intersection)
        cv.circle(img, intersection, 2, (0, 255, 0), -1)
    print(corners)
    cv.imshow('image with four intersection points', img)
    cv.waitKey(0)

    '''
    #skeletonize my masked with lines
    #skl = 1 - blank / 255
    invert = invert(blank)
    cv.imshow("before skeletonizing", invert)
    cv.waitKey(0)
    #skl = morphology.skeletonize(invert)#.astype(np.uint8)
    #cv.imshow("skeletonize on blank", skl)
    #cv.waitKey(0)

    gray = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
    cv.imshow("blank with thershold", thresh)
    cv.waitKey(0)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    new_blank = np.ones_like(img)*255
    for c in contours:
        cv.drawContours(new_blank,c,0,(255,255,255),-1)
    cv.imshow("blank with contours", new_blank)
    cv.waitKey(0)

    '''
    '''

    #getting lines around the grid (mask)
    edges = cv.Canny(skl,5,250,apertureSize = 3)
    lines = cv.HoughLines(edges,1,np.pi/180, 200)
    lines_as_coord = []
    for l in lines:
        p1,p2 = getPoints(l)
        lines_as_coord.append([p1,p2])
        cv.line(mask, p1, p2, (255,0,0),1)
    cv.imshow("lines added", mask)
    cv.waitKey(0)

    print(len(lines))

    '''



    '''
    #dst = 1 - sudoku_masked / 255
    cv.imshow("sudoku_masked", sudoku_masked)
    cv.waitKey(0)
    dst = morphology.skeletonize(sudoku_masked).astype(np.uint8)

    cv.imshow("skeletonize", dst)
    cv.waitKey(0)
    rho = 1
    theta = math.pi / 180
    threshold = 1
    minLineLength = 3
    maxLineGap = 5
    #dst = cv.Canny(dst,5,250,apertureSize = 3)
    lines = np.ndarray([1, 1, 4, 4])
    lines = cv.HoughLinesP(dst, rho, theta, threshold, minLineLength, maxLineGap)
    print('yes',lines)

    for l in lines:
        
        print(l)
        cv.line(sudoku_masked, p1, p2, (0,0,255),2)
    cv.imshow("images with lines", sudoku_masked)
    cv.waitKey(0)


    '''




    '''

    #getting corners of the grid (mask)
    print('number of lines', len(lines))
    corners = []
    pairs_of_lines =  itertools.combinations(lines_as_coord, 2)
    p_l = list(pairs_of_lines)[0]
    print(p_l[0][0],p_l[1][0])
    for pair_of_lines in itertools.combinations(lines_as_coord, 2):
        intersection =  getIntersection(pair_of_lines)
        if not intersection: continue
        corners.append(intersection)
        cv.circle(img, intersection, 2, (0, 255, 0), -1)
    print(corners)

    pointA = corners[0] #any corner of the grid
    closest_to_A = sorted(corners, key = lambda x: square_distance(pointA,x))
    inBetweenPoints = closest_to_A[1:-1]
    pointB = closest_to_A[-1]
    print('closest to A',pointA, closest_to_A)
    lineH1 = [pointA,inBetweenPoints[0]]
    lineH2 = [inBetweenPoints[1], pointB]
    lineV1 = [pointA,inBetweenPoints[1]]
    lineV2 = [inBetweenPoints[0], pointB]

    grid = corners.copy()

    '''





    #net = cv.dnn.readNet('frozen_east_text_detection.pb')
    #blob = cv.dnn.blobFromImage(img, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)
