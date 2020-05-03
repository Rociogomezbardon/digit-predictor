
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
    dim = (960,960)
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

def findNumber(img):
    return None

if __name__ == "__main__":
    img_name = 'test.jpg'
    img = cv.imread(img_name)
    img = scaleImg(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gaus = cv.GaussianBlur(gray, (5,5), 0)
    thresh = cv.adaptiveThreshold(gaus, 255, 1, 1, 11, 2)
    cv.imshow('thresh', thresh)
    cv.waitKey(0)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    #get contour with largest area
    largest_area = max([cv.contourArea(c) for c in contours])
    outside_contour = [c for c in contours if cv.contourArea(c) == largest_area]


    #approximate it to a polygonal shape and draw it on mask.
    # found it here: https://pysource.com/2018/09/25/simple-shape-detection-opencv-with-python-3/
    mask = np.zeros((img.shape),np.uint8)
    corners = cv.approxPolyDP(outside_contour[0],0.01*cv.arcLength(outside_contour[0],True),True)
    cv.drawContours(mask,[corners],0,(255,255,255),-1)
    cv.imshow('the mask with new contour', mask)
    cv.waitKey(0)
    corners=np.squeeze(corners)
    min_x = min([c[0] for c in corners])
    min_y = min([c[1] for c in corners])
    max_x = max([c[0] for c in corners])
    max_y = max([c[1] for c in corners])
    new_corners = np.array([[max_x,min_y],[min_x,min_y],[min_x, max_y],[max_x, max_y]])


    # https://www.programcreek.com/python/example/89422/cv2.warpPerspective
    # https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
    h, status = cv.findHomography(corners, new_corners)
    new_img = cv.warpPerspective(img, h, img.shape[:2])
    cv.imshow('the img with new perspective', new_img)
    cv.waitKey(0)

    # createing a grid of 9x9 cells in the area held between the four new corners
    grid_width = max_x - min_x
    grid_hight = max_y - min_y
    x_array = np.arange(min_x, max_x, grid_width//9)
    y_array = np.arange(min_y, max_y, grid_hight//9)

    # print the points of the grid
    for x in x_array:
        for y in y_array:
            cv.circle(new_img, (x,y), 0, (255,0,0), 5)
    cv.imshow('the img with gridpoints', new_img)
    cv.waitKey(0)

    sudoku_numbers = np.empty((9,9))
    # for each cell print image.
    for j,y in enumerate(y_array[:-1]):
        for i,x in enumerate(x_array[:-1]):
            cell = new_img[y:y_array[j+1], x:x_array[i+1]]
            sudoku_numbers[i][j] = findNumber(cell)

    print(sudoku_numbers)
    #mask my sudoku grid
    #sudoku_masked = np.zeros_like(img)
    #sudoku_masked[mask==255] = img[mask==255]
    #cv.imshow('the mask with new contour', sudoku_masked)
    #cv.waitKey(0)





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
