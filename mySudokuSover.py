import cv2 as cv
import numpy as np

import digit_predictor.Predictor as predictor
import ASP_interface

def pad_and_resize(digit,imsize):
    (w,h) = digit.shape
    w = int(h*2/3)
    digit = cv.resize(digit,(w,h))
    top_bottom_pad = h//7
    new_h = h + 2 * top_bottom_pad
    left_pad = (new_h-w)//2
    right_pad = new_h-(w+left_pad)
    digit = cv.copyMakeBorder(digit, top_bottom_pad , top_bottom_pad , left_pad, right_pad,\
                              borderType = cv.BORDER_CONSTANT, value=[0,0,0] )
    digit = cv.resize(digit,(imsize,imsize))
    return digit

def scaleImg(img):
    dim = (960,960)
    img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return img

def find_cell(x_array, y_array ,x,y):
    i = max([i for i,x_coor in enumerate(x_array) if x_coor < x])
    j = max([j for j,y_coor in enumerate(y_array) if y_coor < y])
    return i,j




if __name__ == "__main__":
    #img_name = 'sudoku_images/su.jpg'    #+input('image name\n')
    #img_name = 'sudoku_images/1113.png'    #+input('image name\n')
    #img_name = 'sudoku_images/1114.JPG'    #+input('image name\n')
    #img_name = 'sudoku_images/IMG-0996.JPG'    #+input('image name\n')
    img_name = 'sudoku_images/IMG-0997.JPG'    #+input('image name\n')
    #img_name = 'sudoku_images/IMG-0998.JPG'    #+input('image name\n')
    #img_name = 'sudoku_images/IMG-0999.JPG'    #+input('image name\n')
    #img_name = 'sudoku_images/IMG-1001.JPG'    #+input('image name\n')
    #img_name = 'sudoku_images/IMG-1002.JPG'    #+input('image name\n')
    #img_name = 'sudoku_images/IMG-1003.JPG'    #+input('image name\n')
    img = cv.imread(img_name)
    img = scaleImg(img)
    cv.imshow('original', img)
    key = cv.waitKey(0)
    if key == 114:  # if either r or R are pressed
        img=cv.transpose(img)
        img=cv.flip(img,flipCode=1)
        cv.imshow('original', img)
        cv.waitKey(0)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gaus = cv.GaussianBlur(gray, (5,5), 0)
    thresh = cv.adaptiveThreshold(gaus, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    #cv.imshow('thresh', thresh)
    #cv.waitKey(0)
    # get contour with largest area (which is sudoku grid outside border),
    # approximate as polygonal shape, and mask the image with the aproximated contour.
    largest_area = max([cv.contourArea(c) for c in contours])
    outside_contour = [c for c in contours if cv.contourArea(c) == largest_area][0]
    mask = np.zeros((img.shape),np.uint8)
    corners = cv.approxPolyDP(outside_contour,0.01*cv.arcLength(outside_contour,True),True)
    
    cv.drawContours(mask,[corners],0,(255,255,255),-1)
    sudoku_masked = np.zeros_like(img)
    sudoku_masked[mask==255] = img[mask==255]
    #cv.imshow('sudoku_masked', sudoku_masked)
    #cv.waitKey(0)

    ############## from grid corners to new grid corners after perspective rectification
    corners=np.squeeze(corners)
    # naming indexes accoring to how the coordinates of the points related to the other points
    top_left_index = np.argmax([-sum(c) for c in corners])
    bottom_right_index = np.argmax([sum(c) for c in corners])
    bottom_left_index = np.argmax([c[1]-c[0] for c in corners])
    top_right_index = np.argmax([c[0]-c[1] for c in corners])


    min_x = min([c[0] for c in corners])
    min_y = min([c[1] for c in corners])
    max_x = max([c[0] for c in corners])
    max_y = max([c[1] for c in corners])

    index_to_new_corner = {top_left_index:[min_x,min_y],\
                           bottom_right_index:[max_x, max_y],\
                           top_right_index:[max_x,min_y],\
                           bottom_left_index:[min_x, max_y]}

    new_corners = np.array([index_to_new_corner[index] for index in range(len(corners))])
    ##############


    # create and apply homography, with current and new corners, to the required image
    h, status = cv.findHomography(corners, new_corners)
    new_perspective = cv.warpPerspective(img, h, img.shape[:2])
    #for i<min_x and i> max_x: new_perspective[i,j] = 255

    #removing a slice of a pixels from each edge of the sudoku grid

    a =  5# constant number added to the mins and maxs to avoide the edge of the sudoku grid.
    min_x = min([c[0] for c in corners])+a
    min_y = min([c[1] for c in corners])+a
    max_x = max([c[0] for c in corners])-a
    max_y = max([c[1] for c in corners])-a
    
    index_to_new_corner = {top_left_index:[min_x,min_y],\
                           bottom_right_index:[max_x, max_y],\
                           top_right_index:[max_x,min_y],\
                           bottom_left_index:[min_x, max_y]}

    new_corners = np.array([index_to_new_corner[index] for index in range(len(corners))])
    
    newmask = np.zeros((img.shape),np.uint8)
    cv.drawContours(newmask,[new_corners],0,(255,255,255),-1)
    new_sudoku_masked = np.zeros_like(img)
    
    new_sudoku_masked[newmask==255] = new_perspective[newmask==255]    
    
    #cv.imshow('sudoku_masked', new_sudoku_masked)
    #cv.waitKey(0)
    # createing a grid of 9x9 cells in the area held between the four new corners
    grid_width = max_x - min_x
    grid_hight = max_y - min_y
    rows = np.arange(min_x, max_x, grid_width//9)
    columns = np.arange(min_y, max_y, grid_hight//9)
    grid_area = grid_width*grid_hight

    #find and predict digits in the new image
    gray = cv.cvtColor(new_sudoku_masked, cv.COLOR_BGR2GRAY)
    gaus = cv.GaussianBlur(gray, (5,5), 0).astype('uint8')
    thresh2 = cv.adaptiveThreshold(gaus, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv.findContours(thresh2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.imshow('new_threshold', thresh2)
    cv.waitKey(0)

    max_area = grid_area//360
    min_area = grid_area//9000
    max_hight = grid_hight//12
    min_hight = grid_hight//30

    #print(grid_area,min_area,max_area)
    #print(grid_hight,min_hight,max_hight)
    predictor.load_model()
    imsize = predictor.getIMSIZE()

    sudoku_numbers = np.empty((9,9)).astype(int)
    asp_lines = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if max_area > area and area > min_area:
            [x,y,w,h] = cv.boundingRect(cnt)
            #print(area,h)
            if max_hight > h and h > min_hight:
                digit = thresh2[y:y+h,x:x+w]
                digit = pad_and_resize(digit,imsize)
                predicted = predictor.predict([digit])
                #print('predicted', predicted[0])
                cv.imshow('digit', digit)
                cv.waitKey(0)
                i,j = find_cell(rows, columns ,x,y)
                sudoku_numbers[i][j] = str(predicted[0])
                asp_lines.append('value(cell({},{}),{}).'.format(i+1,j+1,predicted[0]))

print(np.transpose(sudoku_numbers))
solution = ASP_interface.solve(asp_lines)
for e in solution:
    #i and j go from 0 to 8, but answer goes from 1 to 9
    [i,j,digit] = np.add( [int(c) for c in e if c.isdigit()] , [-1,-1,0] )
    sudoku_numbers[i][j]=digit

print(np.transpose(sudoku_numbers))
