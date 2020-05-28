'''
This application creates images with non-zero digits (1 - 9) in black background and white font.The images are (imsize,imsize) size, and they are converted into (1,imsize*imsize) size, and all of them are saved in 'digits.data' and their true values are saved in 'labels.data', one entry per line. They can be read with np.loadtxt('digits.data') which will return an array of images.
Use  images = np.loadtxt('digits.data').reshape([-1,imsize,imsize]) to reshape images before using them.
'''

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os

def pad_and_resize(digit,imsize):
    (h,w) = digit.shape
    w = int(h*2/3)
    digit = cv2.resize(digit,(w,h))
    top_bottom_pad = h//7
    new_h = h + 2 * top_bottom_pad
    left_pad = (new_h-w)//2
    right_pad = new_h-(w+left_pad)
    digit = cv2.copyMakeBorder(digit, top_bottom_pad , top_bottom_pad , left_pad, right_pad,\
                              borderType = cv2.BORDER_CONSTANT, value=[0,0,0] )
    digit = cv2.resize(digit,(imsize,imsize))
    return digit

def get_digit_pil_font(digit,font_path):
    initial_im_size =  120
    img = Image.new('L', (initial_im_size, initial_im_size),0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 90)
    draw.text((0, 0), digit, font=font, fill=255)
    cv_img = np.array(img)
    '''
    if 'adf/AccanthisADFStdNo2-Italic.otf' in font_path:
        cv2.imshow('img', cv_img)
        cv2.waitKey(0)
    '''
    contours, _ = cv2.findContours(cv_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    [x,y,w,h] = cv2.boundingRect(contours[0])
    cv_img = cv_img[y:y+h,x:x+w]
    cv_img = cv2.resize(cv_img,(int(h*2/3),h))
    return cv_img


def get_digit_cv_font(digit,font, imsize, lineType):
    #the digits written at scale 1 cover around 17x22 pixels
    img = np.zeros(shape=[imsize,imsize], dtype=np.uint8)
    font                   = font #cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,48)
    fontScale              = 2
    fontColor              = 255
    lineType               = lineType
    cv_img = cv2.putText(img,str(digit),bottomLeftCornerOfText,\
                font,fontScale,fontColor,lineType)
    return cv_img

def create_images_and_labels(images_file, labels_file, imsize):
    cv_digits_array,cv_labels_array = get_cv_font_images(imsize)
    pil_digits_array,pil_labels_array = get_pil_font_images(imsize)

    img_digits_array =  np.vstack((cv_digits_array , pil_digits_array))

    labels_array = cv_labels_array + pil_labels_array
    if(images_file):
        np.savetxt(images_file, img_digits_array)
    if(labels_file):
        np.savetxt(labels_file, np.array(labels_array), fmt='%i')

def get_cv_font_images(imsize):
    img_digits_array = None
    labels_array = []
    for digit in range(1,10):
        for font in [0,2,3,4,7]:
            for lineType in range(2,8):
                img = get_digit_cv_font(digit,font,imsize, lineType)
                if img_digits_array is None: img_digits_array = img.reshape([1, imsize**2])
                else:img_digits_array = np.vstack((img_digits_array, img.reshape([1, imsize**2])))
                labels_array.append(digit)
    return img_digits_array, labels_array





def get_pil_font_images(imsize):
    img_digits_array = None
    labels_array = []
    fonts_dir = '/usr/share/fonts/truetype/'
    list_subfolders = [f for f in os.scandir(fonts_dir) if f.is_dir()]
    myfontslist = ['malayalam','lato','adf','crosextra']
    excludefonts = ['Chilanka-Regular.ttf', 'Karumbi.ttf']#they look like handwritting
    fonts_paths = [os.path.join(subf.name,f)  for subf in list_subfolders for f in os.listdir(subf.path) if '.' in f and f not in excludefonts and subf.name in myfontslist ]

    for digit in range(1,10):
        for path in fonts_paths:
            img = get_digit_pil_font(str(digit),path)
            #if 'adf/AccanthisADFStdNo2-Italic.otf' in path:
            #    print(path)
            #    cv2.imshow('img', img)
            #    cv2.waitKey(0)
            img = pad_and_resize(img,imsize)
            '''
            if 'tlwg' in path:
                print(path)
                cv2.imshow('img', img)
                cv2.waitKey(0)
            '''
            if img_digits_array is None: img_digits_array = img.reshape([1, imsize**2])
            else:img_digits_array = np.vstack((img_digits_array, img.reshape([1, imsize**2])))
            labels_array.append(digit)
    return img_digits_array, labels_array

if __name__ == "__main__":
    imsize = 56
    images_file = 'data/DIGITS/digits.data'
    labels_file = 'data/DIGITS/labels.data'
    create_images_and_labels(images_file, labels_file, imsize )

'''
This is the list of fonts used by cv2. We chose a few of them to create our synthetic dataset.

FONT_HERSHEY_SIMPLEX        = 0, //!< normal size sans-serif font
FONT_HERSHEY_PLAIN          = 1, //!< small size sans-serif font
FONT_HERSHEY_DUPLEX         = 2, //!< normal size sans-serif font (more complex
    than FONT_HERSHEY_SIMPLEX)
FONT_HERSHEY_COMPLEX        = 3, //!< normal size serif font
FONT_HERSHEY_TRIPLEX        = 4, //!< normal size serif font (more complex than
    FONT_HERSHEY_COMPLEX)
FONT_HERSHEY_COMPLEX_SMALL  = 5, //!< smaller version of FONT_HERSHEY_COMPLEX
FONT_HERSHEY_SCRIPT_SIMPLEX = 6, //!< hand-writing style font
FONT_HERSHEY_SCRIPT_COMPLEX = 7, //!< more complex variant of
    FONT_HERSHEY_SCRIPT_SIMPLEX
FONT_ITALIC                 = 16 //!< flag for italic font
'''
