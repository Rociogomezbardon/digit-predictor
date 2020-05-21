'''
This application creates images with non-zero digits (1 - 9) in black background and white font.The images are (imsize,imsize) size, and they are converted into (1,imsize*imsize) size, and all of them are saved in 'digits.data' and their true values are saved in 'labels.data', one entry per line. They can be read with np.loadtxt('digits.data') which will return an array of images. 
Use  images = np.loadtxt('digits.data').reshape([-1,imsize,imsize]) to reshape images before using them.
'''

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def get_digit_img(digit,font, imsize, lineType):
    #the digits written at scale 1 cover around 17x22 pixels    
    img = np.zeros(shape=[imsize,imsize], dtype=np.uint8)
    font                   = font #cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,48)
    fontScale              = 2
    fontColor              = 255
    lineType               = lineType

    with_digit = cv2.putText(img,str(digit),bottomLeftCornerOfText,\
                font,fontScale,fontColor,lineType)
    return with_digit

def create_images_and_labels(images_file, labels_file, imsize):
    all_digits = None
    all_labels = []
    for digit in range(1,10):
        for font in [0,2,3,4,7]:
            for lineType in range(2,8):
            
                img = get_digit_img(digit,font,imsize, lineType)
                if all_digits is None: all_digits = img.reshape([1, imsize**2])
                else:all_digits = np.vstack((all_digits, img.reshape([1, imsize**2])))
                all_labels.append(digit)
    np.savetxt(images_file, all_digits)
    np.savetxt(labels_file, np.array(all_labels), fmt='%i')


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
