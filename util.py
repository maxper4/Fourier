# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:03:07 2021

Utility class for manipulating images

@author: Perion Maxence, Pinon Alexandre
"""

from PIL import Image
import numpy as np

# load an image from path on the disk and return the corresponding array
def load_img(path):
    image = Image.open(path)  # open image
    return np.asarray(image)   # convert to array

# show the image represented by the array array_data
def show_img(array_data):
    img = Image.fromarray(array_data)   # convert the array to a PIL Image
    img.show()
    
# return a tuple containing the dimensions of array
def get_array_dimensions(array):
    (height, width, *other) = array.shape # *other because shape can return more values (if array has more dimensions)
    return (height, width)

# convert an rgb array to a gray level one
def rgb_array_to_gray(array):
    (height, width) = get_array_dimensions(array)
    
    grey_values = np.empty([height, width], dtype=np.complex128) # new array
    
    # put in the new array for each pixel the average of r, g and b values
    for i in range(height):
        for j in range(width):
            grey_values[i, j] = (int(array[i, j][0]) + int(array[i, j][1]) + int(array[i, j][2])) / 3 # cast to int to avoid overflow (values from 0 to 255 can enter in an int8)
    
    return grey_values

# convert an array with pixels from 0 to 255 in pixels with float value from 0 to 1
def normalize_img(array):
    (height, width) = get_array_dimensions(array)
    
    values = np.empty([height, width], dtype=np.complex128) # new array
    
    # divide by max value 255, in float
    for i in range(height):
        for j in range(width):
            values[i, j] = array[i, j] / 255.0
            
    return values

# convert an array with pixels with values between 0 and 1 to values between 0 and 255
def unnormalize_img(array):
    (height, width) = get_array_dimensions(array)
    
    values = np.empty([height, width])
    
    # multiply by max value 255, in float
    for i in range(height):
        for j in range(width):
            values[i, j] = array[i, j] * 255.0
            
    return values
