# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 18:53:54 2021

@author: Perion Maxence, Pinon Alexandre
"""
import util
import numpy as np
from cmath import exp
from math import pi

def transformee_fourier(array):
    (width, height) = util.get_array_dimensions(array)
    transformee_fourier = np.empty([width, height], dtype=np.complex128)

    for u in range(width):
        for v in range(height):
            s = 0
            for x in range(width):
                for y in range(height):
                    s += array[x, y] * exp(-2j*pi*(float(u*x)/float(width) + float(v*y)/float(height)))
            transformee_fourier[u, v] = s
    return transformee_fourier

def transformee_fourier_inverse(array):
    (width, height) = util.get_array_dimensions(array)
    transformee_fourier = np.empty([width, height], dtype=np.complex128)

    for u in range(width):
        for v in range(height):
            s = 0
            for x in range(width):
                for y in range(height):
                    s += array[x, y] * exp(2j*pi*(float(u*x)/float(width) + float(v*y)/float(height)))
                    
            transformee_fourier[u, v] = s
    return transformee_fourier

data = util.load_img("test.png")

grey_values = util.normalize_img(util.rgb_array_to_gray(data))
    
tf = transformee_fourier(grey_values)
tf /= tf.max()
itf = transformee_fourier_inverse(tf)
itf /= itf.max()
        
util.show_img(util.unnormalize_img(grey_values))
util.show_img(util.unnormalize_img(tf))
util.show_img(util.unnormalize_img(itf))