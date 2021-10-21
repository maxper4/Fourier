# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 18:53:54 2021

@author: Perion Maxence, Pinon Alexandre
"""
import sys 
sys.setrecursionlimit(100000000)

import util
import numpy as np
from cmath import exp
from math import pi

import time

def transformee_fourier_1d(array):
    width = len(array)
    tf = np.empty([width], dtype=np.complex128)
    
    for u in range(width):
        s = 0
        for x in range(width):
            s += array[x] * exp(-2j*pi*float(u*x)/float(width))
        tf[u] = s
        
    return tf

def transformee_fourier_1d_inverse(array):
    width = len(array)
    tf = np.empty([width], dtype=np.complex128)
    
    for u in range(width):
        s = 0
        for x in range(width):
            s += array[x] * exp(2j*pi*float(u*x)/float(width))
        tf[u] = s
        
    return tf / width

def transformee_fourier_2d(array):
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

def transformee_fourier_2d_inverse(array):
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

def transformee_rapide_1d(array):
    width = len(array)
    transformee_fourier = np.empty(width, dtype=np.complex128)
    
    for u in range(width):
        transformee_fourier[u] = transformee_rapide_1d_point(array, u, width)
        
    return transformee_fourier

def transformee_rapide_1d_point(array, u, initialWidth):
    width = len(array)
    if(width == 1):
        return array[0]

    pair = []
    impair = []
    for i in range(width):
        if(i % 2 == 0):
            pair.append(array[i])
        else:
            impair.append(array[i])

    return transformee_rapide_1d_point(pair, u, initialWidth / 2) + exp(-2j*pi*u/float(initialWidth)) * transformee_rapide_1d_point(impair, u, initialWidth / 2)

data = util.load_img("test3.png")
grey_values = util.normalize_img(util.rgb_array_to_gray(data))
    
m = 5
data = [i/m for i in range(m)]
start = time.time()
tf = transformee_fourier_1d(data)
#print("tf : " + str(time.time() - start))
#start = time.time()
tfr = transformee_rapide_1d(data)
print(tf)
print(tfr)
print("tf rapide : " + str(time.time() - start))

#tf = transformee_fourier(grey_values)
#tf /= tf.max()
#itf = transformee_fourier_inverse(tf)
#itf /= itf.max()
        
#util.show_img(util.unnormalize_img(grey_values))
#util.show_img(util.unnormalize_img(tf))
#util.show_img(util.unnormalize_img(itf))