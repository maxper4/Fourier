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
    (height, width) = util.get_array_dimensions(array)
    transformee_fourier = np.empty([height, width], dtype=np.complex128)

    for v in range(height):
        for u in range(width):
            s = 0
            for y in range(height):
                for x in range(width):
                    s += array[y, x] * exp(-2j*pi*(float(u*x)/float(width) + float(v*y)/float(height)))
            transformee_fourier[v, u] = s
    return transformee_fourier

def transformee_fourier_2d_inverse(array):
    (height, width) = util.get_array_dimensions(array)
    transformee_fourier = np.empty([height, width], dtype=np.complex128)

    for v in range(height):
        for u in range(width):
            s = 0
            for y in range(height):
                for x in range(width):
                    s += array[y, x] * exp(2j*pi*(float(u*x)/float(width) + float(v*y)/float(height)))
                    
            transformee_fourier[v, u] = s
    return transformee_fourier / width / height

def transformee_rapide_1d(array):
    width = len(array)
    if(width == 1):
        return np.asarray([array[0]], dtype=np.complex128)
    
    pair = transformee_rapide_1d(array[::2])
    impair = transformee_rapide_1d(array[1::2])
    tf = np.empty([width], dtype=np.complex128)
    halfWidth = int(width/2)
    
    for u in range(halfWidth):
        tf[u] = pair[u] + exp(-2j * pi * u / width) * impair[u]
        tf[u + halfWidth] = pair[u] + exp(-2j * pi * (u + halfWidth) / width) * impair[u]
        
    return tf    

def transformee_rapide_1d_inverse_recur(array):
    width = len(array)
    if(width == 1):
        return np.asarray([array[0]], dtype=np.complex128)
    
    pair = transformee_rapide_1d_inverse_recur(array[::2])
    impair = transformee_rapide_1d_inverse_recur(array[1::2])
    tf = np.empty([width], dtype=np.complex128)
    halfWidth = int(width/2)
    
    for u in range(halfWidth):
        tf[u] = pair[u] + exp(2j * pi * u / width) * impair[u]
        tf[u + halfWidth] = pair[u] + exp(2j * pi * (u + halfWidth) / width) * impair[u]
        
    return tf

def transformee_rapide_1d_inverse(array):
    width = len(array)
    return transformee_rapide_1d_inverse_recur(array) / width

def transformee_rapide_2d(array):
    (height, width) = util.get_array_dimensions(array)
    tf = np.empty([height, width], dtype=np.complex128)

    tfLignes = []
    for i in range(height):
        tfLignes.append(transformee_rapide_1d(array[i]))
        
    tfColonnes = []
    for i in range(width):
        tfColonnes.append(transformee_rapide_1d([tfLignes[x][i] for x in range(height)]))
    
    for u in range(height):
        for v in range(width):
            tf[u, v] = tfColonnes[v][u]
    
    return tf

def transformee_rapide_2d_inverse(array):
    (height, width) = util.get_array_dimensions(array)
    tf = np.empty([height, width], dtype=np.complex128)

    tfLignes = []
    for i in range(height):
        tfLignes.append(transformee_rapide_1d_inverse(array[i]))
        
    tfColonnes = []
    for i in range(width):
        tfColonnes.append(transformee_rapide_1d_inverse([tfLignes[x][i] for x in range(height)]))
    
    for u in range(height):
        for v in range(width):
            tf[u, v] = tfColonnes[v][u]
    
    return tf
    
data = util.load_img("test3.png")
grey_values = util.normalize_img(util.rgb_array_to_gray(data))
#data = np.asarray([[i/m for i in range(m)] for y in range(n)], dtype=np.complex128)
#print(data)
#start = time.time()
#tf = transformee_fourier_1d(data)
#print("tf : " + str(time.time() - start))
#start = time.time()
#tfr = transformee_rapide_1d(data)
#print(transformee_fourier_1d_inverse(data))

tf = transformee_rapide_2d(grey_values)
itf = transformee_rapide_2d_inverse(tf)
print(np.allclose(tf, np.fft.fft2(grey_values)))
#print(tf)
#print(np.fft.fft2(grey_values))
#print(np.allclose(tf, np.fft.fft2(grey_values)))
#print(np.allclose(tf, transformee_fourier_2d(grey_values)))
util.show_img(util.unnormalize_img(grey_values))
util.show_img(util.unnormalize_img(tf))
util.show_img(util.unnormalize_img(itf))

util.show_img(util.unnormalize_img(np.fft.fft2(grey_values)))