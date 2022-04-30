
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:23:02 2022

@author: khale
"""
import numpy
import cv2

classes=list() 

def imgclassifier(image,thresh):
 #   print(len(classes))
    image=numpy.asarray(image)
    
    found=False
    thisclass=0
    height=image.shape[0]
    width=image.shape[1]
    for i in range(len(classes)):

        distance=cv2.norm( image, classes[i], cv2.NORM_L2 )
        similarity = 1 - distance / ( height * width )
        if similarity>thresh:
            
            found=True
            thisclass=numpy.float32(i+1)
  #          classes[i]=(classes[i]+image)/2
            break
        
    if found==False:
        classes.append(image)
        thisclass=numpy.float32(len(classes)+1)
    else:
        classes[int(thisclass-1)]=(classes[int(thisclass-1)]+image)/2
    
    return thisclass
        
def getarray():
    return classes
def setarray(arrr):
    classes=arrr
