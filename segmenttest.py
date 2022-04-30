# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 14:50:00 2021

@author: khale
"""
import math
import cv2
import numpy as np
import os
from scipy.ndimage import label
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import torch

from classify import imgclassifier
from classify import getarray
from classify import setarray
# Load image, for testing
image = cv2.imread('sq.png')

def getclassarray():
    return getarray()

def setclassarray(arrr):
    setarray(arrr)
    
def prepareData(images,objectcentroids,imgsave,framesize,standardsize,GridSize,classthresh):
    
    newimages=[]
    Xarray=[]
    Yarray=[]
    classes=[]

    
    for i in range(0,len(images)):

        pixels = np.asarray(images[i])
        
        #global centering of pixels
        mean = pixels.mean()
        pixels = pixels - mean
        
        #resize image to standard value
        pixels=cv2.resize(pixels,(standardsize,standardsize))

        Xarray.append(float(math.ceil((objectcentroids[i][0])*GridSize/framesize[0])))
        Yarray.append(float(math.ceil((objectcentroids[i][1])*GridSize/framesize[1])))

        classs=np.float32(imgclassifier(pixels,classthresh))        
        classes.append(classs)
        
        ### save image to file
        if (imgsave):
            name=str(classs)+str(objectcentroids[i][1:5])+".png"
            path=os.path.realpath(__file__)
            path=os.path.dirname(path)
            path=os.path.join(path,"objctimg",name)
           
            
            cv2.imwrite(path, pixels)
                

    return classes,Xarray,Yarray


def segmentframe(image,debug,imgsave,objctsize,gridsize,classthresh):
    
    
    
    ######################  downscaling #########################

    resize=cv2.resize(image,(int(image.shape[1]/2),int(image.shape[0]/1.5)) ,interpolation = cv2.INTER_LINEAR)
    #resize=cv2.resize(image,(64 ,96) ,interpolation = cv2.INTER_LINEAR)
    
    
    if (debug):
        
        cv2.namedWindow('rs',cv2.WINDOW_NORMAL)
        
        
        cv2.resizeWindow('rs', 600,600)
        
        cv2.imshow("rs", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.namedWindow('rs',cv2.WINDOW_NORMAL)
        
        
        cv2.resizeWindow('rs', 600,600)
        
        cv2.imshow("rs", resize)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    
    
    ##########################crop##############################
    
    crop = resize[int(resize.shape[0]*0.23):int(resize.shape[0]*0.93), int(resize.shape[1]/18):int(resize.shape[1])]
    cropc=resize
 
    
    if (debug):
        cv2.namedWindow('crop',cv2.WINDOW_NORMAL)
        
        
        cv2.resizeWindow('crop', 600,600)
        
        
        cv2.imshow("crop", crop)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    ############################# 2greyscale ###############################
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    if (debug):
        cv2.namedWindow('grey',cv2.WINDOW_NORMAL)
        
        
        cv2.resizeWindow('grey', 600,600)
        
        
        cv2.imshow("grey", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    ####################### fore/back ground ###################################
    
    kernel = np.ones((3,3),np.uint8)
    
    dilate = cv2.dilate(gray,kernel,iterations=1)
    
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    erode1= cv2.erode(dilate,kernel,iterations=3)
    dilate1= cv2.dilate(erode1,kernel,iterations=3)
    


    sub=cv2.subtract(blur,dilate1)
    
    if (debug):
        
    
        plot=np.concatenate((dilate, blur), axis=1)
        plotv=np.concatenate((erode1, dilate1), axis=1)
        plot2=np.concatenate((sub, sub), axis=1)
        plot=np.concatenate((plot, plotv), axis=0)
        plot=np.concatenate((plot, plot2), axis=0)
        cv2.namedWindow('blur',cv2.WINDOW_NORMAL)
        
        
      #  cv2.resizeWindow('blur', 600,600)
        
        cv2.imshow("blur", plot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    
    ############################ threshold/binarize ###############################
    
   # gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    
    if (debug):
        cv2.namedWindow('thresh',cv2.WINDOW_NORMAL)
        
        
        cv2.resizeWindow('thresh', 600,600)
        
        
        cv2.imshow("thresh", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    
    #####################broken watershed testing#######################################
    
  #   kernel = np.ones((3,3),np.uint8)

  #   opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
  #   dist_transform = cv2.distanceTransform(thresh , cv2.DIST_L2,3)

  #   bg=erode1
  #   rat, bg = cv2.threshold(bg,0.7*dist_transform.max(),255,0)

  #   peak = cv2.threshold(dist_transform, 0.7*dist_transform.max(),255, cv2.THRESH_BINARY)[1]
  #   peak = np.uint8(peak)
    
  #   unknown=cv2.subtract(thresh,peak)
  #   #unknown = cv2.bitwise_and(dist_transform,dist_transform,mask =255-peak)
        
  #   ret, markers = cv2.connectedComponents(peak)
  #   markers = markers+1
  #   markers[unknown>0] = 0
  #   markers = cv2.watershed(cropc,markers)
  # #  cropc[markers == -1] = [0,255,0]
    
  #   colors = np.random.randint(0, 255, size=(ret+1, 3), dtype=np.uint8)
  #   colors[0] = [0, 0, 0]  # for cosmetic reason we want the background black
  #  # false_colors = colors[markers]
  #  # false_colors_area = false_colors.copy()
  #   # Add one to all labels so that sure background is not 0, but 1
   
    
   
  #   mark = np.uint8(markers)
  #   if (debug):
  #       cv2.namedWindow('wstest',cv2.WINDOW_NORMAL)
       
       
  #       cv2.resizeWindow('wstest', 600,600)
       
       
  #       cv2.imshow("wstest", peak )
    
       
  #       cv2.waitKey(0)
  #       cv2.destroyAllWindows()
       
   
    ###################Find components########################


    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    

    # Create false color image and color background black
    colors = np.random.randint(0, 255, size=(n_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # for cosmetic reason we want the background black
    false_colors = colors[labels]


    if (debug):
        cv2.namedWindow('fc',cv2.WINDOW_NORMAL)
        
        
        cv2.resizeWindow('fc', 600,600)
        
        
        cv2.imshow("fc", false_colors)
    
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ##################binarize results##########################
    
    padding=1
    
    
    img2_fg = cv2.bitwise_and(crop,crop,mask = thresh)
    
    cropborder= cv2.copyMakeBorder(img2_fg,padding,padding,padding,padding,cv2.BORDER_CONSTANT,value=[0,0,0])
    cropborderclrs=cv2.copyMakeBorder(false_colors,padding,padding,padding,padding,cv2.BORDER_CONSTANT,value=[0,0,0])
    if (debug):

        cv2.namedWindow('binary',cv2.WINDOW_NORMAL)
        
        
        cv2.resizeWindow('binary', 600,600)
        
    
        cv2.imshow("binary", img2_fg)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    
    ###################bounding boxes######################
    
    objectimgs=[]
    objectcentroids=[]
    
    for i, centroid in enumerate(centroids[1:], start=1):

        area = stats[i, 4]
        cv2.putText(false_colors, str(area), (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,255,255), 1)
        
    
        Left=stats[i,0]+padding
        Right=Left+stats[i,2]+padding
        Left=Left-padding
    
        Top=stats[i,1]+padding
        Bottom=Top+stats[i,3]+padding
        Top=Top-padding
        
        
        objct=cropborder[Top:Bottom,Left:Right]
        clrobjct=cropborderclrs[Top:Bottom,Left:Right]
        clrobjct[clrobjct!=colors[i]]=0
        img_grey = cv2.cvtColor(clrobjct, cv2.COLOR_BGR2GRAY)
        hresh = cv2.threshold(img_grey, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
      
        objct=cv2.bitwise_and(objct,objct,mask = hresh)
        objectimgs.append(np.array(objct))
        objectcentroids.append(centroid)
        

        if (debug):
            

            cv2.namedWindow('object',cv2.WINDOW_NORMAL)
            
            
            cv2.resizeWindow('object', 600,600)
            
            
            cv2.imshow("object", hresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.namedWindow('object',cv2.WINDOW_NORMAL)
            
            
            cv2.resizeWindow('object', 600,600)
            
            
            cv2.imshow("object", objct)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                
 
    # objectcentroids=torch.from_numpy(np.array(objectcentroids))
    final=np.zeros((6,3),dtype=np.float32)
    if len(objectimgs)>0:
        classes, Xarray,Yarray=prepareData(objectimgs,objectcentroids,imgsave,[false_colors.shape[0],false_colors.shape[1]],objctsize,gridsize,classthresh)
        
        classes=np.vstack(np.array(classes))

        Xarray=np.vstack(Xarray)
        Yarray=np.vstack(Yarray)
 
        

        results=np.concatenate((classes,Xarray),axis=1)
        results=np.concatenate((results,Yarray),axis=1)
        results=results[0:min(results.shape[0],6),:]

        np.random.shuffle(results)
        final[0:min(results.shape[0],6),:]=results
        
        idx=(np.random.choice(results.shape[0], 6-results.shape[0]))
        final[len(results):6,:]=results[idx]
  
        tensortest=torch.from_numpy(results).float().unsqueeze(0)
   
        # images=torch.from_numpy(images)
        # objectcentroids=torch.from_numpy(objectcentroids)
    

  
     
    return final



def applyPCA(image):
    blue,green,red = cv2.split(image) 
    
    pca = PCA(10)
     
    #Applying to red channel and then applying inverse transform to transformed array.
    red_transformed = pca.fit_transform(red)
    red_inverted = pca.inverse_transform(red_transformed)
     
    #Applying to Green channel and then applying inverse transform to transformed array.
    green_transformed = pca.fit_transform(green)
    green_inverted = pca.inverse_transform(green_transformed)
     
    #Applying to Blue channel and then applying inverse transform to transformed array.
    blue_transformed = pca.fit_transform(blue)
    blue_inverted = pca.inverse_transform(blue_transformed)
    
    img_compressed = (np.dstack((red_inverted, red_inverted, red_inverted))).astype(np.uint8)
    plt.figure()
    plt.imshow(img_compressed)


###FOR TESTING
#while(True):
#    print(segmentframe(image,False,False,16,14,-2.1))
