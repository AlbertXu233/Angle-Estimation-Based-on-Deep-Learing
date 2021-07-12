import os
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
def get_filenames(dir,typename="png"):
    filenames = os.listdir(dir)
    results =  []
    for name in filenames:
        if name.split(".")[1]==typename:
            results.append(name)
    results.sort()
    return results
def get_json_data(filename):
    with open(filename,'r',encoding='utf8') as load_f:
        load_data = json.load(load_f)
    data = [x['points'][0] for x in load_data['shapes']][:6]
    data = np.array(data)
    data = np.reshape(data,12)
    return data
softmax = tf.keras.layers.Softmax([0,1])
def pos2img(pnts,finalHeight=20,finalWidth=12,old_width_border=512):
    pnts = pnts*finalWidth//old_width_border
    image = np.zeros([finalHeight,finalWidth,6])
    for i in range(6):
        image[:,:,i] = CenterLabelHeatMap(finalWidth,finalHeight,pnts[2*i],pnts[2*i+1],4)
    return image[:finalHeight]
def pos2imgMulti(pnts,finalHeight=20,finalWidth=12,old_width_border=512,numbers=2):
    pnts = pnts * finalWidth // old_width_border
    image = np.zeros([finalHeight,finalWidth,3*numbers])
    for i in range(3):
        s = 4*i
        for j in range(numbers):
            x1,y1,x2,y2 = pnts[s:s+4]
            x = x1/numbers*(numbers-j) + x2/numbers*j
            y = y1/numbers*(numbers-j) + y2/numbers*j
            image[:, :, i*numbers+j] = CenterLabelHeatMap(finalWidth, finalHeight, x, y, 4)
    return image
def pos2imgLine(pnts,finalHeight=20,finalWidth=12,old_width_border=512,lineWidth=2,lineType=None):
    pnts = pnts*finalWidth//old_width_border
    image = np.zeros([3,finalHeight,finalWidth],np.uint8)
    for i in range(3):
        cv2.line(image[i],(pnts[4*i],pnts[4*i+1]),(pnts[4*i+2],pnts[4*i+3]),1,lineWidth,lineType)
    image = np.swapaxes(image,0,-1)
    image = np.swapaxes(image,0,1)
    return image[:finalHeight].astype(np.float32)
def resizeCropByWidth(img,width):
    h,w = img.shape[:2]
    img = cv2.resize(img,(width,h*width//w))
    if img.shape[0]<2*width:
        img = cv2.copyMakeBorder(img,0,2*width-img.shape[0],0,0,cv2.BORDER_CONSTANT,0)
    else:
        img = img[:2*width]
    return img
def equalhist3channels(img):
    img[:,:,0]=cv2.equalizeHist(img[:,:,0])
    img[:,:,1] = cv2.equalizeHist(img[:,:,1])
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])
    return img
def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap
# pos2imgLine(np.array([1,10,2,10,3,10,4,10,5,10,6,10]),256,128,512)
# img = pos2imgMulti(np.array([10,10,30,30,30,50,50,80,70,70,90,90]),256,128,512,3)
# channels = img.shape[2]
# for c in range(channels):
#     plt.imshow(img[:,:,c])
#     plt.show()