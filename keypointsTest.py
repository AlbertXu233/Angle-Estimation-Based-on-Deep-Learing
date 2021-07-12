import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from preprocess import *
from model import *
from postprocess import *
# from matplotlib import pyplot as plt
finalHeight = 256
finalWidth = 128
FilePath = "Two2One/"
img_filenames = get_filenames(FilePath)
print(img_filenames)
imgs = [cv2.imread(FilePath+name) for name in img_filenames]
old_widths = [img.shape[1] for img in imgs]
imgs = [resizeCropByWidth(img,512) for img in imgs]
# imgs = [equalhist3channels(img) for img in imgs]
json_filenames = get_filenames(FilePath,"json")
print(json_filenames)
poses = [get_json_data(FilePath+name) for name in json_filenames]
poses = [(512*pose//width).astype(np.int) for pose,width in zip(poses,old_widths)]
N = 2
labels = [pos2imgMulti(pos,finalHeight,finalWidth,numbers=N) for pos in poses]
model = tf.keras.models.load_model(f"1Foot{N}Keypoints100.hdf5",compile=False)
prefix = f"Two2OneRes/Keypoints100epoch/{N}/"
os.makedirs(prefix,exist_ok=True)
#保存预测的结果
for x in range(len(imgs)):
    predict = model.predict_step(np.expand_dims(imgs[x]/255,0))[0]
    idx = img_filenames[x].split('.')[0]
    #ground truth
    img = np.copy(imgs[x])
    draw_keyPoints(img,poses[x])
    draw_lines(img,poses[x])
    plt.imsave(f'{prefix}KeyOrigin{idx}.png',img)
    #predict keypoints heatmap
    img = np.copy(imgs[x])
    plt.figure(figsize=(N+0.1,6))
    plt.subplots_adjust(left=0.02,top=0.99,right=0.98,bottom=0.01,wspace=0.02,hspace=0.02)
    for i in range(predict.shape[2]):
        plt.subplot(3,predict.shape[2]//3,i+1)
        # heatmap = np.stack([predict[:, :, i],labels[x][:,:,i],np.zeros([256,128])],-1)
        heatmap = predict[:,:,i]
        plt.imshow(heatmap,)
        plt.axis('off')
    plt.savefig(f'{prefix}KeyPredict{idx}.png')
    # plt.show()
    #target keypoints heatmap
    img = np.copy(imgs[x])
    plt.figure(figsize=(N+0.1,6))
    plt.subplots_adjust(left=0.02,top=0.99,right=0.98,bottom=0.01,wspace=0.02,hspace=0.02)
    for i in range(predict.shape[2]):
        plt.subplot(3,predict.shape[2]//3,i+1)
        heatmap = labels[x][:,:,i]
        plt.imshow(heatmap)
        plt.axis('off')
    plt.savefig(f'{prefix}KeyLabel{idx}.png')
    #final result
    img = heatmap2linesMul(imgs[x],predict)
    plt.imsave(f'{prefix}KeyOut{idx}.png',img)
    plt.close('all')
    print(x)