import os

import cv2
import matplotlib.pyplot as plt
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
labels = [pos2imgLine(pos,finalHeight,finalWidth,lineWidth=4,lineType=cv2.LINE_AA) for pos in poses]
model = tf.keras.models.load_model("1Foot100epoch4width.hdf5",compile=False)
prefix = "Two2OneRes/100epochs4width/"
os.makedirs(prefix,exist_ok=True)
os.makedirs(prefix+"WELSCH/",exist_ok=True)
# os.makedirs(prefix+"WELSCH1/",exist_ok=True)
# os.makedirs(prefix+"WELSCH2/",exist_ok=True)
# os.makedirs(prefix+"WELSCH4/",exist_ok=True)
# os.makedirs(prefix+"WELSCH5/",exist_ok=True)
# os.makedirs(prefix+"L2/",exist_ok=True)
# os.makedirs(prefix+"L1/",exist_ok=True)
# os.makedirs(prefix+"L12/",exist_ok=True)
# os.makedirs(prefix+"FAIR/",exist_ok=True)
#保存预测的结果
for x in range(len(imgs)):
    predict = model.predict_step(np.expand_dims(imgs[x]/255,0))[0]
    idx = img_filenames[x].split('.')[0]
    # predict heatmap
    # plt.figure(figsize=(1,6))
    # plt.subplots_adjust(left=0.02,top=0.99,right=0.98,bottom=0.01,wspace=0.02,hspace=0.02)
    # for i in range(predict.shape[2]):
    #     plt.subplot(3,1,i+1)
    #     # heatmap = np.stack([predict[:, :, i],labels[x][:,:,i],np.zeros([256,128])],-1)
    #     heatmap = predict[:,:,i]
    #     plt.imshow(heatmap,)
    #     plt.axis('off')
    # plt.savefig(prefix + idx + "predict.png")
    # plt.show()
    #target heatmap
    # img = np.copy(imgs[x])
    # plt.figure(figsize=(1,6))
    # plt.subplots_adjust(left=0.02,top=0.99,right=0.98,bottom=0.01,wspace=0.02,hspace=0.02)
    # for i in range(predict.shape[2]):
    #     plt.subplot(3,1,i+1)
    #     heatmap = labels[x][:,:,i]
    #     plt.imshow(heatmap)
    #     plt.axis('off')
    # plt.savefig(prefix + idx + "label.png")
    # plt.close('all')
    #ground truth
    img = np.copy(imgs[x])
    draw_keyPoints(img,poses[x])
    draw_lines(img,poses[x])
    plt.imsave(f'{prefix}LineOrigin{idx}.png',img)
    #预测点
    img = np.copy(imgs[x])
    img = LinesOnImage(img,predict)
    plt.imsave(f'{prefix}LineRaw{idx}.png',img)
    #回归线
    img = np.copy(imgs[x])
    img = line_regression(img,predict,cv2.DIST_WELSCH)
    cv2.imwrite(prefix+"WELSCH/" + idx + ".png",img)
    print(x)
    ##different metrics
    # img = np.copy(imgs[x])
    # img = line_regression(img,predict,cv2.DIST_L2)
    # cv2.imwrite(prefix+"L2/" + idx + ".png",img)
    # img = np.copy(imgs[x])
    # img = line_regression(img, predict, cv2.DIST_L1)
    # cv2.imwrite(prefix + "L1/" + idx + ".png", img)
    # img = np.copy(imgs[x])
    # img = line_regression(img, predict, cv2.DIST_L12)
    # cv2.imwrite(prefix + "L12/" + idx + ".png", img)
    # img = np.copy(imgs[x])
    # img = line_regression(img, predict, cv2.DIST_FAIR)
    # cv2.imwrite(prefix + "FAIR/" + idx + ".png", img)
    ##WELSCH C
    # img = np.copy(imgs[x])
    # img = line_regression(img,predict,cv2.DIST_WELSCH,constant=1)
    # cv2.imwrite(prefix+"WELSCH1/" + idx + ".png",img)
    # img = np.copy(imgs[x])
    # img = line_regression(img,predict,cv2.DIST_WELSCH,constant=2)
    # cv2.imwrite(prefix+"WELSCH2/" + idx + ".png",img)
    # img = np.copy(imgs[x])
    # img = line_regression(img,predict,cv2.DIST_WELSCH,constant=4)
    # cv2.imwrite(prefix+"WELSCH4/" + idx + ".png",img)
    # img = np.copy(imgs[x])
    # img = line_regression(img,predict,cv2.DIST_WELSCH,constant=5)
    # cv2.imwrite(prefix+"WELSCH5/" + idx + ".png",img)
    # img = np.copy(imgs[x])
    # img = line_regression(img,predict,cv2.DIST_WELSCH,constant=0)
    # cv2.imwrite(prefix+"WELSCH/" + idx + ".png",img)
    # print(x)
#衡量预测的角度偏差
# AngleErrors = []
# DetectedNum = 0
# for x in range(len(imgs)):
#     predict = model.predict_step(np.expand_dims(imgs[x] / 255, 0))[0]
#     print(x)
#     res = calAngleError(imgs[x],predict,poses[x],cv2.DIST_WELSCH)
#     if res[0]:
#         if np.max(res[1])<5:
#             AngleErrors.append(res[1])
#             DetectedNum +=1
#     print(res)
# AngleErrors = np.array(AngleErrors)
# print(f'maxium errors:{np.max(AngleErrors)}')
# np.savetxt(prefix+'AngleErrors.txt',AngleErrors)
# print(np.mean(np.abs(AngleErrors), 0),'DetectedNum:',DetectedNum)