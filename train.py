import numpy as np
from keras.preprocessing.image import random_brightness
from preprocess import *
from model import *
from postprocess import *
from tensorflow.python.eager import backprop
import keras as ks
import tensorflow as tf
from matplotlib import pyplot as plt
# ks.preprocessing.image.ImageDataGenerator()
# ks.preprocessing.image.
finalHeight = 256
finalWidth = 128
FilePath = "dataOneFoot/"
img_filenames = get_filenames(FilePath)
print(img_filenames)
imgs = [cv2.imread(FilePath+name) for name in img_filenames]
old_widths = [img.shape[1] for img in imgs]
imgs = [resizeCropByWidth(img,512) for img in imgs]
json_filenames = get_filenames(FilePath,"json")
print(json_filenames)
poses = [get_json_data(FilePath+name) for name in json_filenames]
poses = [(512*pose//width).astype(np.int) for pose,width in zip(poses,old_widths)]
labels = [pos2imgLine(pos,finalHeight,finalWidth,lineWidth=3) for pos in poses]
model = create_hourglass_network(3,1,128,(1024,512),(256,128),bottleneck_block)
model.fit(np.array(imgs)/255,np.array(labels),1,epochs=100)
model.save("1Foot100epoch3width.hdf5")
labels = [pos2imgLine(pos,finalHeight,finalWidth,lineWidth=5) for pos in poses]
model = create_hourglass_network(3,1,128,(1024,512),(256,128),bottleneck_block)
model.fit(np.array(imgs)/255,np.array(labels),1,epochs=100)
model.save("1Foot100epoch5width.hdf5")
labels = [pos2imgLine(pos,finalHeight,finalWidth,lineWidth=6) for pos in poses]
model = create_hourglass_network(3,1,128,(1024,512),(256,128),bottleneck_block)
model.fit(np.array(imgs)/255,np.array(labels),1,epochs=100)
model.save("1Foot100epoch6width.hdf5")
# def train_withDataAugument():
#     trainX, trainY = np.array(imgs[:100]) / 255, np.array(labels[:100])
#     valX, valY = np.array(imgs[100:]) / 255, np.array(labels[100:])
#     epochs = 100
#     steps = 100
#     for epoch in range(epochs):
#         for step in range(steps):
#             x, y = trainX[step], trainY[step]
#             x = random_brightness(x, (0.2, 1.5))
#             x = np.expand_dims(x, 0)
#             y = np.expand_dims(y, 0)
#             y = tf.convert_to_tensor(y)
#             with backprop.GradientTape() as tape:
#                 y_pred = model(x, training=True)
#                 loss = model.compiled_loss(
#                     y, y_pred, regularization_losses=model.losses)
#             model.optimizer.minimize(loss, model.trainable_variables, tape=tape)
#             model.compiled_metrics.update_state(y, y_pred)
#         print(f'loss:{model.metrics[0].result()},acc:{model.metrics[1].result()}')
#     model.save("1FootDataAugument.hdf5")
# ES = ks.callbacks.EarlyStopping(patience=3,verbose=2)
# model = tf.keras.models.load_model("myModel_Line.hdf5",compile=True)
# model.fit(np.array(imgs)/255,np.array(labels),1,epochs=50,validation_split=0.1,callbacks=[ES])

# img = cv2.imread("mae/0origin.png")
# img = cv2.resize(img,(50,81))
# cv2.imshow("1",img)
# cv2.waitKey()