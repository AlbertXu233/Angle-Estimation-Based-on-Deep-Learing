from preprocess import *
from model import *
from postprocess import *
finalHeight = 256
finalWidth = 128
FilePath = "dataset/val/"
img_filenames = get_filenames(FilePath)
print(img_filenames)
imgs = [cv2.imread(FilePath+name) for name in img_filenames]
old_widths = [img.shape[1] for img in imgs]
imgs = [resizeCropByWidth(img,512) for img in imgs]
json_filenames = get_filenames(FilePath,"json")
print(json_filenames)
poses = [get_json_data(FilePath+name) for name in json_filenames]
poses = [(512*pose//width).astype(np.int) for pose,width in zip(poses,old_widths)]
labels = [pos2imgLine(pos,finalHeight,finalWidth,lineWidth=4,lineType=cv2.LINE_AA) for pos in poses]
model = tf.keras.models.load_model("checkPoints/100epoch4width.hdf5",compile=False)
# 衡量预测的角度偏差
prefix = 'valRes/'
AngleErrors = []
DetectedNum = 0
for x in range(len(imgs)):
    predict = model.predict_step(np.expand_dims(imgs[x] / 255, 0))[0]
    print(x)
    res = calAngleError(imgs[x],predict,poses[x],cv2.DIST_WELSCH)
    if res[0]:
        if np.max(res[1])<5:
            AngleErrors.append(res[1])
            DetectedNum +=1
    print(res)
AngleErrors = np.array(AngleErrors)
print(f'maxium errors:{np.max(AngleErrors)}')
np.savetxt(prefix+'AngleErrors.txt',AngleErrors)
print(np.mean(np.abs(AngleErrors), 0),'DetectedNum:',DetectedNum)