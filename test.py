from preprocess import *
from model import *
from postprocess import *
from matplotlib import pyplot as plt
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
prefix = "valRes/"
os.makedirs(prefix,exist_ok=True)
os.makedirs(prefix+"WELSCH/",exist_ok=True)
#保存预测的结果
for x in range(len(imgs)):
    predict = model.predict_step(np.expand_dims(imgs[x]/255,0))[0]
    idx = img_filenames[x].split('.')[0]
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
