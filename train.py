from preprocess import *
from model import *
from postprocess import *
finalHeight = 256
finalWidth = 128
FilePath = "dataset/train/"
img_filenames = get_filenames(FilePath)
print(img_filenames)
imgs = [cv2.imread(FilePath+name) for name in img_filenames]
old_widths = [img.shape[1] for img in imgs]
imgs = [resizeCropByWidth(img,512) for img in imgs]
json_filenames = get_filenames(FilePath,"json")
print(json_filenames)
poses = [get_json_data(FilePath+name) for name in json_filenames]
poses = [(512*pose//width).astype(np.int) for pose,width in zip(poses,old_widths)]
labels = [pos2imgLine(pos,finalHeight,finalWidth,lineWidth=4) for pos in poses]
model = create_hourglass_network(3,1,128,(1024,512),(256,128),bottleneck_block)
model.fit(np.array(imgs)/255,np.array(labels),1,epochs=100)
model.save("100epoch4width.hdf5")