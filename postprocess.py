import cv2
import numpy as np
def draw_keyPoints(image,pnts):
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    for i in range(len(pnts)//2):
        cv2.circle(image,(pnts[2*i],pnts[2*i+1]),3,colors[i//(len(pnts)//6)],4)
def draw_lines(image,pnts,lwidth=4):
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    dxy=[]
    angleLabel=['a: ','b: ']
    for i in range(3):
        s = 4*i
        if abs(pnts[s+1] - pnts[s+3])>0.01:
            topx = (pnts[s]-pnts[s+2])/(pnts[s+1]-pnts[s+3])*(-pnts[s+1]) + pnts[s]
            bottomx = (pnts[s]-pnts[s+2])/(pnts[s+1]-pnts[s+3])*(1024-pnts[s+1]) + pnts[s]
            cv2.line(image, (int(topx), 0), (int(bottomx), 1024), colors[i], lwidth, cv2.LINE_AA)
            dx = (bottomx - topx)/1024
            dxy.append(np.array([dx, 1]) / np.sqrt(dx**2+1))
        else:
            cv2.line(image, (0, pnts[s+1]), (512, pnts[s+1]), colors[i], lwidth, cv2.LINE_AA)
            dxy.append(np.array([1,0]))
    for c in range(2):
        angle = np.arccos(np.dot(dxy[c],dxy[c+1]))*180/3.14
        cv2.putText(image,angleLabel[c]+"%.2f"%angle,(30,40*c+40),1,3,(255,0,0),thickness=3)
def heatmap2lines(image,heatmap):
    h,w,_ = image.shape
    pnts = []
    for i in range(heatmap.shape[-1]):
        y,x = np.where(heatmap[:,:,i]==np.max(heatmap[:,:,i]))
        pnts.append(x[0])
        pnts.append(y[0])
    pnts = np.array(pnts)*w//128
    draw_keyPoints(image,pnts)
    draw_lines(image,pnts)
    return image
def heatmap2linesMul(image,heatmap):
    h,w,_ = image.shape
    pnts = []
    for i in range(heatmap.shape[-1]):
        y,x = np.where(heatmap[:,:,i]==np.max(heatmap[:,:,i]))
        pnts.append(x[0])
        pnts.append(y[0])
    pnts = np.array(pnts)*w//128
    draw_keyPoints(image,pnts)
    pntsAll = np.reshape(pnts,[3,len(pnts)//6,2])
    angleLabel = ["a: ", "b: "]
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    dxy = []
    for c in range(3):
        pnts = pntsAll[c,:,::-1]
        dy, dx, y0, x0 = cv2.fitLine(pnts, cv2.DIST_L1, 0, 0.01, 0.01)
        if abs(dy) > 0.01:
            top_x = dx / dy * (-y0) + x0
            bottom_x = dx / dy * (h - y0) + x0
            cv2.line(image, (int(top_x), 0), (int(bottom_x), int(h)), colors[c], 2, cv2.LINE_AA)
        else:
            cv2.line(image, (0, int(y0)), (int(w), int(y0)), colors[c], 2, cv2.LINE_AA)
        # print(top_x,bottom_x,dy)
        dxy.append(np.array([dx[0], dy[0]]))
    if (len(dxy) == 3):
        for c in range(2):
            angle = np.arccos(np.dot(dxy[c], dxy[c + 1])) * 180 / 3.14
            cv2.putText(image, angleLabel[c] + "%.2f" % angle, (30, 40 * c + 40), 1, 3, (255, 0, 0), thickness=3)
    return image
def LinesOnImage(image,heatmap):
    # heatmap = np.where(heatmap>0.5,255,0).astype(np.uint8)
    heatmap = (255*heatmap).numpy().astype(np.uint8)
    h,w,_ = image.shape
    n_heatmap = np.zeros([h,w,_])
    for c in range(heatmap.shape[2]):
        n_heatmap[:,:,c]=cv2.resize(heatmap[:,:,c],(w,h))
    heatmap = np.array(n_heatmap,np.uint8)
    image = cv2.add(image,heatmap)
    # image = (image + heatmap)//2
    return image.astype(np.uint8)
def line_regression(image,heatmap,type=cv2.DIST_WELSCH,constant=0):
    heatmap = heatmap.numpy()
    for c in range(heatmap.shape[-1]):
        heatmap[:,:,c] = heatmap[:,:,c]/np.max(heatmap[:,:,c])
    heatmap = np.where(heatmap>0.5,1,0).astype(np.uint8)
    colors = [(0,0,255),(0,255,0),(255,0,0)]
    h,w,_ = image.shape
    n_heatmap = np.zeros([h,w,_])
    for c in range(heatmap.shape[2]):
        n_heatmap[:,:,c]=cv2.resize(heatmap[:,:,c],(w,h))
    heatmap = np.array(n_heatmap,np.uint8)
    dxy = []
    angleLabel = ["a: ", "b: "]
    for c in range(heatmap.shape[2]):
        pnts = np.where(heatmap[:,:,c])
        if len(pnts[0]) == 0:
            continue
        pnts = np.swapaxes(pnts,0,-1)
        dy,dx,y0,x0 = cv2.fitLine(pnts,type,constant,0.01,0.01)
        if abs(dy) > 0.01:
            top_x = dx/dy*(-y0) +x0
            bottom_x = dx/dy*(h-y0) + x0
            cv2.line(image, (int(top_x), 0), (int(bottom_x), int(h)), colors[c], 2, cv2.LINE_AA)
        else:
            cv2.line(image, (0, int(y0)), (int(w), int(y0)), colors[c], 2, cv2.LINE_AA)
        # print(top_x,bottom_x,dy)
        dxy.append(np.array([dx[0],dy[0]]))
    if (len(dxy)==3):
        for c in range(2):
            angle = np.arccos(np.dot(dxy[c],dxy[c+1]))*180/3.14
            cv2.putText(image,angleLabel[c]+"%.2f"%angle,(30,40*c+40),1,3,(0,0,255),thickness=3)
    return image
def heatmap2scatter(image,heatmap):
    heatmap = np.where(heatmap>0.5,1,0).astype(np.uint8)
    colors = [(0,0,255),(0,255,0),(255,0,0)]
    h,w,_ = image.shape
    n_heatmap = np.zeros([h,w,_])
    for c in range(heatmap.shape[2]):
        n_heatmap[:,:,c]=cv2.resize(heatmap[:,:,c],(w,h))
    heatmap = np.array(n_heatmap,np.uint8)
    for c in range(heatmap.shape[2]):
        pnts = np.where(heatmap[:,:,c])
        if len(pnts[0]) == 0:
            continue
        pnts = np.swapaxes(pnts,0,-1)
        pnts = pnts[::16,1::-1]
        for x,y in pnts:
            image[y,x] = colors[c]
    return image
def calAngleError(image,heatmap,pos,type=cv2.DIST_WELSCH,const=0):
    dxy = []
    gtLineAngles = []
    for i in range(3):
        dx = pos[4*i+2] - pos[4*i]
        dy = pos[4*i+3] - pos[4*i+1]
        gtLineAngles.append(np.array([dx,dy])/np.sqrt(dx*dx+dy*dy))

    for c in range(heatmap.shape[2]):
        pnts = np.where(heatmap[:,:,c] > 0.5*np.max(heatmap[:,:,c]))
        if len(pnts[0]) == 0:
            continue
        pnts = np.swapaxes(pnts,0,-1)
        dy,dx,y0,x0 = cv2.fitLine(pnts,type,const,0.01,0.01)
        dxy.append(np.array([dx[0],dy[0]]))
    if (len(dxy)==3):
        angleDiffs = []
        for c in range(2):
            angle = np.arccos(np.abs(np.dot(dxy[c],dxy[c+1])))*180/3.14
            gtAngle = np.arccos(np.abs(np.dot(gtLineAngles[c],gtLineAngles[c+1])))*180/3.14
            diff = abs(angle -gtAngle)
            # if diff>90:
            #     diff = 180-diff
            angleDiffs.append(diff)
        return True,np.array(angleDiffs)
    else:
        return [False]
def calAngleErrorKey(heatmap,pos):
    dxy = []
    gtLineAngles = []
    for i in range(3):
        dx = pos[4*i+2] - pos[4*i]
        dy = pos[4*i+3] - pos[4*i+1]
        gtLineAngles.append(np.array([dx,dy])/np.sqrt(dx*dx+dy*dy))
    pnts = []
    for i in range(heatmap.shape[-1]):
        y,x = np.where(heatmap[:,:,i]==np.max(heatmap[:,:,i]))
        pnts.append(x[0])
        pnts.append(y[0])
    pnts = np.array(pnts)*4
    for i in range(heatmap.shape[-1]//2):
        s = 4*i
        x1,y1,x2,y2 = pnts[s:s+4]
        dx = x2-x1
        dy = y2-y1
        dxy.append(np.array([dx,dy])/np.sqrt(dx**2+dy**2))
    if (len(dxy)==3):
        angleDiffs = []
        for c in range(2):
            angle = np.arccos(np.abs(np.dot(dxy[c],dxy[c+1])))*180/3.14
            gtAngle = np.arccos(np.abs(np.dot(gtLineAngles[c],gtLineAngles[c+1])))*180/3.14
            diff = abs(angle -gtAngle)
            # if diff>90:
            #     diff = 180-diff
            angleDiffs.append(diff)
        return True,np.array(angleDiffs)
    else:
        return [False]