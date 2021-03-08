import os
import matplotlib.pyplot as plt
import cv2
import re
import numpy as np

file = r'/home/yuzhengbo/下载/mAP-master/input'
file1 = file+'/'+os.listdir(file)[0]+'/'+'I48E001004_wow_462_max.txt'    #用于存放测试真实框文件目录
file2 = file+'/'+os.listdir(file)[1]+'/'+'I48E001004_wow_462_max.txt'    #存放预测框文件目录
file3 = '/home/yuzhengbo/下载/yolov4-pytorch-master/VOCdevkit/VOC2007/JPEGImages/I48E001004_wow_462_max.jpg'     #测试的图片位置
# print(file1,'\n',file2)
def ax_real(file1):     #得到真实框的位置为list
    real = []
    with open(file1,'r')as f:
       real.append(f.readlines())
    real = [i.strip() for i in real[0]]
    # print(real)
    ax =[]
    for i in real:
       ax.append([ int(i) for i in re.findall(' \d+',i)])
    return ax

def ax_pre(file1):      #得到预测框的位置为list
    real = []
    with open(file1,'r')as f:
       real.append(f.readlines())   
    real = [i.strip() for i in real[0]]
    # print(real)
    ax =[]
    for i in real:
       ax.append([ int(i) for i in re.findall(' -*\d+',i) if abs(int(i))>=1 or abs(float(i))<0.01])
    return [i[1:] for i in ax]

ax1 = ax_real(file1)
ax2 = ax_pre(file2)



def iou(predicted_bound, ground_truth_bound):       #计算IOU
    """
    computing the IoU of two boxes.
    Args:
        box: (xmin, ymin, xmax, ymax),通过左下和右上两个顶点坐标来确定矩形位置
    Return:
        IoU: IoU of box1 and box2.
    """
    pxmin, pymin, pxmax, pymax = predicted_bound
    # print("预测框P的坐标是：({}, {}, {}, {})".format(pxmin, pymin, pxmax, pymax))
    gxmin, gymin, gxmax, gymax = ground_truth_bound
    # print("原标记框G的坐标是：({}, {}, {}, {})".format(gxmin, gymin, gxmax, gymax))

    parea = (pxmax - pxmin) * (pymax - pymin)  # 计算P的面积
    garea = (gxmax - gxmin) * (gymax - gymin)  # 计算G的面积
    # print("预测框P的面积是：{}；原标记框G的面积是：{}".format(parea, garea))

    # 求相交矩形的左下和右上顶点坐标(xmin, ymin, xmax, ymax)
    xmin = max(pxmin, gxmin)  # 得到左下顶点的横坐标
    ymin = max(pymin, gymin)  # 得到左下顶点的纵坐标
    xmax = min(pxmax, gxmax)  # 得到右上顶点的横坐标
    ymax = min(pymax, gymax)  # 得到右上顶点的纵坐标

    # 计算相交矩形的面积
    w = xmax - xmin
    h = ymax - ymin
    if w <=0 or h <= 0:
        return 0

    area = w * h  # G∩P的面积
    # area = max(0, xmax - xmin) * max(0, ymax - ymin)  # 可以用一行代码算出来相交矩形的面积
    # print("G∩P的面积是：{}".format(area))

    # 并集的面积 = 两个矩形面积 - 交集面积
    IoU = area / (parea + garea - area)

    return IoU

# print(iou(ax1[0],ax2[0]))


def recall(ax1,ax2):
    l = len(ax1)
    q = 0
    for i in ax1:
        print(i)
        for j in ax2:
            print(j)
            if iou(i,j)>=0.5:
                q += 1
                break
    return q/l

# print(recall(ax1,ax2))

# print(ax1,'\n',ax2)


#在测试图上将真实框和预测框标注出
# img =plt.imread(file3)
# cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
# for i in ax1:
#     cv2.rectangle(img,(i[0],i[1]),(i[2],i[3]),(0,255,0),3)
# for j in ax2:
#     cv2.rectangle(img,(j[0],j[1]),(j[2],i[3]),(255,0,0),3)
# plt.imshow(img)
# plt.show()

def rec(a,b):
    l = len(a)
    q = 0
    lb = len(b)
    for i in a:
       for j in b:
           if iou(i,j)>0.5:
               q += 1
               break
    return float(q/l),q,l,lb        #计算recall,正确预测出目标的框数量,真实框数量,预测框数量


file_pre = '/home/yuzhengbo/下载/mAP-master_1/input/detection-results'     #存放预测框txt文件的父文件夹
file_real = '/home/yuzhengbo/下载/mAP-master_1/input/ground-truth'      #存放真实框txt文件的父文件夹
recall = []
q,l,lb = 0,0,0
for i in os.listdir(file_pre)[1:]:
    # print(i)
    # print(q)
    file1 = file_real+'/'+i
    file2 = file_pre+'/'+i
    # print(file1,'\n',file2)
    ax1 = []
    ax2 = []
    ax1 = ax_real(file1)
    ax2 = ax_pre(file2)
    # print(ax1)
    # print(ax2)
    print(rec(ax1,ax2)[0])
    q += rec(ax1,ax2)[1]
    l += rec(ax1,ax2)[2]
    lb += rec(ax1,ax2)[3]
    # print(np.array(ax1).shape,np.array(ax2).shape)
    # print(ax2)
    # print(ax1)
    # print(ax2)
    
print('recall:',q/l)    
print(q,l,lb)     #正确预测出目标的框数量,真实框数量,预测框数量
    


        