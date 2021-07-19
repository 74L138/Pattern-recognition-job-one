import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# 反相灰度图，将黑白阈值颠倒,挨个像素处理
def accessPiexl(img):
    # change_gray(img)
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
       for j in range(width):
           img[i][j] = 255 - img[i][j]
    return img

# 反相二值化图像
def accessBinary(img, threshold=165):
    img = accessPiexl(img)#反色
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)#高斯滤波 根据高斯的距离对周围的点进行加权,求平均值1，0.8， 0.6， 0.8
    # 进行腐蚀操作，去除边缘毛躁
    img = cv2.erode(img, kernel, iterations=1)
    #利用阈值函数，二值化
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -9)#自适应滤波
    # 边缘膨胀
    img = cv2.dilate(img, kernel, iterations=1)#被执行的次数
    return img

# 显示结果及边框
def showResults(path, borders, results=None):
    img = cv2.imread(path)
    # 绘制
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        if results:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        # cv2.circle(img, border[0], 1, (0, 255, 0), 0)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.imwrite('out.jpg',img)

# 寻找边缘，返回边框的左上角和右下角
def findBorderContours(path, maxArea=1800):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    borders = []
    for contour in contours:
        # 将边缘拟合成一个边框
        x, y, w, h = cv2.boundingRect(contour)
        if w*h > maxArea:
            border = [(x, y), (x+w, y+h)]
            borders.append(border)
    return borders

# path = './data/2.jpg'
# img = cv2.imread(path, 0)
# img = accessBinary(img)
# cv2.imshow('accessBinary', img)
# cv2.waitKey(0)
# borders = findBorderContours(path)
# showResults(path, borders)

def transUSPS(path, borders, size=(28, 28)):
    imgData = np.zeros((len(borders), size[0], size[0], 1), dtype='uint8')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    # print(img)
    # print(borders)[[(171, 305), (209, 419)], [(396, 299), (450, 366)], [(683, 267), (743, 384)], [(1044, 266), (1094, 336)],
    # [(1080, 247), (1135, 268)], [(862, 246), (943, 380)], [(1244, 186), (1289, 360)], [(359, 27), (370, 39)]]
    for i, border in enumerate(borders):
        borderImg = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
        # 根据最大边缘拓展像素
        extendPiexl = (max(borderImg.shape) - min(borderImg.shape)) // 2
        targetImg = cv2.copyMakeBorder(borderImg, 7, 7, extendPiexl + 7, extendPiexl + 7, cv2.BORDER_CONSTANT)
        targetImg = cv2.resize(targetImg, size)
        targetImg = np.expand_dims(targetImg, axis=-1)
        imgData[i] = targetImg
    return imgData

prediction = []
def predict(modelpath, imgData):
    model = torch.load(modelpath)
    with torch.no_grad():
        for i, data in enumerate(imgData):
            data = data.type(torch.FloatTensor)
            print(data)
            test_pred = model(data.cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                prediction.append(y)

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

path = './data/2.jpg'
model = 'model.pt'
borders = findBorderContours(path)

# imgData = transUSPS(path, borders)
# for i, img in enumerate(imgData):
#     cv2.imshow('test', img)
#     cv2.waitKey(0)
#     name = 'extract/test_' + str(i) + '.jpg'
#     cv2.imwrite(name, img)



imgData = transUSPS(path, borders)
imgData = np.transpose(imgData,[0,3,1,2])
imgData = ImgDataset(imgData)
test_loader = DataLoader(imgData, batch_size=1, shuffle=False)
results = predict(model, test_loader)
showResults(path, borders, prediction)
