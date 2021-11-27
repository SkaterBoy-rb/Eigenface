'''
Created on 2018年11月19日

@author: coderwangson
'''
"#codeing=utf-8"
import numpy as np
import cv2 as cv
import os
import tkinter as tk
import tkinter.filedialog
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
IMAGE_SIZE = (50, 50)
file_num = 40

def revive(array, size_img = IMAGE_SIZE):
    array_ = np.array(array)
    img = array_.reshape(size_img)
    img = cv.resize(img, (92*3, 112*3))
    return img

def createDatabase(path, num = file_num):
    # 查看路径下所有文件
    TrainFiles = os.listdir(path)
    T = []
    # 计算有几个文件（图片命名都是以 序号.jpg方式）减去Thumbs.db
    for i in range(1, num + 1):
        Train_Number = 5 + 1
        sub_path = path + "s" + str(i) + "/"
        # 把所有图片转为1-D并存入T中
        for j in range(1, Train_Number):
            image = cv.imread(sub_path + str(j) + '.pgm', cv.IMREAD_GRAYSCALE)
            image=cv.resize(image,IMAGE_SIZE)
            # 转为1-D
            image = image.reshape(image.size,1)
            T.append(image)
    T = np.array(T)
        # 不能直接T.reshape(T.shape[1],T.shape[0]) 这样会打乱顺序，
    T = T.reshape(T.shape[0],T.shape[1])
    return np.mat(T).T

def eigenfaceCore(T, num = file_num*5):
    # 把均值变为0 axis = 1代表对各行求均值
    m = T.mean(axis = 1)
    m_temp = np.mean(T, axis=1).astype(np.uint8)
    # cv.imshow("平均脸", revive(m_temp).astype(np.uint8))
    # cv.waitKey()
    A = T-m
    L = np.dot(A.T, A) / A.shape[1]
#     L = np.cov(A,rowvar = 0)
    # 计算AT *A的 特征向量和特征值V是特征值，D是特征向量
    V, D = np.linalg.eig(L)
    index_V = np.argsort(-V)
    index_V = index_V[:num]
    """
    # L_eig = []
    # for i in range(num):
    #     L_eig.append(D[index_V[i], :])
    # # for i in range(A.shape[1]):
    # #     L_eig.append(D[:, i])
    # L_eig = np.mat(np.reshape(np.array(L_eig),(-1,len(L_eig))))
    # #print((L_eig == D.T).all)
    # # 计算 A *AT的特征向量
    # eigenface = A * L_eig
    """
    eigenface = A * D
    eigenface = eigenface/np.linalg.norm(eigenface, axis=0) #归一化特征向量
    eigenface = eigenface[:, index_V]
    imgs = []
    for i in range(5):
        imgs.append(revive((eigenface + m)[:,i]).astype(np.uint8))
    imgs = np.hstack(imgs)
    # cv.imshow("eigenface", imgs)
    # cv.waitKey()
    return eigenface,m,A

def recognize(testImage, eigenface,m,A, show = True):
    #_,trainNumber = np.shape(eigenface)
    projectedImage = eigenface.T*(A)
    _, trainNumber = np.shape(projectedImage)

    testImageArray = cv.imdecode(np.fromfile(testImage,dtype=np.uint8),cv.IMREAD_GRAYSCALE)
    testImageArray=cv.resize(testImageArray,IMAGE_SIZE)
    testImageArray = testImageArray.reshape(testImageArray.size,1)
    testImageArray = np.mat(np.array(testImageArray))
    differenceTestImage = testImageArray - m
    projectedTestImage = np.dot(eigenface.T, differenceTestImage)
    temp = m + np.dot(eigenface, projectedTestImage)
    if show:
        myreconstruct(temp)
    distance = []
    for i in range(0, trainNumber):
        q = projectedImage[:, i]
        # 求范式
        temp = np.linalg.norm(projectedTestImage - q)
        distance.append(temp)

    minDistance = min(distance)
    index = distance.index(minDistance)
    res = (int)(index / 5) + 1
    index_pgm = (index % 5) + 1
    if show:
        img = cv.imread('./picture/s' + str(res) + '/' + str(index_pgm) + '.pgm', cv.IMREAD_GRAYSCALE)
        cv.imshow(str(res) + '.pgm', cv.resize(img, (92 * 3, 112 * 3)))
        cv.waitKey()
    return res

def myreconstruct(temp):
    cv.imshow("eigenface", revive(temp).astype(np.uint8))
    cv.moveWindow("eigenface", 500, 500)
    cv.waitKey()

def mytrain():
    T = createDatabase('./picture/')
    with open('./model.txt', 'w') as f:
        np.savetxt(f, T)


#点击选择图片时调用
def mytest(filename):
    testimage = filename
    with open('./model.txt', 'r') as f:
        T = np.loadtxt(f)
    eigenface, m, A = eigenfaceCore(np.mat(T))
    print(recognize(testimage, eigenface, m, A))


def rank():
    with open('./model.txt', 'r') as f:
        T = np.loadtxt(f)

    path = "./picture/"
    xlist = [10, 25, 50, 100, 200]
    finished_list = []
    for i in range(len(xlist)):
        eigenface, m, A = eigenfaceCore(np.mat(T), xlist[i])
        num = 0
        for j in range(1, file_num+1):
            Train_Number = 10 + 1
            sub_path = path + "s" + str(j) + "/"
            # 把所有图片转为1-D并存入T中
            for k in range(6, Train_Number):
                res = recognize(sub_path + str(k) + '.pgm', eigenface, m, A, False)
                if res == j:
                    num += 1

        finished_list.append(num/(file_num*5) * 100)

    plt.figure()
    plt.xlabel("PCs数")  # x轴上的名字
    plt.ylabel("Rank_1识别率")  # y轴上的名字
    plt.plot(np.array(xlist), np.array(finished_list), 'r', marker='o', linestyle='-', linewidth=2, label='t')
    plt.legend(loc='best')
    plt.show()

# 构建可视化界面
def gui():
    root = tk.Tk()
    root.title("pca face")
    mytrain()
    #np.savetxt()


    def select():
        filename = tkinter.filedialog.askopenfilename()
        if filename != '':
            s=filename # pgm图片文件名 和 路径。
            im=Image.open(s)
            tkimg=ImageTk.PhotoImage(im) # 执行此函数之前， Tk() 必须已经实例化。
            l.config(image=tkimg)
            btn1.config(command=lambda : mytest(filename))
            btn1.config(text = "开始识别")
            btn1.pack()
            # 重新绘制
            root.mainloop()
    # 显示图片的位置
    l = tk.Label(root)
    l.pack()

    l1 = tk.Label(root)
    l1.pack()

    btn = tk.Button(root,text="选择识别的图片",command=select)
    btn.pack()

    btn2 = tk.Button(root, text="批量识别", command=rank)
    btn2.pack()
    
    btn1 = tk.Button(root) # 开始识别按钮，刚开始不显示
    root.mainloop()
if __name__ == "__main__":
    gui()
