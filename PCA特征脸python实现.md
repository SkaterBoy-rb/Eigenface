﻿# PCA特征脸python实现  

## PCA原理   

PCA全名为主成分分析，其主要目的就是寻找一个矩阵，然后把原来的一组带有相关性的矩阵映射到寻找到的那个矩阵中，达到降维的目的。**一般的，如果我们有M个N维向量，想将其变换为由R个N维向量表示的新空间中，那么首先将R个基按行组成矩阵A，然后将向量按列组成矩阵B，那么两矩阵的乘积AB就是变换结果，其中AB的第m列为A中第m列变换后的结果。** 这句话就相当于找到了一个R行N列矩阵，然后乘一个N行M列矩阵，这样就得到了一个R行M列矩阵（其中R<=N），达到降维的目的。其中M和N的含义为，M可以代表样本个数，而N代表每个样本的特征个数，所以最终结果就是把原来N个特征变为了R个特征，达到降维目的。  

## PCA算法描述   

1、构建一个样本集合$S =\{T_1,T_2,...,T_M\}$,$S$ 可以看做是一个N行M列的矩阵，也就是有M个样本，每个样本有N个特征。其中$T_i$是一个向量。
2、0均值化，为了便于计算方差的时候需要减去均值，所以如果本身样本是零均值的，就方便计算。  

$m = \frac{1}{M}\sum_{i=1}^{M}T_i$ ,这个是计算均值在python中可以使用  

    m = T.mean(axis = 1)  
    
进行计算，其中axis = 1代表按行求均值。  
然后$A = T -m$ 这个相当于把每个样本都减去均值，这样之后就相当于做了0均值化。   

3、计算投影矩阵（就是相当于上面的那个R行M列矩阵）  
这个投影矩阵其实就是由$A*A^T$矩阵的特征向量构成，但是由于大多数情况$A*A^T$的维度太大（$A*A^T$是N行N列矩阵，如果是一张图片的话N就代表像素点个数，所以是相当大的），所以这个时候就利用数学的小技巧转化为先求$A^T*A$的特征向量矩阵V，其中V的每一列是一个特征向量，那么V是一个M行M列的矩阵，然后我们再从V中取出前R个最大特征值对应的特征向量，所以V就变成了M行R列矩阵，然后$C = AV$,那么这个C矩阵就是计算出的投影矩阵，C为一个N行R列的矩阵。  

![](../img/PCA特征脸python实现_1.jpg)  

4、把原来样本进行投影  

第三步我们得到了一个N行R列的矩阵C，其中每一列是一个特征向量，但是我们在讲PCA原理的时候我们需要一个R行N列的矩阵，每一行是一个特征向量，所以我们可以使用$C^T$,所以我们投影后的样本变为$P = C^T A$ 其中P就是一个R行M列的矩阵，可以看出已经达到了降维的目的。  

## python实现特征脸   

### 特征脸   

特征脸就是我们上面求得的C矩阵，所谓的基于特征脸进行的人脸识别，就是先把人脸映射到一个低纬空间，然后再计算映射后的脸之间的距离，把距离最近的两个特征脸归为同一个人的脸。  

所以特征脸的步骤为：  

1、加载训练集中的脸，转为一个M行N列矩阵T  

2、对T进行0均值化  

3、找到T的投影矩阵C  

4、计算投影后的矩阵P  

5、加载一个测试图片，并利用C矩阵也把其投影为test_P  

6、计算test_P和P中每个样本的距离，选出最近的那个即可  

### python代码  


    def eigenfaceCore(T, num = 100):
        # 把均值变为0 axis = 1代表对各行求均值
        m = T.mean(axis = 1)
        m_temp = np.mean(T, axis=1).astype(np.uint8)
        # cv.imshow("平均脸", revive(m_temp).astype(np.uint8))
        # cv.waitKey()
        A = T-m
        L = (A.T)*(A)
        #     L = np.cov(A,rowvar = 0)
        # 计算AT *A的 特征向量和特征值V是特征值，D是特征向量
        V, D = np.linalg.eig(L)
        index_V = np.argsort(-V)
        L_eig = []
        for i in range(num):
            L_eig.append(D[index_V[i], :])
        # for i in range(A.shape[1]):
        #     L_eig.append(D[:, i])
        L_eig = np.mat(np.reshape(np.array(L_eig),(-1,len(L_eig))))
        #print((L_eig == D.T).all)
        # 计算 A *AT的特征向量
        eigenface = A * L_eig
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
        projectedTestImage = eigenface.T*(differenceTestImage)
        temp = np.matmul( eigenface, projectedTestImage)
        if show:
            cv.imshow("eigenface", revive(temp + m).astype(np.uint8))
            cv.moveWindow("eigenface", 500, 500)
            cv.waitKey()
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
        
        
上面就是全部的基于特征脸的人脸识别。    


## 参考  

> [pca原理]http://blog.codinglabs.org/articles/pca-tutorial.html  
> [特征脸原理]https://blog.csdn.net/smartempire/article/details/21406005  
> https://blog.csdn.net/qq_16936725/article/details/51761685
> https://blog.csdn.net/zawdd/article/details/8087280  
> https://www.zhihu.com/question/67157462/answer/251754530
> https://bbs.csdn.net/topics/391905372
https://blog.csdn.net/tinym87/article/details/6957438
https://blog.csdn.net/fjdmy001/article/details/78498150?locationNum=1&fps=1
https://blog.csdn.net/Abit_Go/article/details/77938938
https://fishc.com.cn/thread-73738-1-1.html
https://blog.csdn.net/jjjndk1314/article/details/80620139
https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html


  
