# face-xray
这个版本可以用opencv-python           3.4.7.28


高版本的cv2不兼容，不知道哪个版本开始的。。。


我是用的celeba数据集，男女分开换脸的。

首先要用shape_predictor_68_face_landmarks.dat获取每个人脸图片的68个特征点坐标。
坐标记录在landmark_db.txt里面。

然后就可以直接换脸了，不要男女图片都放一起换，换出来真的丑的一批。
数据集这一块也是坑，celeb没有直接给1024的高清图，需要借助脚本自己去变大。

最坑的就是训练部分了，我的妈，mask和label明明就相当于两个label。我他妈的服了，这怎么放模型里预测。

生成数据集这部分，在https://github.com/yakamoz5/Uni 这人的代码上改动了一下。
他压根没想用到mask，所以在他的报告里直接说这xray不太行，但是换脸方法不是xray的啊，明明就是简单的faceswap。算个锤子的xray方法。



