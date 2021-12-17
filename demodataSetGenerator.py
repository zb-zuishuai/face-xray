import numpy as np
import os
import random
import cv2
import json
import utils as ut
from DeepFakeMask import dfl_full, facehull, components, extended
from skimage import io
from skimage import transform as sktransform
from PIL import Image
from imgaug import augmenters as iaa
from tqdm import tqdm



# Path to all pre-computed facial landmarks.所有预先计算的面部标志的路径。
# Pre compute them using the landmarkDBGenerator.py使用landmarkDBGenerator.py预先计算它们
landmark_path = "E:/facexray/face-x-ray-master/dataset/landmarks/landmark_db.txt"

# path to images. Should of course match the landmarks图像的路径。当然应该与地标相匹配
img_dir_path = "E:/facexray/face-x-ray-master/dataset/images/original/"

# Path to Directory where new images will be stored、存储新映像的目录的路径
data_set_path = "E:/facexray/face-x-ray-master/dataset/images"


def random_get_hull(landmark, img1):
    # 使用facehull方法
    # hull_type = random.choice([0, 1, 2, 3])
    hull_type = 3
    if hull_type == 0:
        mask = dfl_full(landmarks=landmark.astype(
            'int32'), face=img1, channels=3).mask
        return mask / 255
    elif hull_type == 1:
        mask = extended(landmarks=landmark.astype(
            'int32'), face=img1, channels=3).mask
        return mask / 255
    elif hull_type == 2:
        mask = components(landmarks=landmark.astype(
            'int32'), face=img1, channels=3).mask
        return mask / 255
    elif hull_type == 3:
        mask = facehull(landmarks=landmark.astype(
            'int32'), face=img1, channels=3).mask
        return mask / 255


def random_erode_dilate(mask, ksize=None):
    # 使用cv2腐蚀和膨胀进行掩模加工
    # 使用此方法创建的假图应在其性能上有所不同
    # 遮罩边框
    if random.random() > 0.5:
        if ksize is None:
            ksize = random.randint(1, 5)#有改动
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8) * 255
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.erode(mask, kernel, 1) / 255
    else:
        if ksize is None:
            ksize = random.randint(1, 5)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8) * 255
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.dilate(mask, kernel, 1) / 255
    return mask


# borrow from https://github.com/MarekKowalski/FaceSwap
def blendImages(src, dst, mask, featherAmount=0.2):

    maskIndices = np.where(mask != 0)

    # prepare mask for src and dst为src和dst准备掩码
    src_mask = np.ones_like(mask)
    dst_mask = np.zeros_like(mask)

    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

    # weight by distance to nearest contour edge of mask (hull)到遮罩（外壳）最近轮廓边缘的距离权重
    # this ensures more smooth borders when masks这样可以确保遮罩时边界更加平滑
    weights = np.clip(dists / featherAmount, 0, 1)

    # depending on weights, blend images根据权重，混合图像
    # take weights proportion from src从src获取权重比例
    # take (1-weights) proportion from dst从dst中获取（1-权重）比例
    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0],
                                                                               maskIndices[1]] + (
                                                              1 - weights[:, np.newaxis]) * dst[
                                                      maskIndices[0], maskIndices[1]]

    # 不使用这个。
    composedMask = np.copy(dst_mask)
    composedMask[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src_mask[maskIndices[0], maskIndices[1]] + (
            1 - weights[:, np.newaxis]) * dst_mask[maskIndices[0], maskIndices[1]]

    return composedImg, composedMask


# borrow from https://github.com/MarekKowalski/FaceSwap
def colorTransfer(src, dst, mask):
    """
    src, dst        images, numpy arrays
    mask            image mask, numpy array

    Color correct the destination face region given source对给定源的目标面区域进行颜色校正
    """

    transferredDst = np.copy(dst)

    # indices of mask > 0掩模指数>0
    maskIndices = np.where(mask != 0)

    # Given mask, take from src and dest the corresponding regions给定掩码，从src中提取并删除相应区域
    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    # calculate means of both face regions计算两个面区域的平均值
    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    # color correct as done in FaceSwap在FaceSwap中进行颜色校正
    # first subtract mean of dst from dst首先从dst中减去dst的平均值
    # then add mean of src to dst然后将src的平均值添加到dst中
    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    # return new dst image where mask region is color corrected返回经过颜色校正的新dst图像
    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst


class DataSetGenerator():
    def __init__(self, landmarks_db_path=landmark_path,
                 image_path=img_dir_path, data_set_path=data_set_path):
        self.landmarks_db = self._read_landmarks_pairs(landmarks_db_path)
        self.image_names = self._get_images(image_path)
        self.image_path = image_path
        self.data_set_path = data_set_path
        # piecewise affine transform. See XrayDemo notebook (at the bottom)分段仿射变换。参见XrayDemo notebook（底部）
        self.distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.05))])

    def _get_images(self, img_dir_path):
        return [i for i in os.listdir(img_dir_path) if i.lower().endswith((".png", ".jpg", ".jpeg"))]

    def _read_landmarks_pairs(self, landmark_path):
        with open(landmark_path, 'r') as myfile:
            landmark_db_f = myfile.read()
        return json.loads(landmark_db_f)

    def get_blended_face(self, background_face_path):

        # 1. load image and landmark加载图像和地标
        background_face = io.imread(self.image_path + background_face_path)
        background_landmark = np.array(self.landmarks_db[background_face_path])
        im_y = background_face.shape[0]
        im_x = background_face.shape[1]

        # 2. get nearest face最近搜索
        foreground_face_path = ut.get_nearest_face(background_face_path,
                                                   self.landmarks_db)
        foreground_face = io.imread(self.image_path + foreground_face_path)

        # 3. down sample randomly before blending混合前随机取下样品
        down_sample_factor = random.uniform(0.5, 1)
        aug_size_y = int(im_y * down_sample_factor)
        aug_size_x = int(im_x * down_sample_factor)

        background_landmark[:, 0] = background_landmark[:, 0] * (aug_size_x / im_x)
        background_landmark[:, 1] = background_landmark[:, 1] * (aug_size_y / im_y)

        foreground_face = sktransform.resize(foreground_face, (aug_size_y, aug_size_x), preserve_range=True).astype(
            np.uint8)
        background_face = sktransform.resize(background_face, (aug_size_y, aug_size_x), preserve_range=True).astype(
            np.uint8)

        # 4. get face_hull face mask戴上遮罩
        mask = random_get_hull(background_landmark, background_face)

        # 5. random deform mask随机变形遮罩
        mask = self.distortion.augment_image(mask)
        mask = random_erode_dilate(mask)

        # filter empty mask after deformation过滤空的
        if np.sum(mask) == 0:
            raise NotImplementedError

        # 6. apply color transfer应用颜色转移
        foreground_face = colorTransfer(background_face, foreground_face, mask * 255)

        # 7. blend faces
        blended_face, mask = blendImages(foreground_face, background_face, mask * 255)
        blended_face = blended_face.astype(np.uint8)

        # 8. resize back to default resolution将大小调整回默认分辨率
        blended_face = sktransform.resize(blended_face, (im_y, im_x), preserve_range=True).astype(np.uint8)
        mask = sktransform.resize(mask, (im_y, im_x), preserve_range=True)
        mask = mask[:, :, 0:1]
        return blended_face, mask

    def create_dataset(self):

        for img_name in tqdm(self.image_names):
            background_face_path = img_name
            fake = random.randint(0, 1)

            if self.landmarks_db.get(img_name) == None:
                continue
            if fake:
                # do fake processing steps from above.
                # see get_blended_face
                try:
                    face_img, mask = self.get_blended_face(background_face_path)
                    mask = (1 - mask) * mask * 4
                except:
                    continue
            else:
                # image will not be faked. this will be a real image in our
                # dataset
                face_img = io.imread(self.image_path + background_face_path)
                mask = np.zeros((face_img.shape[0], face_img.shape[1], 1))

            im_y = face_img.shape[0]
            im_x = face_img.shape[1]

            # randomly downsample after BI pipeline
            # valid for reals and fakes
            if random.randint(0, 1):
                down_sample_factor = random.uniform(0.6, 1)

                aug_size_y = int(im_y * down_sample_factor)
                aug_size_x = int(im_x * down_sample_factor)

                face_img = Image.fromarray(face_img)#实现array到image的转换

                if random.randint(0, 1):
                    face_img = face_img.resize((aug_size_x, aug_size_y), Image.BILINEAR)#双线性插值（bilinear）
                else:
                    face_img = face_img.resize((aug_size_x, aug_size_y), Image.NEAREST)#最近邻插值（nearest）

                face_img = face_img.resize((im_x, im_y), Image.BILINEAR)

                face_img = np.array(face_img)



            # random flip
            if random.randint(0, 1):
                face_img = np.flip(face_img, 1)
                mask = np.flip(mask, 1)

            #混合图保存
            im = Image.fromarray(face_img)
            im.save(self.data_set_path+ ("/fake/" if fake else "/real/")+ img_name.split(".")[0]+ ("_fake" if fake else "_real") + ".jpeg",quality=random.randint(95, 100))

            #添加的，遮罩保存
            '''
            np.array([mask for i in range(3)]).transpose(1,2,0)
            or
            np.repeat(mask[...,np.newaxis],3,2)
            其中mask的shape是(height,width).如果是(height,width,1)
            那么应用第一种方法就需要删除维度,第二种方法就需要去掉[...,np.newaxis]这一部分。
            #np.newaxis是新建一个维度
            '''
            mask = np.repeat(mask, 3, 2)#变成三通道
            immask = Image.fromarray((mask * 255).astype(np.uint8))
            immask.save(self.data_set_path+ ("/fakemask/" if fake else "/realmask/")+ img_name.split(".")[0]+ ("_fake" if fake else "_real") + ".jpeg",quality=random.randint(95, 100))




def main():
    dataSetGenerator = DataSetGenerator()
    dataSetGenerator.create_dataset()


if __name__ == "__main__":
    main()
