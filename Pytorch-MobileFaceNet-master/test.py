import argparse
import functools
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time

import cv2
import numpy as np
import torch
from PIL import ImageDraw, ImageFont, Image

from detection.face_detect import MTCNN
from utils.utils import add_arguments, print_arguments


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('image_path',               str,     'dataset/test2.jpg',                 '预测图片路径')
add_arg('face_db_path',             str,     'face_db',                          '人脸库路径')
add_arg('threshold',                float,   0.621726,                                '判断相识度的阈值')
add_arg('mobilefacenet_model_path', str,     'mobilefacenet_scripted.pt',     'MobileFaceNet预测模型的路径')
# add_arg('mobilefacenet_model_path', str,     'save_model/mobilefacenet.pth',     'MobileFaceNet预测模型的路径')
add_arg('mtcnn_model_path',         str,     'save_model/mtcnn',                 'MTCNN预测模型的路径')
args = parser.parse_args()
print_arguments(args)


class Predictor:
    def __init__(self, mtcnn_model_path, mobilefacenet_model_path, face_db_path, threshold=0.7):
        self.threshold = threshold
        self.mtcnn = MTCNN(model_path=mtcnn_model_path) # 加载mtcnn人脸检测模型
        self.device = torch.device("cuda")

        # 加载模型
        self.model = torch.jit.load(mobilefacenet_model_path)
        self.model.to(self.device)
        self.model.eval()

        self.faces_db = self.load_face_db(face_db_path)

    def load_face_db(self, face_db_path):
        faces_db = {}
        for path in os.listdir(face_db_path):
            name = os.path.basename(path).split('.')[0]
            image_path = os.path.join(face_db_path, path)
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1) # 从指定的内存缓存中读取数据，并把数据转换(解码)成图像格式
            imgs, _ = self.mtcnn.infer_image(img)
            if imgs is None or len(imgs) > 1:
                print('人脸库中的 %s 图片包含不是1张人脸，自动跳过该图片' % image_path)
                continue
            imgs = self.process(imgs)
            feature = self.infer(imgs[0])
            faces_db[name] = feature[0][0]
        return faces_db

    @staticmethod
    def process(imgs):
        imgs1 = []
        for img in imgs:
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 127.5
            imgs1.append(img)
        return imgs1

    # 预测图片
    def infer(self, imgs):
        assert len(imgs.shape) == 3 or len(imgs.shape) == 4
        if len(imgs.shape) == 3:
            imgs = imgs[np.newaxis, :]
        # TODO 不知为何不支持多张图片预测
        '''
        imgs = torch.tensor(imgs, dtype=torch.float32, device=self.device)
        features = self.model(img)
        features = features.detach().cpu().numpy()
        '''
        features = []
        for i in range(imgs.shape[0]):
            img = imgs[i][np.newaxis, :]
            img = torch.tensor(img, dtype=torch.float32, device=self.device)
            # 执行预测
            feature = self.model(img)
            feature = feature.detach().cpu().numpy()
            features.append(feature)
        return features

if __name__ == '__main__':
    predictor = Predictor(args.mtcnn_model_path, args.mobilefacenet_model_path, args.face_db_path,
                          threshold=args.threshold)
    print(len(predictor.faces_db['B哥']))