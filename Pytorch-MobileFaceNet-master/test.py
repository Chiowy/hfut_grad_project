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
add_arg('image_path',               str,     'face_db/於佳威.jpg',                 '预测图片路径')
add_arg('face_db_path',             str,     'dataset/lfw-align-128',                          '人脸库路径')
add_arg('threshold',                float,   0.664017,                                '判断相识度的阈值')
add_arg('mobilefacenet_model_path', str,     'save_model/mobilefacenet.pth',     'MobileFaceNet预测模型的路径')
add_arg('mtcnn_model_path',         str,     'save_model/mtcnn',                 'MTCNN预测模型的路径')
args = parser.parse_args()

if __name__ == '__main__':
    faces_db = {}
    for path in os.listdir(args.face_db_path):
        for img_path in os.listdir(os.path.join(args.face_db_path, path)):
            name = os.path.basename(img_path).split('.')[0]
            image_path = os.path.join(args.face_db_path, path, img_path)
            print(image_path)