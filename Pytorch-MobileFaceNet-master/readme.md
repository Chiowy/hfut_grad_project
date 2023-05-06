# MobileFaceNet
本项目参考了ArcFace的损失函数结合MobileNet，意在开发一个模型较小，
但识别准确率较高且推理速度快的一种人脸识别项目，
该项目训练数据使用精炼的MS–Celeb–1M数据集，一共有85742个人，共5822653张图片，使用lfw-align-128数据集作为测试数据。
## 项目结构

````
Pytorch-MobileFaceNet-master
├── dataset
│   └── user.json
├── docs
│   └── history.md
├── pyproject.toml
├── src
│   ├── requests
│   │   └── __init__.py
│   └── sample
│       ├── __init__.py
│       ├── user
│       │   └── __init__.py
│       └── views
│           └── __init__.py
├── tests
│   ├── __init__.py
│   ├── user
│   │   └── __init__.py
│   └── views
│       └── __init__.py
└── tox.ini
````
# 