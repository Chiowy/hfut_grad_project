# MobileFaceNet
本项目参考了ArcFace的损失函数结合MobileNet，意在开发一个模型较小，
但识别准确率较高且推理速度快的一种人脸识别项目，
该项目训练数据使用精炼的MS–Celeb–1M数据集，一共有85742个人，共5822653张图片，使用lfw-align-128数据集作为测试数据。

## 数据集准备
本项目提供了标注文件，存放在`dataset`目录下，解压即可。另外需要下载下面这两个数据集，下载完解压到`dataset`目录下。
 - 精炼的MS–Celeb–1M数据集: [百度网盘](https://pan.baidu.com/s/1eXohwNBHbbKXh5KHyItVhQ)
 - lfw-align-128下载地址：[百度网盘](https://pan.baidu.com/s/1tFEX0yjUq3srop378Z1WMA) 提取码：b2ec

然后执行下面命令，将提取人脸图片到`dataset/images`，并把整个数据集打包为二进制文件，这样可以大幅度的提高训练时数据的读取速度。
```shell
python create_dataset.py
```
## 训练
执行`train.py`即可
```shell
python train.py
```
训练参数如下
```
-----------  Configuration Arguments -----------
batch_size: 32
gpus: 0
learning_rate: 0.001
num_epoch: 50
num_workers: 0
resume: save_model/epoch_42
save_model: save_model/
test_list_path: dataset/lfw_test.txt
train_root_path: dataset/train_data
------------------------------------------------
正在加载数据标签...
数据加载完成，总数据量为：1169331, 类别数量为：16336
[2023-05-06 23:51:12.781233] 总数据类别为：16336
```
训练输出如下：
```
成功加载模型参数和优化方法参数
[2023-05-06 23:51:15.895334] Train epoch 43, batch: 0/36542, loss: 1.478138, accuracy: 0.687500, lr: 0.001000, eta: 1 day, 9:44:27
[2023-05-06 23:51:35.587972] Train epoch 43, batch: 100/36542, loss: 1.689338, accuracy: 0.781250, lr: 0.001000, eta: 22:35:48
[2023-05-06 23:51:55.261651] Train epoch 43, batch: 200/36542, loss: 1.351907, accuracy: 0.718750, lr: 0.001000, eta: 22:52:16
[2023-05-06 23:52:15.186016] Train epoch 43, batch: 300/36542, loss: 1.067373, accuracy: 0.781250, lr: 0.001000, eta: 22:05:11
[2023-05-06 23:52:34.881330] Train epoch 43, batch: 400/36542, loss: 0.783306, accuracy: 0.906250, lr: 0.001000, eta: 22:08:45
[2023-05-06 23:52:54.698910] Train epoch 43, batch: 500/36542, loss: 0.961445, accuracy: 0.843750, lr: 0.001000, eta: 22:20:58
[2023-05-06 23:53:14.779787] Train epoch 43, batch: 600/36542, loss: 1.340354, accuracy: 0.687500, lr: 0.001000, eta: 22:03:28
[2023-05-06 23:53:34.617960] Train epoch 43, batch: 700/36542, loss: 1.021514, accuracy: 0.812500, lr: 0.001000, eta: 21:50:14
[2023-05-06 23:53:54.545049] Train epoch 43, batch: 800/36542, loss: 0.873053, accuracy: 0.843750, lr: 0.001000, eta: 22:15:09
[2023-05-06 23:54:14.323455] Train epoch 43, batch: 900/36542, loss: 0.769634, accuracy: 0.906250, lr: 0.001000, eta: 22:14:37
[2023-05-06 23:54:34.512699] Train epoch 43, batch: 1000/36542, loss: 0.756399, accuracy: 0.875000, lr: 0.001000, eta: 22:15:07
[2023-05-06 23:54:54.599053] Train epoch 43, batch: 1100/36542, loss: 1.244270, accuracy: 0.718750, lr: 0.001000, eta: 22:34:07
[2023-05-06 23:55:25.731874] Train epoch 43, batch: 1200/36542, loss: 1.383540, accuracy: 0.781250, lr: 0.001000, eta: 1 day, 3:18:23

```
# 评估
执行`eval.py`即可
```shell
python eval.py
```