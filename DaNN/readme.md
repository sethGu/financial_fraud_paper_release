# DaNN 域自适应神经网络(Domain Adaptive Neural Networks)

- - -

使用PyTorch的 **Domain Adaptive Neural Networks (DaNN)** 复现。 原版论文在 https://link.springer.com/chapter/10.1007/978-3-319-13560-1_76.

对于领域适应来说，DaNN是一个相当简单的神经网络(只有1个隐藏层)。 但它的思想对神经网络的最大平均偏差(MMD)适应具有重要意义。 从那时起，许多研究都遵循这一想法，将MMD或其他测量(如珊瑚流失、瓦瑟斯坦距离)嵌入更深的网络(如AlexNet、ResNet)。 

对于 **深度迁移学习** 来讲，这是一个必须要学会以及实现的算法

## 数据

采用了 *Office+Caltech10* 数据。 可以从 [这下载](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md#download) 并将其放入新文件夹 `data`，或自行规定文件路径。

- - -

## 使用

采用 Python 3.6 和 PyTorch 0.3.0.

<!-- - `DaNN.py` is the DaNN model
- `mmd.py` is the MMD measurement. You're welcome to change it to others.
- `data_loader.py` is the help function for loading data.
- `main.py` is the main training and test program. You can directly run this file by `python main.py`. -->

- - -

### Reference

Ghifary M, Kleijn W B, Zhang M. Domain adaptive neural networks for object recognition[C]//Pacific Rim International Conference on Artificial Intelligence. Springer, Cham, 2014: 898-904.
