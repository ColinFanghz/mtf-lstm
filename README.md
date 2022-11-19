# MTF-LSTM

## 介绍
论文"基于混合示教长短时记忆的高速公路车辆轨迹预测"中数据处理及MTF-LSTM模型实现代码，模型在Pytorch框架下实现。

## 数据处理
本文用到的数据集是NGSIM US101和I-80路段数据
数据集阿里云盘下载地址：https://www.aliyundrive.com/s/M9ukHzrUsjL 提取码：24mv
云盘中所给到的数据集中包含原始数据集和处理后数据集，采用处理后的数据集可直接进入模型训练阶段。

### NGSIM数据处理
NGSIM数据处理流程如图所示：

![image](./img/NGSIM_data.png)

第1步：轨迹数据滤波，将原始US101和I-80的原始数据放入下图文件夹，运行代码"trajectory_denoise.py"，结果如下：

![image](./img/N_step1.png)

第2步：移除不必要特征以及添加新特征，运行代码"preprocess.py"，结果如下：

![image](./img/N_step2.png)

第3步：根据需要添加横、纵向速度和加速度特征，运行代码"add_v_a.py"，结果如下：

![image](./img/N_step3.png)

第4步：按照滑动窗口法提取所需8s轨迹序列，运行代码"final_DP.py"，结果如下：

![image](./img/N_step4.png)

第5步：最终合并US101和I-80数据集，为保证数据的均衡性以及充分利用数据集，随机采样10组数据集，每组按照6:2:2的比例划分训练集、测试集和验证集；运行代码"merge_data.py".


The files (totalData_X.npy, totalData_y.npy) are inputs of models (DNN, LSTM and Transformer). You need change the path for training.


## 模型训练

MTF-LSTM模型训练，运行代码"MTF-LSTM.py"

MTF-LSTM-SP模型训练，运行代码"MTF-LSTM-SP.py"

本文训练好的MTF-LSTM和MTF-LSTM-SP模型保存在文件夹/algorithm/models中，可直接运行来看模型训练效果，鉴于存储空间太大，放入阿里云盘中，通过上述链接可下载。
将models文件夹直接放入algorithm文件夹即可运行。

MTF-LSTM模型：运行"MTF-LSTM-test.py"
MTF-LSTM-SP模型：运行"MTF-LSTM-SP-test.py"

## 后记

有任何代码问题，欢迎联系：fhz_colin@xs.ustb.edu.cn