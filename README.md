# MTF-LSTM

## 介绍
论文"基于混合示教长短时记忆网络的车辆轨迹预测"中数据处理及MTF-LSTM模型实现代码，模型在Pytorch框架下实现。

## 依赖包
numpy                         1.23.4
torch                         1.10.1
sklearn                       0.0
scikit-learn                  0.24.2


## 数据处理
本文用到的数据集是NGSIM US101和I-80路段数据
数据集百度云盘下载地址：https://pan.baidu.com/s/17j0gR-vVW2chDv0JAZlJZQ 
提取码：xklg
云盘中所给到的数据集中包含原始数据集、处理后数据集和训练好的模型，采用处理后的数据集可直接进入模型训练和模型测试阶段。

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

## 模型训练及测试

MTF-LSTM模型训练，运行代码"MTF-LSTM.py"

MTF-LSTM-SP模型训练，运行代码"MTF-LSTM-SP.py"

本文训练好的MTF-LSTM和MTF-LSTM-SP模型保存在文件夹/algorithm/models中，可直接运行来看模型训练效果，鉴于存储空间太大，放入云盘中，通过上述链接可下载。
将models文件夹直接放入algorithm文件夹即可运行。

MTF-LSTM模型：运行"MTF-LSTM-test.py"
MTF-LSTM-SP模型：运行"MTF-LSTM-SP-test.py"

## 后记

有任何代码问题，欢迎联系：fhz_colin@xs.ustb.edu.cn