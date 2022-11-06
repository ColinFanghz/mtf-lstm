"""
@Author: Fhz
@Create Date: 2022/11/6 20:55
@File: merge_data.py
@Description: 
@Modify Person Date: 
"""
import numpy as np
from sklearn import model_selection


def loadAllData(X_path, Y_path):

    X_Data = np.load(file=X_path)
    y = np.load(file=Y_path)
    X = X_Data[:, ::-1, :]
    print("The length of total data is: {}".format(len(y)))

    x_0 = []
    x_1 = []
    x_2 = []
    y_0 = []
    y_1 = []
    y_2 = []

    for i in range(len(y)):
        y_tmp = y[i]
        if y_tmp == 0:
            x_0.append(X[i])
            y_0.append(y[i])
        elif y_tmp == 1:
            x_1.append(X[i])
            y_1.append(y[i])
        elif y_tmp == 2:
            x_2.append(X[i])
            y_2.append(y[i])

    left_length = len(y_0)
    center_length = len(y_1)
    right_length = len(y_2)

    min_length = min([left_length, center_length, right_length])

    print("The length of left length is: {}".format(left_length))
    print("The length of center length is: {}".format(center_length))
    print("The length of right length is: {}".format(right_length))

    if min_length/left_length == 1:
        x_0_test = x_0
        y_0_test = y_0
    else:
        x_0_train, x_0_test, y_0_train, y_0_test = model_selection.train_test_split(x_0, y_0, test_size=min_length/left_length)

    if min_length/center_length == 1:
        x_1_test = x_1
        y_1_test = y_1
    else:
        x_1_train, x_1_test, y_1_train, y_1_test = model_selection.train_test_split(x_1, y_1, test_size=min_length/center_length)

    if min_length/right_length == 1:
        x_2_test = x_2
        y_2_test = y_2
    else:
        x_2_train, x_2_test, y_2_train, y_2_test = model_selection.train_test_split(x_2, y_2, test_size=min_length/right_length)

    print("The length of left length is: {}".format(len(y_0_test)))
    print("The length of center length is: {}".format(len(y_1_test)))
    print("The length of right length is: {}".format(len(y_2_test)))

    # save x_data
    x_0_data = np.array(x_0_test)
    x_1_data = np.array(x_1_test)
    x_2_data = np.array(x_2_test)

    X_DATA = np.vstack([x_0_data, x_1_data])
    X_DATA = np.vstack([X_DATA, x_2_data])

    # save y_data
    y_0_data = np.array(y_0_test)
    y_1_data = np.array(y_1_test)
    y_2_data = np.array(y_2_test)

    Y_DATA = np.vstack([y_0_data, y_1_data])
    Y_DATA = np.vstack([Y_DATA, y_2_data])

    return X_DATA, Y_DATA


def getEqualNum(x, y, test_size):
    x_0 = []
    x_1 = []
    x_2 = []
    y_0 = []
    y_1 = []
    y_2 = []

    for i in range(len(y)):
        y_tmp = y[i]
        if y_tmp == 0:
            x_0.append(x[i])
            y_0.append(y[i])
        elif y_tmp == 1:
            x_1.append(x[i])
            y_1.append(y[i])
        elif y_tmp == 2:
            x_2.append(x[i])
            y_2.append(y[i])

    print("length of left lane change: {}".format(len(y_0)))
    print("length of lane keep: {}".format(len(y_1)))
    print("length of right lane change: {}".format(len(y_2)))

    x_0 = np.array(x_0)
    y_0 = np.array(y_0)
    x_1 = np.array(x_1)
    y_1 = np.array(y_1)
    x_2 = np.array(x_2)
    y_2 = np.array(y_2)

    x_train_0, x_test_0, y_train_0, y_test_0 = model_selection.train_test_split(x_0, y_0, test_size=test_size)
    x_train_1, x_test_1, y_train_1, y_test_1 = model_selection.train_test_split(x_1, y_1, test_size=test_size)
    x_train_2, x_test_2, y_train_2, y_test_2 = model_selection.train_test_split(x_2, y_2, test_size=test_size)

    x_train = np.vstack((x_train_0, x_train_1))
    x_train = np.vstack((x_train, x_train_2))
    y_train = np.vstack((y_train_0, y_train_1))
    y_train = np.vstack((y_train, y_train_2))
    x_test = np.vstack((x_test_0, x_test_1))
    x_test = np.vstack((x_test, x_test_2))
    y_test = np.vstack((y_test_0, y_test_1))
    y_test = np.vstack((y_test, y_test_2))

    return x_train, x_test, y_train, y_test


def mergeDataset(index, num, X_path, Y_path):
    X_train = "X_train_{}.npy".format(index)
    y_train = "y_train_{}.npy".format(index)
    X_test = "X_test_{}.npy".format(index)
    y_test = "y_test_{}.npy".format(index)
    X_valid = "X_valid_{}.npy".format(index)
    y_valid = "y_valid_{}.npy".format(index)

    X = []
    y = []

    for i in range(num):
        print("Start process data: {}".format(i + 1))

        X_tmp, Y_tmp = loadAllData(X_path[i], Y_path[i])
        if len(y) > 0:
            X = np.vstack([X, X_tmp])
            y = np.vstack([y, Y_tmp])
        else:
            X = X_tmp
            y = Y_tmp

    seq_train_test, seq_valid, y_train_test, y_valid_seq = getEqualNum(X, y, test_size=0.2)
    seq_train, seq_test, y_train_seq, y_test_seq = getEqualNum(seq_train_test, y_train_test, test_size=0.25)

    np.save(file=X_train, arr=seq_train)
    np.save(file=y_train, arr=y_train_seq)
    np.save(file=X_test, arr=seq_test)
    np.save(file=y_test, arr=y_test_seq)
    np.save(file=X_valid, arr=seq_valid)
    np.save(file=y_valid, arr=y_valid_seq)


if __name__ == '__main__':

    num = 6

    X = ["../final_DP/X_data_0400.npy",
         "../final_DP/X_data_0500.npy",
         "../final_DP/X_data_0515.npy",
         "../final_DP/X_data_1317.npy",
         "../final_DP/X_data_1914.npy",
         "../final_DP/X_data_2783.npy"]

    y = ["../final_DP/y_data_0400.npy",
         "../final_DP/y_data_0500.npy",
         "../final_DP/y_data_0515.npy",
         "../final_DP/y_data_1317.npy",
         "../final_DP/y_data_1914.npy",
         "../final_DP/y_data_2783.npy"]

    for i in range(10):
        mergeDataset(i, num, X, y)
