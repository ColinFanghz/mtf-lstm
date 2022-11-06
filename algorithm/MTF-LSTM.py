"""
@Author: Fhz
@Create Date: 2022/11/6 15:37
@File: LSTM_encoder_decoder.py
@Description:
@Modify Person Date:
"""
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def feature_scaling(x_seq):
    x_min = x_seq.min()
    x_max = x_seq.max()
    if x_min == x_max:
        x_new = x_min * np.ones(shape=x_seq.shape)
    else:
        x_new = (2 * x_seq - (x_max + x_min)) / (x_max - x_min)
    return x_new, x_min, x_max


def de_feature_scaling(x_new, x_min, x_max):
    x_ori = np.ones(shape=(len(x_max), 80, 44))
    for i in range(len(x_max)):
        for j in range(3):
            if x_min[i, j] == x_max[i, j]:
                x_ori[i, :, j] = x_min[i, j]
            else:
                x_ori[i, :, j] = (x_new[i, :, j] * (x_max[i, j] - x_min[i, j]) + x_max[i, j] + x_min[i, j]) / 2

    return x_ori


def data_diff(data):
    data_diff = np.diff(data)
    data_0 = data[0]
    return data_0, data_diff


def de_data_diff(data_0, data_diff):
    data = np.ones(shape=(len(data_diff), 80, 44))
    data[:, 0, :] = data_0
    for i in range(79):
        data[:, i + 1, :] = data[:, i, :] + data_diff[:, i, :]

    return data


def dataNormal(seq):
    seq_len = len(seq)
    seq_norm = np.zeros(shape=(seq_len, 79, 44))
    seq_norm_feature = np.zeros(shape=(seq_len, 3, 44))

    for i in range(seq_len):
        for j in range(44):
            seq_tmp = seq[i, :, j]  # initial seq
            seq_tmp_FS, seq_tmp_min, seq_tmp_max = feature_scaling(seq_tmp)  # feature scaling
            seq_tmp_0, seq_tmp_diff = data_diff(seq_tmp_FS)  # seq diff
            seq_norm[i, :, j] = seq_tmp_diff  # store norm data

            # store norm feature data
            seq_norm_feature[i, 0, j] = seq_tmp_min
            seq_norm_feature[i, 1, j] = seq_tmp_max
            seq_norm_feature[i, 2, j] = seq_tmp_0

    return seq_norm, seq_norm_feature


def get_train_dataset(train_data, batch_size):
    x = train_data[:, :29, :]
    y = train_data[:, 29:, :]

    x_data = torch.from_numpy(x.copy())
    y_data = torch.from_numpy(y.copy())

    x_data = x_data.to(torch.float32)
    y_data = y_data.to(torch.float32)

    train_dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_loader


def get_test_dataset(test_data, test_seq_NF, batch_size):
    x_data = torch.from_numpy(test_data.copy())
    x_data = x_data.to(torch.float32)

    y_data = torch.from_numpy(test_seq_NF.copy())
    y_data = y_data.to(torch.float32)

    test_dataset = TensorDataset(x_data, y_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    return test_loader


def LoadData(num):
    train_x = np.load(file="../data_process/NGSIM/merge_data/X_train_{}.npy".format(num))
    train_y = np.load(file="../data_process/NGSIM/merge_data/y_train_{}.npy".format(num))

    test_x = np.load(file="../data_process/NGSIM/merge_data/X_test_{}.npy".format(num))
    test_y = np.load(file="../data_process/NGSIM/merge_data/y_test_{}.npy".format(num))

    valid_x = np.load(file="../data_process/NGSIM/merge_data/X_valid_{}.npy".format(num))
    valid_y = np.load(file="../data_process/NGSIM/merge_data/y_valid_{}.npy".format(num))

    return train_x, train_y, test_x, test_y, valid_x, valid_y


class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4):
        super(lstm_encoder, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(batch_first=True,
                            input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=0.2
                            )

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input)
        return lstm_out, self.hidden


class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4):
        super(lstm_decoder, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(batch_first=True,
                            input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=0.2
                            )
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(input.unsqueeze(1), encoder_hidden_states)
        output = self.fc(lstm_out.squeeze(1))
        return output, self.hidden


class MyLstm(nn.Module):
    def __init__(self, input_size=44, hidden_size=128, target_len=50, TR=0.1):
        super(MyLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target_len = target_len
        self.TR = TR

        self.encoder = lstm_encoder(input_size=self.input_size, hidden_size=self.hidden_size)
        self.decoder = lstm_decoder(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, input, target, training_prediction="recursive"):

        encoder_output, encoder_hidden = self.encoder(input)
        decoder_input = input[:, -1, :]
        decoder_hidden = encoder_hidden

        outputs = torch.zeros(input.shape[0], self.target_len, input.shape[2])

        if training_prediction == "recursive":
            # recursive
            for t in range(self.target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t, :] = decoder_output
                decoder_input = decoder_output

        if training_prediction == "teacher_forcing":
            # teacher_forcing
            for t in range(self.target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t, :] = decoder_output
                decoder_input = target[:, t, :]

        if training_prediction == "mixed_teacher_forcing":
            # mixed_teacher_forcing
            teacher_forcing_ratio = self.TR
            for t in range(self.target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t, :] = decoder_output

                if random.random() < teacher_forcing_ratio:
                    decoder_input = target[:, t, :]
                else:
                    decoder_input = decoder_output

        return outputs


if __name__ == '__main__':

    # 数据集循环，0-9
    for dataset_num in range(10):
        seq_train, y_train, seq_test, y_test, seq_valid, y_valid = LoadData(dataset_num)

        x_norm_train, x_norm_train_feature = dataNormal(seq_train)
        x_norm_test, x_norm_test_feature = dataNormal(seq_test)
        x_norm_valid, x_norm_valid_feature = dataNormal(seq_valid)

        batch_size = 1024
        epochs = 100
        learning_rate = 0.001

        train_loader = get_train_dataset(x_norm_train, batch_size)
        test_loader = get_train_dataset(x_norm_test, batch_size)
        valid_loader = get_test_dataset(x_norm_valid, x_norm_valid_feature, batch_size)

        # 示教学习率循环，0.1-0.9
        for tr in range(9):
            TR = (tr + 1)/10

            for times in range(3):
                model_name = "models_all_data/LSTM_ED10_D{}_R{}_T{}.pkl".format(dataset_num, tr+1, times+1)
                print(model_name)

                model = MyLstm(TR=TR)
                mse_loss = nn.MSELoss(reduction='sum')
                mse_loss1 = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

                device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
                model.to(device)
                print(device)

                loss_train = []
                loss_test = []
                loss_test_history = 1000
                for i in range(epochs):

                    # trian loader
                    loss_tmp = []
                    for batch_idx, (x_seq, y_label) in enumerate(train_loader):
                        x_seq = x_seq.to(device)
                        y_label = y_label.to(device)

                        x_seq = x_seq.to(torch.float32)
                        y_label = y_label.to(torch.float32)

                        pred = model(x_seq, y_label, training_prediction="mixed_teacher_forcing")
                        pred = pred.to(device)
                        loss = mse_loss(y_label[:, :, :2], pred[:, :, :2])

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        loss_tmp.append(loss.item())

                    loss_tmp_np = np.array(loss_tmp)
                    loss_train_mean = loss_tmp_np.mean()
                    print("Epoch:{},Training loss: {}".format(i, loss_train_mean))
                    loss_train.append(loss_train_mean)

                    # test loader
                    loss_tmp = []
                    for batch_idx, (x_seq, y_label) in enumerate(test_loader):
                        x_seq = x_seq.to(device)
                        y_label = y_label.to(device)

                        x_seq = x_seq.to(torch.float32)
                        y_label = y_label.to(torch.float32)

                        with torch.no_grad():
                            pred = model(x_seq, y_label, training_prediction="mixed_teacher_forcing")
                            pred = pred.to(device)
                            loss = mse_loss(y_label[:, :, :2], pred[:, :, :2])

                            loss_tmp.append(loss.item())

                    loss_tmp_np = np.array(loss_tmp)
                    loss_test_mean = loss_tmp_np.mean()
                    print("Epoch:{},Testing loss: {}".format(i, loss_test_mean))
                    loss_test.append(loss_test_mean)

                    if loss_test_history > loss_test_mean:
                        if loss_test_mean / loss_train_mean < 1.05:
                            print("The model is not over-fitting, save it.")
                            torch.save(model.state_dict(), model_name)
                            loss_test_history = loss_test_mean

                    if i == epochs - 1:
                        # valid loader
                        loss_tmp = []
                        loss_1s_tmp = []
                        loss_2s_tmp = []
                        loss_3s_tmp = []
                        loss_4s_tmp = []
                        loss_5s_tmp = []

                        for batch_idx, (x_seq, x_seq_NF) in enumerate(valid_loader):
                            x_data = x_seq.to(device)
                            x_data = x_data.to(torch.float32)

                            x_seq_NF = x_seq_NF.to(device)
                            x_seq_NF = x_seq_NF.to(torch.float32)

                            # muti-steps prediction
                            x_data_ori = x_data.clone()
                            x_tmp = x_data[:, :29, :]
                            y_tmp = x_data[:, 29:, :]
                            print(model_name)

                            model.load_state_dict(torch.load(model_name))

                            with torch.no_grad():
                                pred = model(x_tmp, y_tmp, training_prediction="recursive")

                                pred = pred.to(device)
                                loss = mse_loss(y_tmp[:, :, :2], pred[:, :, :2])

                                loss_tmp.append(loss.item())

                            x_data[:, 29:, :] = pred
                            x_seq_NF = x_seq_NF.cpu().numpy()

                            pred_seq_np = x_data.cpu().numpy()
                            pred_seq_dediff = de_data_diff(x_seq_NF[:, 2, :], pred_seq_np)
                            pred_seq_ori = de_feature_scaling(pred_seq_dediff, x_seq_NF[:, 0, :], x_seq_NF[:, 1, :])

                            x_data_ori_np = x_data_ori.cpu().numpy()
                            x_data_ori_dediff = de_data_diff(x_seq_NF[:, 2, :], x_data_ori_np)
                            x_data_oo = de_feature_scaling(x_data_ori_dediff, x_seq_NF[:, 0, :], x_seq_NF[:, 1, :])

                            pred_seq_ori_torch = torch.from_numpy(pred_seq_ori)
                            x_data_oo_torch = torch.from_numpy(x_data_oo)

                            pred_seq_ori_torch = pred_seq_ori_torch.to(torch.float32)
                            x_data_oo_torch = x_data_oo_torch.to(torch.float32)

                            # Get RMSE_loss of each prediction step
                            loss_1s = mse_loss1(x_data_oo_torch[:, 39, :2], pred_seq_ori_torch[:, 39, :2]) ** 0.5
                            loss_2s = mse_loss1(x_data_oo_torch[:, 49, :2], pred_seq_ori_torch[:, 49, :2]) ** 0.5
                            loss_3s = mse_loss1(x_data_oo_torch[:, 59, :2], pred_seq_ori_torch[:, 59, :2]) ** 0.5
                            loss_4s = mse_loss1(x_data_oo_torch[:, 69, :2], pred_seq_ori_torch[:, 69, :2]) ** 0.5
                            loss_5s = mse_loss1(x_data_oo_torch[:, 79, :2], pred_seq_ori_torch[:, 79, :2]) ** 0.5

                            loss_1s_tmp.append(loss_1s)
                            loss_2s_tmp.append(loss_2s)
                            loss_3s_tmp.append(loss_3s)
                            loss_4s_tmp.append(loss_4s)
                            loss_5s_tmp.append(loss_5s)

                        loss_1s_tmp_np = np.array(loss_1s_tmp)
                        loss_2s_tmp_np = np.array(loss_2s_tmp)
                        loss_3s_tmp_np = np.array(loss_3s_tmp)
                        loss_4s_tmp_np = np.array(loss_4s_tmp)
                        loss_5s_tmp_np = np.array(loss_5s_tmp)

                        loss_1s_mean = loss_1s_tmp_np.mean()
                        loss_2s_mean = loss_2s_tmp_np.mean()
                        loss_3s_mean = loss_3s_tmp_np.mean()
                        loss_4s_mean = loss_4s_tmp_np.mean()
                        loss_5s_mean = loss_5s_tmp_np.mean()

                        loss_tmp_np = np.array(loss_tmp)
                        loss_valid_mean = loss_tmp_np.mean()
                        print("Epoch:{},valid loss: {}".format(i, loss_valid_mean))
