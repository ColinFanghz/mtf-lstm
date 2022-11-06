"""
@Author: Fhz
@Create Date: 2022/11/6 20:51
@File: final_DP.py
@Description: 
@Modify Person Date: 
"""
import pandas as pd
import numpy as np
import time


def getTargetVehicle(path):
    dataS = pd.read_csv(path)
    veh_list = dataS.Vehicle_ID.unique()
    veh_left = []
    veh_center = []
    veh_right = []

    for veh_id in veh_list:
        frame_ori = dataS[dataS.Vehicle_ID == veh_id]
        LC_list = frame_ori.Lane_Change_Label.unique()
        for LC_id in LC_list:
            if LC_id == 0:
                veh_left.append(veh_id)
            elif LC_id == 1:
                veh_center.append(veh_id)
            else:
                veh_right.append(veh_id)

    veh_new = list(set(veh_left + veh_right))
    veh_new_np = np.array(veh_new)
    veh_new_np.sort()

    return veh_new_np


class featureExtract():
    def __init__(self, path, veh_id, X_length):
        super(featureExtract, self).__init__()
        self.path = path
        self.veh_id = veh_id
        self.X_length = X_length
        self.dataS = self.getVehicleIDData()

    def getVehicleIDData(self):
        dataS = pd.read_csv(self.path)
        frame_ori = dataS[dataS.Vehicle_ID == self.veh_id]
        GT_list = frame_ori.Global_Time.unique()
        GT_min = np.min(GT_list)
        GT_max = np.max(GT_list)

        frame_time = dataS[dataS.Global_Time >= GT_min]
        frame_time_1 = frame_time[frame_time.Global_Time <= GT_max]

        return frame_time_1

    def getData(self, veh_id):
        '''
        :param veh_id: vehicle ID
        :return: get feature data of veh_id
        '''
        AvailableTime = self.getAvailableTime(veh_id)

        print("*****Getting vehicle {} feature data*****".format(veh_id))
        length_time = len(AvailableTime)

        # All characteristic parameters have 44 dimensions in total
        # Dimension 0-5 target vehicle: (x, y, v_x, v_y, a_x, a_y)
        # Dimension 6-41 are features of surrounding vehicles
        # (left-front left-rear center-front center-rear right-front right-rear)
        # (delta_x, delta_y, v_x, v_y, a_x, a_y) * 6
        # Dimension 42-43 left and right lane flag positions

        X = 1000 * np.ones(shape=(length_time, self.X_length, 44))

        # Driving intention recognition label
        y = 1000 * np.ones(shape=(length_time, 1))

        for i in range(length_time):

            # target vehicle feature writing
            self_condition = self.getCondition(veh_id, AvailableTime[i])
            X[i, :, :6] = self_condition

            # Surrounding vehicles feature writing
            surround_vehicles = self.getOtherVehicles(veh_id, AvailableTime[i])

            # left-front vehicle
            if surround_vehicles[0] == 10000: # if this position doesn't have vehicle
                X[i, :, 6] = 3 * np.ones(shape=(self.X_length))
                X[i, :, 7] = 1000 * np.ones(shape=(self.X_length))
                X[i, :, 8] = self_condition[:, 2]
                X[i, :, 9] = self_condition[:, 3]
                X[i, :, 10] = self_condition[:, 4]
                X[i, :, 11] = self_condition[:, 5]
            else:
                surround_condition = self.getCondition(surround_vehicles[0], AvailableTime[i])
                X[i, :, 6] = surround_condition[:, 0] - self_condition[:, 0]
                X[i, :, 7] = surround_condition[:, 1] - self_condition[:, 1]
                X[i, :, 8] = surround_condition[:, 2]
                X[i, :, 9] = surround_condition[:, 3]
                X[i, :, 10] = surround_condition[:, 4]
                X[i, :, 11] = surround_condition[:, 5]

            # left-rear vehicle
            if surround_vehicles[1] == 10000:
                X[i, :, 12] = 3 * np.ones(shape=(self.X_length))
                X[i, :, 13] = 1000 * np.ones(shape=(self.X_length))
                X[i, :, 14] = self_condition[:, 2]
                X[i, :, 15] = self_condition[:, 3]
                X[i, :, 16] = self_condition[:, 4]
                X[i, :, 17] = self_condition[:, 5]
            else:
                surround_condition = self.getCondition(surround_vehicles[1], AvailableTime[i])
                X[i, :, 12] = surround_condition[:, 0] - self_condition[:, 0]
                X[i, :, 13] = surround_condition[:, 1] - self_condition[:, 1]
                X[i, :, 14] = surround_condition[:, 2]
                X[i, :, 15] = surround_condition[:, 3]
                X[i, :, 16] = surround_condition[:, 4]
                X[i, :, 17] = surround_condition[:, 5]

            # center-front vehicle
            if surround_vehicles[2] == 10000:
                X[i, :, 18] = 0 * np.ones(shape=(self.X_length))
                X[i, :, 19] = 1000 * np.ones(shape=(self.X_length))
                X[i, :, 20] = self_condition[:, 2]
                X[i, :, 21] = self_condition[:, 3]
                X[i, :, 22] = self_condition[:, 4]
                X[i, :, 23] = self_condition[:, 5]
            else:
                surround_condition = self.getCondition(surround_vehicles[2], AvailableTime[i])
                X[i, :, 18] = surround_condition[:, 0] - self_condition[:, 0]
                X[i, :, 19] = surround_condition[:, 1] - self_condition[:, 1]
                X[i, :, 20] = surround_condition[:, 2]
                X[i, :, 21] = surround_condition[:, 3]
                X[i, :, 22] = surround_condition[:, 4]
                X[i, :, 23] = surround_condition[:, 5]

            # center-rear vehicle
            if surround_vehicles[3] == 10000:
                X[i, :, 24] = 0 * np.ones(shape=(self.X_length))
                X[i, :, 25] = 1000 * np.ones(shape=(self.X_length))
                X[i, :, 26] = self_condition[:, 2]
                X[i, :, 27] = self_condition[:, 3]
                X[i, :, 28] = self_condition[:, 4]
                X[i, :, 29] = self_condition[:, 5]
            else:
                surround_condition = self.getCondition(surround_vehicles[3], AvailableTime[i])
                X[i, :, 24] = surround_condition[:, 0] - self_condition[:, 0]
                X[i, :, 25] = surround_condition[:, 1] - self_condition[:, 1]
                X[i, :, 26] = surround_condition[:, 2]
                X[i, :, 27] = surround_condition[:, 3]
                X[i, :, 28] = surround_condition[:, 4]
                X[i, :, 29] = surround_condition[:, 5]

            # right-front vehicle
            if surround_vehicles[4] == 10000:
                X[i, :, 30] = -3 * np.ones(shape=(self.X_length))
                X[i, :, 31] = 1000 * np.ones(shape=(self.X_length))
                X[i, :, 32] = self_condition[:, 2]
                X[i, :, 33] = self_condition[:, 3]
                X[i, :, 34] = self_condition[:, 4]
                X[i, :, 35] = self_condition[:, 5]
            else:
                surround_condition = self.getCondition(surround_vehicles[4], AvailableTime[i])
                X[i, :, 30] = surround_condition[:, 0] - self_condition[:, 0]
                X[i, :, 31] = surround_condition[:, 1] - self_condition[:, 1]
                X[i, :, 32] = surround_condition[:, 2]
                X[i, :, 33] = surround_condition[:, 3]
                X[i, :, 34] = surround_condition[:, 4]
                X[i, :, 35] = surround_condition[:, 5]

            # right-rear vehicle
            if surround_vehicles[5] == 10000:
                X[i, :, 36] = -3 * np.ones(shape=(self.X_length))
                X[i, :, 37] = 1000 * np.ones(shape=(self.X_length))
                X[i, :, 38] = self_condition[:, 2]
                X[i, :, 39] = self_condition[:, 3]
                X[i, :, 40] = self_condition[:, 4]
                X[i, :, 41] = self_condition[:, 5]
            else:
                surround_condition = self.getCondition(surround_vehicles[5], AvailableTime[i])
                X[i, :, 36] = surround_condition[:, 0] - self_condition[:, 0]
                X[i, :, 37] = surround_condition[:, 1] - self_condition[:, 1]
                X[i, :, 38] = surround_condition[:, 2]
                X[i, :, 39] = surround_condition[:, 3]
                X[i, :, 40] = surround_condition[:, 4]
                X[i, :, 41] = surround_condition[:, 5]

            for ii in range(self.X_length):
                frame_t = self.getOneState(veh_id, AvailableTime[i] - ii)
                # Left and right lane flag
                X[i, ii, 42] = int(frame_t.Left_Label)
                X[i, ii, 43] = int(frame_t.Right_Label)

                # 3s trajectory
                if ii == 29:
                    y[i] = int(frame_t.Lane_Change_Label)

        return X, y

    def getAvailableTime(self, veh_id):
        '''
        :param veh_id: vehicle ID
        :return: get available time of data
        '''
        dataS = self.dataS
        frame_ori = dataS[dataS.Vehicle_ID == veh_id]

        Available_time = []

        for i in range(len(frame_ori)):
            t_tmp = float(frame_ori.iloc[i, 1])
            if i >= self.X_length - 1:
                Available_time.append(t_tmp)

        return Available_time

    def getCondition(self, veh_id, t_tmp):
        '''
        :param veh_id: vehicle ID
        :param t_tmp: time stamp
        :return:
        '''
        dataS = self.dataS
        condition = np.zeros(shape=(self.X_length, 6))

        frame_ori = dataS[dataS.Vehicle_ID == veh_id]
        for i in range(self.X_length):
            frame = frame_ori[frame_ori.Global_Time == t_tmp - i]
            if not frame.empty:
                frame_history = frame
            else:
                frame = frame_history

            frame = frame[['Local_X', 'Local_Y', 'v_x', 'v_y', 'a_x', 'a_y']]
            condition[i, :] = frame

        return condition

    def getOtherVehicles(self, veh_id, t_tmp):
        """
        :param veh_id: vehicle ID
        :param t_tmp: time stamp
        :return: Get surrounding vehicle ID of veh_id in t_tmp.
        """
        frame_ori = self.dataS
        frame_vehicle = frame_ori[frame_ori.Vehicle_ID == veh_id]
        frame_self = frame_vehicle[frame_vehicle.Global_Time == t_tmp]

        self_index = frame_self.index.tolist()[0]

        frame_t = frame_ori[frame_ori.Global_Time == t_tmp]
        frame_surround = frame_t.drop(self_index)

        # Method of getting surrounding vehicles
        # step 1: Get all vehicle IDs at the current time
        # step 2: Pass the first round of screening of adjacent lanes (lateral direction)
        # step 3: Filter the second round through the dynamic window (longitudinal direction)

        lane_self = float(frame_self.Lane_ID)
        y_value = float(frame_self.Local_Y)
        x_value = float(frame_self.Local_X)

        # dynamic window value
        distance = 60
        # Delete vehicle IDS outside the dynamic window
        frame_surround = frame_surround[frame_surround['Local_Y'] < y_value + distance]
        frame_surround = frame_surround[frame_surround['Local_Y'] > y_value - distance]

        self_left_label = float(frame_self.Left_Label)
        self_right_label = float(frame_self.Right_Label)

        # Get left lane ID
        if self_left_label:
            lane_left = lane_self - 1
        else:
            lane_left = 100

        # Get right lane ID
        if self_right_label:
            lane_right = lane_self + 1
        else:
            lane_right = 100

        # surround vehicles
        surround_vehicles = 10000 * np.ones(6)

        # left lane
        frame_left = frame_surround[frame_surround['Lane_ID'] == lane_left]
        # right lane
        frame_right = frame_surround[frame_surround['Lane_ID'] == lane_right]
        # center lane
        frame_center = frame_surround[frame_surround['Lane_ID'] == lane_self]

        # get left lane vehicle IDs
        if not frame_left.empty:
            delta_y_pos = 100
            delta_y_neg = -100

            for i in range(len(frame_left)):
                y_tmp = float(frame_left.iloc[i, 3])

                delta_y = y_tmp - y_value

                if delta_y > 0:
                    if delta_y < delta_y_pos:
                        delta_y_pos = delta_y
                        surround_vehicles[0] = frame_left.iloc[i, 0]
                else:
                    if delta_y > delta_y_neg:
                        delta_y_neg = delta_y
                        surround_vehicles[1] = frame_left.iloc[i, 0]

        # get center lane vehicle IDs
        if not frame_center.empty:
            delta_y_pos = 100
            delta_y_neg = -100

            for i in range(len(frame_center)):
                y_tmp = float(frame_center.iloc[i, 3])

                delta_y = y_tmp - y_value

                if delta_y > 0:
                    if delta_y < delta_y_pos:
                        delta_y_pos = delta_y
                        surround_vehicles[2] = frame_center.iloc[i, 0]
                else:
                    if delta_y > delta_y_neg:
                        delta_y_neg = delta_y
                        surround_vehicles[3] = frame_center.iloc[i, 0]

        # get right lane vehicle IDs
        if not frame_right.empty:
            delta_y_pos = 100
            delta_y_neg = -100

            for i in range(len(frame_right)):
                y_tmp = float(frame_right.iloc[i, 3])

                delta_y = y_tmp - y_value

                if delta_y > 0:
                    if delta_y < delta_y_pos:
                        delta_y_pos = delta_y
                        surround_vehicles[4] = frame_right.iloc[i, 0]
                else:
                    if delta_y > delta_y_neg:
                        delta_y_neg = delta_y
                        surround_vehicles[5] = frame_right.iloc[i, 0]

        return surround_vehicles

    def getOneState(self, veh_id, t_tmp):
        """
        :param veh_id: vehicle ID
        :param t_tmp: time stamp
        :return: data of veh_id in t_tmp
        """
        frame_ori = self.dataS
        frame_vehicle = frame_ori[frame_ori.Vehicle_ID == veh_id]
        frame_t = frame_vehicle[frame_vehicle.Global_Time == t_tmp]

        return frame_t


if __name__ == '__main__':

    path_in = ["../add_v_a/trajectory_2783_add.csv",
               "../add_v_a/trajectory_1914_add.csv",
               "../add_v_a/trajectory_1317_add.csv",
               "../add_v_a/trajectory_0400_add.csv",
               "../add_v_a/trajectory_0500_add.csv",
               "../add_v_a/trajectory_0515_add.csv"]

    path_X_out = ["X_data_2783.npy",
                  "X_data_1914.npy",
                  "X_data_1317.npy",
                  "X_data_0400.npy",
                  "X_data_0500.npy",
                  "X_data_0515.npy"]

    path_y_out = ["y_data_2783.npy",
                  "y_data_1914.npy",
                  "y_data_1317.npy",
                  "y_data_0400.npy",
                  "y_data_0500.npy",
                  "y_data_0515.npy"]

    for i in range(len(path_in)):
        veh_list = getTargetVehicle(path_in[i])

        print("*****The length of {} veh_list is:{}*****".format(i, len(veh_list)))

        X_length = 80

        X = []
        y = []
        for veh_id in veh_list:

            print("*****Start process veh_id {}*****".format(veh_id))
            start_time = time.time()
            FE = featureExtract(path_in[i], veh_id, X_length)
            X_tmp, y_tmp = FE.getData(veh_id)
            if len(y) > 0:
                X = np.vstack([X, X_tmp])
                y = np.vstack([y, y_tmp])
            else:
                X = X_tmp
                y = y_tmp

            end_time = time.time()
            print("*****End process veh_id {}*****".format(veh_id))
            print("*****time cost: {}*****".format(end_time-start_time))
            print()

        # Save the processed data into a new file
        np.save(file=path_X_out[i], arr=X)
        np.save(file=path_y_out[i], arr=y)
