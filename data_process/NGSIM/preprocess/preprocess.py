"""
@Author: Fhz
@Create Date: 2022/11/6 20:48
@File: preprocess.py
@Description: 
@Modify Person Date: 
"""
import pandas as pd
import numpy as np
import math
import csv


class preprocess():
    def __init__(self, path_ori, path_final):
        super(preprocess, self).__init__()
        self.path_ori = path_ori
        self.path_final = path_final
        self.dataRefresh = self.RefreshData()
        self.laneChangeLable = self.getLaneChangeLabel()

    def unitConversion(self, frame):
        '''
        :param df: data with unit feet
        :return: data with unit meter
        '''
        ft_to_m = 0.3048

        frame.loc[:, 'Global_Time'] = frame.loc[:, 'Global_Time'] / 100
        for strs in ["Global_X", "Global_Y", "Local_X", "Local_Y", "v_Length", "v_Width", "v_Vel"]:
            frame.loc[:, strs] = frame.loc[:, strs] * ft_to_m
            frame.loc[:, 'v_Vel'] = frame.loc[:, 'v_Vel'] * ft_to_m * 3.6

        return frame

    def getHeadingAngle(self, s_state, e_state):
        '''
        :param s_state: start state
        :param e_state: end state
        :return: heading Angle
        '''
        headingAngle = math.atan2((e_state[0] - s_state[0]), (e_state[1] - s_state[1]))
        headingAngle = headingAngle * 180 / math.pi

        return headingAngle

    def getLeftLabel(self, lane_id):
        '''
        :param lane_id: lane ID
        :return: Determine whether there is a lane on the left. (0-no, 1-yes)
        '''
        if 1 < lane_id < 7:
            return 1  # Lane_Id : 2-6
        else:
            return 0  # Lane_Id : 1,7,8

    def getRightLabel(self, lane_id):
        '''
        :param lane_id: lane ID
        :return: Determine whether there is a lane on the right. (0-no, 1-yes)
        '''
        if lane_id < 6:
            return 1  # Lane_Id : 1-5
        else:
            return 0  # Lane_Id : 6,7,8

    def RefreshData(self):
        '''
        :return: Remove unwanted dimensions.
                 Add new dimensions (Heading_Angle、Left_Label、Right_Label、Lane_Change_Label)
        '''
        data_new = pd.DataFrame(columns=["Vehicle_ID",
                                         "Global_Time",
                                         "Local_X",
                                         "Local_Y",
                                         "v_Vel",
                                         "v_Acc",
                                         "Lane_ID",
                                         "Heading_Angle",
                                         "Left_Label",
                                         "Right_Label",
                                         "Lane_Change_Label"
                                         ])

        data_tmp = pd.DataFrame(columns=["Vehicle_ID",
                                         "Global_Time",
                                         "Local_X",
                                         "Local_Y",
                                         "v_Vel",
                                         "v_Acc",
                                         "Lane_ID",
                                         "Heading_Angle",
                                         "Left_Label",
                                         "Right_Label",
                                         "Lane_Change_Label"
                                         ])

        dataS = pd.read_csv(self.path_ori)
        max_vehiclenum = np.max(dataS.Vehicle_ID.unique())
        max_vehiclenum = int(max_vehiclenum)
        print(max_vehiclenum)

        for i in range(max_vehiclenum + 1):
            frame_ori = dataS[dataS.Vehicle_ID == i]
            if len(frame_ori) == 0:
                print("The vehicle of ID {} is empty".format(i))
                continue

            frame_ori = self.unitConversion(frame_ori)
            t_first = np.min(frame_ori.Global_Time.unique())
            print("Vehicle ID: {}, length of data: {}".format(i, len(frame_ori)))

            for j in range(len(frame_ori)):
                t_tmp = t_first + j
                frame = frame_ori[frame_ori.Global_Time == t_tmp]
                x_value = float(frame.Local_X)
                y_value = float(frame.Local_Y)

                if j < len(frame_ori) - 1:
                    frame_1 = frame_ori[frame_ori.Global_Time == t_tmp + 1]
                    x_value_1 = float(frame_1.Local_X)
                    y_value_1 = float(frame_1.Local_Y)

                    s_state = [x_value, y_value]
                    e_state = [x_value_1, y_value_1]

                    heading_angle = self.getHeadingAngle(s_state, e_state)

                lane_id = int(frame.Lane_ID)
                left_label = self.getLeftLabel(lane_id)
                right_label = self.getRightLabel(lane_id)

                data_tmp.loc[1] = [frame.iloc[0, 0],
                                     frame.iloc[0, 3],
                                     frame.iloc[0, 4],
                                     frame.iloc[0, 5],
                                     frame.iloc[0, 11],
                                     frame.iloc[0, 12],
                                     frame.iloc[0, 13],
                                     heading_angle,
                                     left_label,
                                     right_label,
                                     1]  # Initial Lane_Change_Label is set as 1.

                data_new = data_new.append(data_tmp, ignore_index=True)

        return data_new

    def getLaneChangeLabel(self):
        '''
        :return: Add lane change label
        '''

        dataS = self.dataRefresh
        max_vehiclenum = np.max(dataS.Vehicle_ID.unique())
        max_vehiclenum = int(max_vehiclenum)
        print(max_vehiclenum)

        # Store label data
        label_storage = []

        for i in range(max_vehiclenum + 1):
            frame_ori = dataS[dataS.Vehicle_ID == i]
            if len(frame_ori) == 0:
                continue

            t_first = np.min(frame_ori.Global_Time.unique())
            print("Vehicle ID: {}, length of data: {}".format(i, len(frame_ori)))

            lane_change_time = []  # lane change time stamp
            t_history = t_first  # history lane change time stamp
            for j in range(len(frame_ori) - 1):
                t_tmp = t_first + j
                frame = frame_ori[frame_ori.Global_Time == t_tmp]
                frame_1 = frame_ori[frame_ori.Global_Time == t_tmp + 1]

                lane_id = float(frame.Lane_ID)
                lane_id_1 = float(frame_1.Lane_ID)
                label_end = 1

                # Store lane change time stamp
                if lane_id > lane_id_1:  # left lane change
                    print("Vehicle ID: {}, time stamp: {}, lane change label: {}".format(i, t_tmp, 0))
                    lane_change_time.append([t_history, t_tmp, 0])
                    t_history = t_tmp
                    label_end = 0
                elif lane_id < lane_id_1:
                    print("Vehicle ID: {}, time stamp: {}, lane change label: {}".format(i, t_tmp, 2))
                    lane_change_time.append([t_history, t_tmp, 2])
                    t_history = t_tmp
                    label_end = 2

            lane_change_time.append([t_history, t_first + len(frame_ori) - 1, label_end])

            if len(lane_change_time) == 1:
                continue
            else:
                ### lane_change_time: First point, index from back to front
                t0, t1, label0 = lane_change_time[0]
                t0 = int(t0)
                t1 = int(t1)

                # Reduce the area within 40 steps
                if t1 - t0 > 40:
                    t0 = t1 - 40

                count_heading = 0
                if label0 == 0:
                    for tmp in range(t1, t0 - 1, -1):
                        frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                        if frame_heading.iloc[0, 7] > -1:  # left heading angle threshold
                            count_heading = count_heading + 1
                            if count_heading >= 3:
                                break
                            else:
                                label_storage.append([i, tmp, label0])
                        else:
                            label_storage.append([i, tmp, label0])
                            count_heading = 0

                elif label0 == 2:
                    for tmp in range(t1 + 1, t0 - 1, -1):
                        frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                        if frame_heading.iloc[0, 7] < 1:  # right heading angle threshold
                            count_heading = count_heading + 1
                            if count_heading >= 3:
                                break
                            else:
                                label_storage.append([i, tmp, label0])
                        else:
                            label_storage.append([i, tmp, label0])
                            count_heading = 0

                ### lane_change_time: middle point
                if len(lane_change_time) > 2:
                    for o in range(1, len(lane_change_time) - 1):
                        t_0_no_use, t_1_no_use, label_0 = lane_change_time[o - 1]
                        t_0, t_1, label_1_no_use = lane_change_time[o]

                        t_0 = int(t_0)
                        t_1 = int(t_1)
                        # Reduce the area within 40 steps
                        # Front half area, indexed from front to back
                        if t_1 - t_0 > 40:
                            t1 = t_0 + 40

                        count_heading = 0
                        if label_0 == 0:
                            for tmp in range(t_0, t1 + 1):
                                frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                                if frame_heading.iloc[0, 7] > -1:
                                    count_heading = count_heading + 1
                                    if count_heading >= 3:
                                        break
                                    else:
                                        label_storage.append([i, tmp, label0])
                                else:
                                    label_storage.append([i, tmp, label0])
                                    count_heading = 0

                        elif label_0 == 2:
                            for tmp in range(t_0, t1 + 1):
                                frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                                if frame_heading.iloc[0, 7] < 1:
                                    count_heading = count_heading + 1
                                    if count_heading >= 3:
                                        break
                                    else:
                                        label_storage.append([i, tmp, label0])
                                else:
                                    label_storage.append([i, tmp, label0])
                                    count_heading = 0

                        # Reduce the area within 40 steps
                        # Second half area, indexed from back to front
                        if t_1 - t_0 > 40:
                            t0 = t_1 - 40

                        count_heading = 0
                        if label_0 == 0:
                            for tmp in range(t_1, t0 - 1, -1):
                                frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                                if frame_heading.iloc[0, 7] > -1:
                                    count_heading = count_heading + 1
                                    if count_heading >= 3:
                                        break
                                    else:
                                        label_storage.append([i, tmp, label0])
                                else:
                                    label_storage.append([i, tmp, label0])
                                    count_heading = 0

                        elif label_0 == 2:
                            for tmp in range(t_1, t0 - 1, -1):
                                frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                                if frame_heading.iloc[0, 7] < 1:
                                    count_heading = count_heading + 1
                                    if count_heading >= 3:
                                        break
                                    else:
                                        label_storage.append([i, tmp, label0])
                                else:
                                    label_storage.append([i, tmp, label0])
                                    count_heading = 0

                ### lane_change_time: Final point
                t_0_no_use, t_1_no_use, label_0 = lane_change_time[len(lane_change_time) - 2]
                t_0, t_1, label_1_no_use = lane_change_time[len(lane_change_time) - 1]

                t_0 = int(t_0)
                t_1 = int(t_1)

                # Reduce the area within 40 steps, Front half area
                if t_1 - t_0 > 40:
                    t1 = t_0 + 40

                count_heading = 0
                if label_0 == 0:
                    for tmp in range(t_0, t1 + 1):
                        frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                        if frame_heading.iloc[0, 7] > -1:
                            count_heading = count_heading + 1
                            if count_heading >= 3:
                                break
                            else:
                                label_storage.append([i, tmp, label0])
                        else:
                            label_storage.append([i, tmp, label0])
                            count_heading = 0

                elif label_0 == 2:
                    for tmp in range(t_0, t1 + 1):
                        frame_heading = frame_ori[frame_ori.Global_Time == tmp]
                        if frame_heading.iloc[0, 7] < 1:
                            count_heading = count_heading + 1
                            if count_heading >= 3:
                                break
                            else:
                                label_storage.append([i, tmp, label0])
                        else:
                            label_storage.append([i, tmp, label0])
                            count_heading = 0

            lane_change_time = []

        # Remove duplicate data
        label_storage_new = []
        for label_tmp in label_storage:
            if label_tmp not in label_storage_new:
                label_storage_new.append(label_tmp)

        data_new = pd.DataFrame(columns=["Vehicle_ID", "Global_Time", "lane_change_label"])
        data_tmp = pd.DataFrame(columns=["Vehicle_ID", "Global_Time", "lane_change_label"])

        for ii in range(len(label_storage_new)):
            data_tmp.loc[1] = label_storage[ii]
            data_new = data_new.append(data_tmp, ignore_index=True)

        return data_new

    def replaceLabel(self):
        '''
        :return: replace "xxx_addLabel.csv" lane change label with "xxx_label.csv".
                 store the result to new file "xxx_Final_label.csv".
        '''

        dataS = self.dataRefresh
        dataS_1 = self.laneChangeLable

        ID_lists = dataS_1.Vehicle_ID.unique()

        f = open(self.path_final, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Vehicle_ID",
                             "Global_Time",
                             "Local_X",
                             "Local_Y",
                             "v_Vel",
                             "v_Acc",
                             "Lane_ID",
                             "Heading_Angle",
                             "Left_Label",
                             "Right_Label",
                             "Lane_Change_Label"])

        for i in range(len(dataS)):
            dataS_tmp = dataS.iloc[i, :]
            veh_id = dataS_tmp.iloc[0]
            if veh_id not in ID_lists:
                csv_writer.writerow(dataS_tmp)
            else:
                time_tmp = dataS_tmp.iloc[1]
                dataS_1_tmp = dataS_1[dataS_1.Vehicle_ID == veh_id]
                dataS_1_tmp1 = dataS_1_tmp[dataS_1_tmp.Global_Time == time_tmp]
                if len(dataS_1_tmp1) == 0:
                    csv_writer.writerow(dataS_tmp)
                else:
                    csv_writer.writerow([dataS_tmp.iloc[0],
                                         dataS_tmp.iloc[1],
                                         dataS_tmp.iloc[2],
                                         dataS_tmp.iloc[3],
                                         dataS_tmp.iloc[4],
                                         dataS_tmp.iloc[5],
                                         dataS_tmp.iloc[6],
                                         dataS_tmp.iloc[7],
                                         dataS_tmp.iloc[8],
                                         dataS_tmp.iloc[9],
                                         dataS_1_tmp1.iloc[0, 2]])

            if i % 10000 == 0:
                print("Written: {}".format(i))

        f.close()


if __name__ == '__main__':

    path_ori = ["../trajectory_denoise/trajectories_2783_denoise.csv",
                "../trajectory_denoise/trajectories_1914_denoise.csv",
                "../trajectory_denoise/trajectories_1317_denoise.csv",
                "../trajectory_denoise/trajectories_0400_denoise.csv",
                "../trajectory_denoise/trajectories_0500_denoise.csv",
                "../trajectory_denoise/trajectories_0515_denoise.csv"]

    path_final = ["trajectory_2783_Final_label.csv",
                  "trajectory_1914_Final_label.csv",
                  "trajectory_1317_Final_label.csv",
                  "trajectory_0400_Final_label.csv",
                  "trajectory_0500_Final_label.csv",
                  "trajectory_0515_Final_label.csv"]

    for i in range(6):
        Pre = preprocess(path_ori[i], path_final[i])
        print("********************* Start Process {} file ********************* ".format(i))
        Pre.replaceLabel()