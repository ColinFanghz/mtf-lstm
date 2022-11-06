"""
@Author: Fhz
@Create Date: 2022/11/6 20:50
@File: add_v_a.py
@Description: 
@Modify Person Date: 
"""
import csv
import numpy as np
import pandas as pd


def addVA(path_final, path_add):
    dataS = pd.read_csv(path_final)
    max_vehicle = np.max(dataS.Vehicle_ID.unique())
    print("The max vehicle ID is: {}".format(max_vehicle))

    f = open(path_add, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Vehicle_ID",
                         "Global_Time",
                         "Local_X",
                         "Local_Y",
                         "v_x",
                         "v_y",
                         "a_x",
                         "a_y",
                         "Lane_ID",
                         "Heading_Angle",
                         "Left_Label",
                         "Right_Label",
                         "Lane_Change_Label"])

    for i in range(int(max_vehicle)):
        frame_ori = dataS[dataS.Vehicle_ID == i + 1]
        if len(frame_ori) == 0:
            print("The vehicle ID {} is empty.".format(i + 1))
            continue
        else:
            print("Process vehicle ID {}.".format(i + 1))
            frame_np = frame_ori.values
            X_np = frame_np[:, 2]
            Y_np = frame_np[:, 3]
            v_x = 10 * np.diff(X_np)
            v_y = 10 * np.diff(Y_np)
            a_x = 10 * np.diff(v_x)
            a_y = 10 * np.diff(v_y)
            for ii in range(len(a_x)):
                csv_writer.writerow([frame_np[ii, 0],
                                     frame_np[ii, 1],
                                     frame_np[ii, 2],
                                     frame_np[ii, 3],
                                     v_x[ii],
                                     v_y[ii],
                                     a_x[ii],
                                     a_y[ii],
                                     frame_np[ii, 6],
                                     frame_np[ii, 7],
                                     frame_np[ii, 8],
                                     frame_np[ii, 9],
                                     frame_np[ii, 10]
                                     ])
    f.close()



if __name__ == '__main__':
    path_final = ["../preprocess/trajectory_2783_Final_label.csv",
                  "../preprocess/trajectory_1914_Final_label.csv",
                  "../preprocess/trajectory_1317_Final_label.csv",
                  "../preprocess/trajectory_0400_Final_label.csv",
                  "../preprocess/trajectory_0500_Final_label.csv",
                  "../preprocess/trajectory_0515_Final_label.csv"]

    path_add = ["trajectory_2783_add.csv",
                "trajectory_1914_add.csv",
                "trajectory_1317_add.csv",
                "trajectory_0400_add.csv",
                "trajectory_0500_add.csv",
                "trajectory_0515_add.csv"]

    for i in range(len(path_final)):
        addVA(path_final[i], path_add[i])
