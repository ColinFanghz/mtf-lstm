"""
@Author: Fhz
@Create Date: 2022/11/6 20:47
@File: trajectory_denoise.py
@Description: 
@Modify Person Date: 
"""
import csv
import copy
import pywt
import numpy as np
import pandas as pd


def denoiseData(data):

    data_denoise = copy.deepcopy(data)
    vx = 10 * np.diff(data)
    Wavelet = 'sym11' # filter function

    for i in range(3):
        coffs = pywt.wavedec(vx, Wavelet, level=5)

        ca = coffs[0] # approximation coefficients
        cd1 = coffs[1] # details coefficients
        cd2 = coffs[2]
        cd3 = coffs[3]
        cd4 = coffs[4]
        cd5 = coffs[5]

        cdd1 = 0 * cd1
        cdd2 = 0 * cd2
        cdd3 = 0 * cd3
        cdd4 = 0 * cd4

        coffs_re = []
        coffs_re.append(ca)
        coffs_re.append(cdd1)
        coffs_re.append(cdd2)
        coffs_re.append(cdd3)
        coffs_re.append(cdd4)
        coffs_re.append(cd5)

        vx = pywt.waverec(coffs_re, Wavelet)

    vx = vx[:len(data) - 1]
    for j in range(len(vx)):
        data_denoise[j+1] = data_denoise[j] + vx[j] / 10

    return data_denoise


def reWriteData(path_ori, path_denoise):
    for i in range(len(path_ori)):
        print("******Start process data {} *******".format(i))
        dataS = pd.read_csv(path_ori[i])
        veh_id_max = np.max(dataS.Vehicle_ID.unique())
        print("The max vehicle id is {}".format(veh_id_max))

        f = open(path_denoise[i], 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Vehicle_ID",
                             "Frame_ID",
                             "Total_Frames",
                             "Global_Time",
                             "Local_X",
                             "Local_Y",
                             "Global_X",
                             "Global_Y",
                             "v_Length",
                             "v_Width",
                             "v_Class",
                             "v_Vel",
                             "v_Acc",
                             "Lane_ID",
                             "Preceeding",
                             "Following",
                             "Space_Hdwy",
                             "Time_Hdwy"])

        for j in range(veh_id_max):
            veh_id = j + 1
            frame_ori = dataS[dataS.Vehicle_ID == veh_id]
            if len(frame_ori) == 0:
                print("******vehicle_id {} is empty. *******".format(veh_id))
                continue
            else:
                frame_values = frame_ori.values
                x = frame_values[:, 4]
                x_denoise = denoiseData(x)
                frame_values[:, 4] = x_denoise
                csv_writer.writerows(frame_values)
                print("******vehicle_id {} is written. *******".format(veh_id))

        f.close()
        print("******End process data {} *******".format(i))


def test(path_ori, path_denoise):
    for i in range(len(path_ori)):
        data_ori = pd.read_csv(path_ori[i])
        data_denoise = pd.read_csv(path_denoise[i])

        print("The length of initial data {} is {}".format(i, len(data_ori)))
        print("The length of denoise data {} is {}".format(i, len(data_denoise)))


if __name__ == '__main__':
    path_ori = ["trajectories-0750am-0805am.csv",
                "trajectories-0805am-0820am.csv",
                "trajectories-0820am-0835am.csv",
                "trajectories-0400-0415.csv",
                "trajectories-0500-0515.csv",
                "trajectories-0515-0530.csv"]

    path_denoise = ["trajectories_2783_denoise.csv",
                    "trajectories_1914_denoise.csv",
                    "trajectories_1317_denoise.csv",
                    "trajectories_0400_denoise.csv",
                    "trajectories_0500_denoise.csv",
                    "trajectories_0515_denoise.csv"]

    reWriteData(path_ori, path_denoise)
    # test(path_ori, path_denoise)
