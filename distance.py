import pandas as pd
import numpy as np
from emddist import *
import stayPointDetection_basic as basic
import sklearn
import os


def emddistance(file_a, file_b):
    data_a = pd.read_csv(file_a)
    # print(data_a)
    data_b = pd.read_csv(file_b)
    line_a = np.array(data_a.loc[:, ['laltitude', 'longitude']])
    line_b = np.array(data_b.loc[:, ['laltitude', 'longitude']])
    weights_a = np.ones(line_a.shape[0])
    weights_b = np.ones(line_b.shape[0])
    signature_a = (line_a, weights_a)
    signature_b = (line_b, weights_b)
    emd = getEMD(signature_a, signature_b)
    # print("emd=" + str(emd))
    return emd


# emddistance(r"C:\Users\wisdo\Documents\StayPoint\rome_project\rome_each\a2_1_basic.csv",
#             r"C:\Users\wisdo\Documents\StayPoint\rome_project\rome_each\a2_1_density.csv")
file_dir = r"C:\Users\wisdo\Documents\StayPoint\rome_project\rome_each"
file_name_a = r"11_1_basic.csv"
file_choose = os.path.join(file_dir, file_name_a)
# r"C:\Users\wisdo\Documents\StayPoint\rome_project\rome_each\a2_1_basic.csv"
filelist = {}
for dirname, dirnames, filenames in os.walk(file_dir):
    filenames.remove(file_name_a)
    filenum = len(filenames)
    for i in range(filenum):
        gpsfile_a = os.path.join(dirname, filenames[i])
        filelist[filenames[i].strip("_basic.csv")] = emddistance(file_choose, gpsfile_a)
dist_list=pd.Series(filelist).sort_values()
# dist_list.to_csv(os.path.join(r"C:\Users\wisdo\Documents\StayPoint\rome_project\rome_each_dist",file_name_a))
print(dist_list.iloc[0:5])