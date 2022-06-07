import data_retrieval as DR
import subject_process as sp
import numpy as np
# import pandas as pd
# import pingouin as pg
import matplotlib.pyplot as pl


data_full = DR.db_allattempt
data_num = data_full[:, 2:].astype(float)
data_out = data_num[:, 3:]
# COLUMN CATEGORIES
# 0:  'name'
# 1:  'category'
# 2:  0 'view_rotation_flex'
# 3:  1 'view_rotation_varvalg'
# 4:  2 'view_rotation_intext'
# 5:  3 'user_angle'            # user input of angle between whitesides & epicondylar
# 6:  4 'true_proj_angle'       # real angle between true whitesides & epicondylar projection
# 7:  5 'user_deviation'        # error between user input & real whitesides-epicondylar angle
# 8:  6 'epi_deviation'         # user drawn epicondylar angle error
# 9:  7 'white_deviation'       # user drawn whitesides angle error
# 10: 8 'postcond_deviation'    # user drawn posteriorcondylar angle error

rem = sp.outlierCheck(data_out,5)
if rem.size != 0:
    data_num = np.delete(data_num, rem, 0)    # delete rows containing outliers
    data_out = np.delete(data_out, rem, 0)    # delete rows containing outliers
    data_full = np.delete(data_full, rem, 0)    # delete rows containing outliers
    # NOTE may be possible to only omit certain values in that row

subset = np.unique(data_num[:,0:3], axis=0)
categories = ['TKA', 'PKA']
raters = np.unique(data_full[:,0], axis=0)

print("\nOverall Mean Deviations:", np.mean(data_num[:, 5:], axis=0))
print("Overall StandDev. of Deviations:", np.std(data_num[:, 5:], axis=0))
print("Mean TKA Deviations:", np.mean(data_num[data_full[:,1]==categories[0], 5:], axis=0))
print("StandDev. of TKA Deviations:", np.std(data_num[data_full[:,1]==categories[0], 5:], axis=0))
print("Mean PKA Deviations:", np.mean(data_num[data_full[:,1]==categories[1], 5:], axis=0))
print("StandDev. of PKA Deviations:", np.std(data_num[data_full[:,1]==categories[1], 5:], axis=0))
print("\nInterclass Correlation PMCC between TKA & PKA Deviations")
print("UserDev:",np.corrcoef(data_num[data_full[:,1]==categories[0], 5], data_num[data_full[:,1]==categories[1],5])[0, 1])
print("EpiDev:",np.corrcoef(data_num[data_full[:,1]==categories[0], 6], data_num[data_full[:,1]==categories[1],6])[0, 1])
print("WhiteDev:",np.corrcoef(data_num[data_full[:,1]==categories[0], 7], data_num[data_full[:,1]==categories[1],7])[0, 1])
print("PCADev:",np.corrcoef(data_num[data_full[:,1]==categories[0], 8], data_num[data_full[:,1]==categories[1],8])[0, 1])


# variability in the repeated measurements made by same user
# group by raters
label_key = ['A', 'B']
tester_label = []
for n in range(0, len(data_num)):
    for k in range(0, len(raters)):
        if np.array_equal(raters[k],data_full[n, 0]) == True:
            tester_label += [label_key[k]]
tester_label = np.array(tester_label)


# split by TKA & PKA
data_TKA1 = []
data_TKA2 = []
data_PKA1 = []
data_PKA2 = []
for n in range(0,len(data_full)):
    if data_full[n,1]=="TKA":
        if tester_label[n] == 'A':
            data_TKA1 += [data_num[n, 5:]]
        elif tester_label[n] == 'B':
            data_TKA2 += [data_num[n, 5:]]
    elif data_full[n,1]=="PKA":
        if tester_label[n] == 'A':
            data_PKA1 += [data_num[n, 5:]]
        elif tester_label[n] == 'B':
            data_PKA2 += [data_num[n, 5:]]
data_TKA1 = np.array(data_TKA1)
data_TKA2 = np.array(data_TKA2)
data_PKA1 = np.array(data_PKA1)
data_PKA2 = np.array(data_PKA2)

# take absolute to be consistent with subject intraobserver data
TKA_diff = np.abs(data_TKA1 - data_TKA2)    # row contains difference between 1st and 2nd attempt
PKA_diff = np.abs(data_PKA1 - data_PKA2)    # col indicates which deviation is being compared
TKA_indi_mean = (data_TKA1 + data_TKA2)/2
PKA_indi_mean = (data_PKA1 + data_PKA2)/2
# TKA_ovr_mean = np.mean(TKA_diff)
# PKA_ovr_mean = np.mean(PKA_diff)  # not that different ovr_mean
complete_indi_mean = np.vstack([TKA_indi_mean, PKA_indi_mean])
complete_abs_diff = np.vstack([TKA_diff, PKA_diff])
ovr_mean = np.mean(complete_abs_diff)
pl.scatter(TKA_indi_mean, TKA_diff, label="TKA", marker="x", color="black",alpha=0.5,s=20)
pl.scatter(PKA_indi_mean, PKA_diff, label="PKA", marker="o", color="black",alpha=0.5,s=20)
pl.plot([0, complete_indi_mean.max()], [ovr_mean, ovr_mean],'--', color="red")
pl.xlabel("Average Absolute Deviation (˚)")
pl.ylabel("Difference in Absolute Deviation (˚)")
pl.legend()
pl.show()
