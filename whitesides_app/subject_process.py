import data_retrieval as DR
import numpy as np

data_full = DR.db_subjects
data_num = data_full[:, 2:].astype(float)
data_out = data_num[:, 3:]
print(data_out[0])
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

def outlierCheck(mat, maxdev):
    mean = np.mean(mat, axis=0)
    std = np.std(mat, axis=0)
    distance_from_mean = mat-mean
    rows2omit = []
    for i in range(0, len(mean)):
        not_outlier = distance_from_mean[:,i] < maxdev * std[i]
        #print(~not_outlier)
        if np.where(~not_outlier)[0].size != 0:
            rows2omit += [np.where(~not_outlier)[0][0]]
    rem = np.unique(np.array(rows2omit))   # return row indices to remove
    mat = np.delete(mat, rem, 0)    # delete rows containing outliers
    # NOTE may be possible to only omit certain values in that row
    return mat

data_out = outlierCheck(data_out,4)
