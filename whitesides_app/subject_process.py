import data_retrieval as DR
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as pl



data_full = DR.db_subjects
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
    return rem

#print(data_full[:, :6])
rem = outlierCheck(data_out,5)
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

# group by views
view_num = []
for n in range(0, len(data_num)):
    for k in range(0, len(subset)):
        if np.array_equal(subset[k],data_num[n, 0:3]) == True:
            view_num += [k+1]
# group by raters
label_key = ['A', 'B', 'C', 'D', 'E','F']
tester_label = []
for n in range(0, len(data_num)):
    for k in range(0, len(raters)):
        if np.array_equal(raters[k],data_full[n, 0]) == True:
            tester_label += [label_key[k]]

# group by category
cat_label = []
for n in range(0, len(data_num)):
    for k in range(0, len(categories)):
        if np.array_equal(categories[k],data_full[n, 1]) == True:
            cat_label += [k+1]

# INTER OBSERVER RELIABILITY
# Interclass Correlation Coefficient Calculation (ICC)
# try group by view & repetition number (i.e. 3 repeats, 5views for each category)
# should no from 1 to 30
repeat_counter = np.zeros(len(subset))
view_rep_group = []
for n in range(0, len(data_num)):
    if tester_label[n] != tester_label[n-1]:    # check if tester has changed
        repeat_counter = np.zeros(len(subset))  # if so reset counter
    for k in range(0, len(subset)): # check which view matches
        if view_num[n]==k+1:
            repeat_counter[k] = repeat_counter[k]+ 1
            view_rep_group += [int((k*6) + repeat_counter[k])]
#create DataFrame
for c in range(5,9):
    d = {'view': view_rep_group,
            'tester': tester_label,
            'deviation': data_num[:, 5].tolist()}
    columns=['view', 'tester', 'deviation']
    df = pd.DataFrame(data=d, columns=columns)
    icc = pg.intraclass_corr(data=df, targets='view', raters='tester', ratings='deviation', nan_policy='omit')
    # icc = pg.intraclass_corr(data=data, targets='Wine', raters='Judge',ratings='Scores').round(3)
    print(icc.set_index('Type'))

# for c in range(5,9):
#     d = {'category': cat_label,
#             'tester': tester_label,
#             'deviation': data_num[:, 5].tolist()}
#     columns=['category', 'tester', 'deviation']
#     df = pd.DataFrame(data=d, columns=columns)
#     icc = pg.intraclass_corr(data=df, targets='category', raters='tester', ratings='deviation', nan_policy='omit')
#     # icc = pg.intraclass_corr(data=data, targets='Wine', raters='Judge',ratings='Scores').round(3)
#     print(icc.set_index('Type'))

# INTEROBSERVER VARIABILITY (IOV)
# variability between users for different views
view_num = np.array(view_num)
IOV_views_mean = []
IOV_views_std = []
for n in range(0,len(subset)):
    IOV_views_mean +=[np.mean(data_num[view_num == n+1, 5:],axis=0)]
    IOV_views_std +=[np.std(data_num[view_num == n+1, 5:],axis=0)]
IOV_views_mean = np.array(IOV_views_mean)
IOV_views_std = np.array(IOV_views_std)



# INTRA OBSERVER VARIABILITY

# I think this below is wrong -- simply a mean & std plot for each view
# plot of means including std bars
view_key = ["View 1", "View 2", "View 3", "View 4", "View 5"]
pl.errorbar(view_key, IOV_views_mean[:,0], yerr=IOV_views_std[:,0], marker="x",capsize=5,elinewidth=0.5,lw=2,color="black" ,label="Whitesides-Epicondylar")
pl.errorbar(view_key, IOV_views_mean[:,1], yerr=IOV_views_std[:,1], marker="x",capsize=5,elinewidth=0.5,lw=2,color="green",label="Epicondylar axis")
pl.errorbar(view_key, IOV_views_mean[:,2], yerr=IOV_views_std[:,2], marker="x",capsize=5,elinewidth=0.5,lw=2,color="red",label="Whiteside line")
pl.errorbar(view_key, IOV_views_mean[:,3], yerr=IOV_views_std[:,3], marker="x",capsize=5,elinewidth=0.5,lw=2,color="blue",label="PCA")
#pl.legend()
pl.ylabel("Mean Deviation Angle (˚)")
pl.show()

# bar plot of std trends
x_axis = np.arange(len(view_key))
pl.xticks(x_axis, view_key)
pl.bar(x_axis-0.3,IOV_views_std[:,0], width=0.2,color="black",label="Whitesides-Epicondylar")
pl.bar(x_axis-0.1,IOV_views_std[:,1], width=0.2,color="green",label="Epicondylar axis")
pl.bar(x_axis+0.1,IOV_views_std[:,2], width=0.2,color="red",label="Whiteside line")
pl.bar(x_axis+0.3,IOV_views_std[:,3], width=0.2,color="blue",label="PCA")
pl.ylabel("Variability in Deviation Angle (˚)")
#pl.legend(bbox_to_anchor =(0.65, 1))
pl.show()


# variability in the repeated measurements made by same user
# iterate through observers:
tester_label=np.array(tester_label)
max_x = 0
complete_abs_diff = []
complete_indi_mean = []
for p in range(0, len(label_key)):  # for each tester
    data_by_tester = data_out[tester_label==label_key[p]]
    view_num_by_tester = view_num[tester_label==label_key[p]]

    # im interested in variability in any repeated measurements
    # don't care which particular deviation is being measured (e.g. epi_dev or white_dev)
    abs_diff = []
    indi_mean = []
    tester_mean = []
    mk_type = ["o", "x", "d", "v", "+", "^"]
    for k in range(0,4):    # for each line deviation measured

        for n in range(0, len(subset)): # for each repeated view
            #print(data_by_tester[view_num_by_tester==n+1, k+2])
            if len(data_by_tester[view_num_by_tester==n+1, k+2]) != 0:
                abs_diff += [data_by_tester[view_num_by_tester==n+1, k+2].max() - data_by_tester[view_num_by_tester==n+1, k+2].min()]
                indi_mean += [(data_by_tester[view_num_by_tester==n+1, k+2].max() + data_by_tester[view_num_by_tester==n+1, k+2].min())/2]
        tester_mean += [np.mean(abs_diff)]
        # print("Tester AbsDiff Mean", np.mean(abs_diff))
        # print("View AbsDiff", np.asarray(abs_diff))

        # I get rel absolute differences of over 100% a lot, not v useful
        # probably due to small amount of deviation anyways
        # rel_abs_diff = np.asarray(abs_diff)/np.asarray(indi_mean)
        # print(rel_abs_diff)
    complete_indi_mean += [np.array(indi_mean)]
    complete_abs_diff += [np.array(abs_diff)]
    label = "Subject "+str(p+1)
    pl.scatter(indi_mean, abs_diff, label=label, marker=mk_type[p], color="black")
complete_indi_mean = np.hstack(complete_indi_mean)
complete_abs_diff = np.hstack(complete_abs_diff)

ovr_mean = np.mean(complete_abs_diff)
print("Mean Difference in Absolute Deviation", ovr_mean)
pl.plot([0, complete_indi_mean.max()], [ovr_mean, ovr_mean],'--', color="red")
pl.xlabel("Average Absolute Deviation (˚)")
pl.ylabel("Difference in Absolute Deviation (˚)")
pl.legend()
pl.show()
