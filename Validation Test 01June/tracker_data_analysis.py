import csv
from itertools import islice
import numpy as np
import math
import matplotlib.pyplot as pl
import numpy.polynomial.polynomial as poly

def parse_csv(rows_out, name, headslice):
    file = open(name);

    # read & ignore header rows
    csvreader = csv.reader(islice(file, headslice, None))

    layer = np.array([]);
    for row in csvreader:
        # ignore frame col, empty last col
        float_row = np.array([num for num in row[1:-1]])
        layer = np.append(layer, [float_row])
    layer = np.reshape(layer,(rows_out,-1))
    layer[layer == ''] = '0';         # replace empty with 0
    layer = np.array(layer.astype(np.float64));   # convert to float
    file.close();
    return layer;


### !-------------- USING PROBE DATA (6d.csv) ---------------------------------!
#   again, ignore 005

# digprobe_filenames = ['FDL005.csv','FDM006.csv', 'FPL007.csv', 'FPM008.csv'];
# expprobe_filenames = ['StraightOn009.csv', 'FlexUp010.csv', 'FlexDown011.csv', 'Lateral012.csv', 'Medial013.csv'];
digprobe_filenames = ['FDL014.csv','FDM015.csv', 'FPL016.csv', 'FPM017.csv'];
digglasses_filenames = ['GL002.csv','GM003.csv', 'GR004.csv'];
expprobe_filenames = ['StraightOn019.csv', 'FlexUp022.csv', 'FlexDown023.csv', 'Lateral037.csv', 'Medial031.csv',
                        'StraightOn021.csv', 'FlexUp039.csv', 'FlexDown042.csv', 'Medial043.csv'];


# EXTRACT DIGITISATION DATA
headslice = 5;
dig_frame = 100;
data_probe = [];
for item in digprobe_filenames:
    data_probe += [parse_csv(dig_frame, item, headslice)];
data_probe = np.array(data_probe);  # data_probe[file, frame num, column]
# cols = calib q0, qx, qy, qz, x, y, z, error, pin1 q0, qx, qy, qz, x, y, z
# calib x, y, z are the ones we are meant to use for digitisation
data_glasses = []       # extract pin digitisation data
for item in digglasses_filenames:
    data_glasses += [parse_csv(dig_frame, item, headslice)];
data_glasses = np.array(data_glasses);

# average digitisation columns (only calib x, y, z)
dig_probe = [];
for i in range(0, len(digprobe_filenames)):
    dig_probe += [np.average(data_probe[i, :, 4:7], axis=0)];
dig_probe = np.array(dig_probe);    # dig_probe = [file, cols]
# e.g. dig_probe[0,0] = x pos for FL001 digitisation
#      dig_probe[0,1] = y pos for FL001 digitisation, etc.

# average digitisation columns (only calib x, y, z)
dig_glasses_pin = [];
dig_glasses_probe = [];
for i in range(0, len(digglasses_filenames)):
    dig_glasses_probe += [np.average(data_glasses[i, :, 4:7], axis=0)];
    dig_glasses_pin += [np.average(data_glasses[i, :, 8:15], axis=0)];
dig_glasses_probe = np.array(dig_glasses_probe);    # dig_probe = [file, cols]
dig_glasses_pin = np.average(np.array(dig_glasses_pin), axis=0).reshape((1,-1));

### ------------------- BUILD GLASSESS COORDINATE SYSTEM -----------------------
# set origin to front middle
origin_g = dig_glasses_probe[1]
mid = (dig_glasses_probe[0] - dig_glasses_probe[2])/2 + dig_glasses_probe[2]
z_vec_g = (origin_g - mid)/np.linalg.norm(origin_g - mid)
y_vec_g = np.cross((dig_glasses_probe[0] - dig_glasses_probe[2]), z_vec_g)
y_vec_g = y_vec_g/np.linalg.norm(y_vec_g)
x_vec_g = np.cross(z_vec_g, y_vec_g)
x_vec_g = x_vec_g/np.linalg.norm(x_vec_g)
glasses_basis = np.array([x_vec_g, y_vec_g, z_vec_g])

# fig = pl.figure()
# ax = pl.axes(projection='3d')
# ax.scatter(dig_glasses_probe[0,0], dig_glasses_probe[0,1], dig_glasses_probe[0,2])
# ax.scatter(origin_g[0], origin_g[1], origin_g[2])
# ax.scatter(dig_glasses_probe[2,0], dig_glasses_probe[2,1], dig_glasses_probe[2,2])
# ax.scatter(mid[0], mid[1], mid[2])
# ax.quiver(origin_g[0], origin_g[1], origin_g[2], x_vec_g[0]*50, x_vec_g[1]*50, x_vec_g[2]*50, color="red")
# ax.quiver(origin_g[0], origin_g[1], origin_g[2], y_vec_g[0]*50, y_vec_g[1]*50, y_vec_g[2]*50, color="green")
# ax.quiver(origin_g[0], origin_g[1], origin_g[2], z_vec_g[0]*50, z_vec_g[1]*50, z_vec_g[2]*50, color="blue")
# ax.set_ylim((-500,-600))
# pl.show()

# CONVERT PIN ORIENTATION DATA FROM QUATERNIONS TO EULER ROTATIONS
def quaternion2euler(ds):
    RxRyRz = []
    for i in range(0, len(ds)):
        q0 = ds[i,0]
        qx = ds[i,1]
        qy = ds[i,2]
        qz = ds[i,3]
        Rx=math.atan2((2*(q0*qx+qy*qz)),(1-2*(qx**2+qy**2)))*180/np.pi;
        Ry=math.asin(2*(q0*qy-qz*qx)) * 180/np.pi;
        Rz=math.atan2((2*(q0*qz+qx*qy)),(1-2*(qy**2+qz**2)))*180/np.pi;
        RxRyRz += [Rx,Ry,Rz];
    return np.array(RxRyRz).reshape((-1,3))
dig_pin_euler = quaternion2euler(dig_glasses_pin)[0]
dig_pin_translation = origin_g - dig_glasses_pin[0,4:]

# ---------------------- EXTRACT EXPERIMENT DATA ------------------------------
exp1_probe = parse_csv(1000, expprobe_filenames[0], headslice)
exp2_probe = parse_csv(1000, expprobe_filenames[1], headslice)#[:800]
exp3_probe = parse_csv(1000, expprobe_filenames[2], headslice)
exp4_probe = parse_csv(1000, expprobe_filenames[3], headslice)
exp5_probe = parse_csv(1000, expprobe_filenames[4], headslice)
exp6_probe = parse_csv(1000, expprobe_filenames[5], headslice)
exp7_probe = parse_csv(1000, expprobe_filenames[6], headslice)
exp8_probe = parse_csv(1000, expprobe_filenames[7], headslice)
exp9_probe = parse_csv(1000, expprobe_filenames[8], headslice)

# need to slice some data to cut off unecessary readings

# pin1 indicates the tracker pin we use (in this case for GLASSES)
# therefore we use pin1 x, y, z values (n.b. these are relative to tracker)
# so we need to convert to find relative to global coordinate system from digitisation
# N.B. take avg of pin position/orientation, reshape so 1 row in matrix
# start numbering from 1
pin1_1 = np.average(exp1_probe[:, 12:15], axis=0).reshape((1,-1))
pin1_1_euler = np.average(quaternion2euler(exp1_probe[:, 8:12]), axis=0).reshape((1,-1))
probe_1 = exp1_probe[:, 4:7]

pin1_2 = np.average(exp2_probe[:, 12:15], axis=0).reshape((1,-1))
pin1_2_euler = np.average(quaternion2euler(exp2_probe[:, 8:12]), axis=0).reshape((1,-1))
probe_2 = exp2_probe[:, 4:7]

pin1_3 = np.average(exp3_probe[:, 12:15], axis=0).reshape((1,-1))
pin1_3_euler = np.average(quaternion2euler(exp3_probe[:, 8:12]), axis=0).reshape((1,-1))
probe_3 = exp3_probe[:, 4:7]

pin1_4 = np.average(exp4_probe[:, 12:15], axis=0).reshape((1,-1))
pin1_4_euler = np.average(quaternion2euler(exp4_probe[:, 8:12]), axis=0).reshape((1,-1))
probe_4 = exp4_probe[:, 4:7]

pin1_5 = np.average(exp5_probe[:, 12:15], axis=0).reshape((1,-1))
pin1_5_euler = np.average(quaternion2euler(exp5_probe[:, 8:12]), axis=0).reshape((1,-1))
probe_5 = exp5_probe[:, 4:7]

pin1_6 = np.average(exp6_probe[:, 12:15], axis=0).reshape((1,-1))
pin1_6_euler = np.average(quaternion2euler(exp6_probe[:, 8:12]), axis=0).reshape((1,-1))
probe_6 = exp6_probe[:, 4:7]

pin1_7 = np.average(exp7_probe[:, 12:15], axis=0).reshape((1,-1))
pin1_7_euler = np.average(quaternion2euler(exp7_probe[:, 8:12]), axis=0).reshape((1,-1))
probe_7 = exp7_probe[:, 4:7]

pin1_8 = np.average(exp8_probe[:, 12:15], axis=0).reshape((1,-1))
pin1_8_euler = np.average(quaternion2euler(exp8_probe[:, 8:12]), axis=0).reshape((1,-1))
probe_8 = exp8_probe[:, 4:7]

pin1_9 = np.average(exp9_probe[:, 12:15], axis=0).reshape((1,-1))
pin1_9_euler = np.average(quaternion2euler(exp9_probe[:, 8:12]), axis=0).reshape((1,-1))
probe_9 = exp9_probe[:, 4:7]

pin1_straight = (pin1_1 + pin1_6)/2
pin1_straight_euler = (pin1_1_euler + pin1_6_euler)/2

pin1_flexup = (pin1_2 + pin1_7)/2
pin1_flexup_euler = (pin1_2_euler + pin1_7_euler)/2

pin1_flexdown = (pin1_3 + pin1_8)/2
pin1_flexdown_euler = (pin1_3_euler + pin1_8_euler)/2

pin1_lateral = pin1_4   #(pin1_2 + pin1_7)/2
pin1_lateral_euler = pin1_4_euler   #(pin1_2_euler + pin1_7_euler)/2

pin1_medial = (pin1_5 + pin1_9)/2
pin1_medial_euler = (pin1_5_euler + pin1_9_euler)/2

def coordTransform(angle_val):
  alpha = angle_val[0]
  beta = angle_val[1]
  gamma = angle_val[2]
  Rx = np.array([[1, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)]])
  Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]])
  Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma), np.cos(gamma), 0],
                [0, 0, 1]])
  R = Rx.dot(Ry).dot(Rz)
  return R

# Rotate & translate to find glasses middle position from pin
# Find required rotation for viewing position/orientation
euler_rot_straight = pin1_straight_euler - dig_pin_euler
R_straight = coordTransform(euler_rot_straight[0]*np.pi/180)
pin_TransRot_straight = np.matmul(R_straight, dig_pin_translation)    # rotate trans vector
pin1_straight = pin1_straight + pin_TransRot_straight  # add trans vector to find glasses front position
# repeat for other experiments
euler_rot_flexup = pin1_flexup_euler - dig_pin_euler
R_flexup = coordTransform(euler_rot_flexup[0]*np.pi/180)
pin_TransRot_flexup = np.matmul(R_flexup, dig_pin_translation)
pin1_flexup = pin1_flexup + pin_TransRot_flexup

euler_rot_flexdown = pin1_flexdown_euler - dig_pin_euler
R_flexdown = coordTransform(euler_rot_flexdown[0]*np.pi/180)
pin_TransRot_flexdown = np.matmul(R_flexdown, dig_pin_translation)
pin1_flexdown = pin1_flexdown + pin_TransRot_flexdown

euler_rot_lateral = pin1_lateral_euler - dig_pin_euler
R_lateral = coordTransform(euler_rot_lateral[0]*np.pi/180)
pin_TransRot_lateral = np.matmul(R_lateral, dig_pin_translation)
pin1_lateral = pin1_lateral + pin_TransRot_lateral

euler_rot_medial = pin1_medial_euler - dig_pin_euler
R_medial = coordTransform(euler_rot_medial[0]*np.pi/180)
pin_TransRot_medial = np.matmul(R_medial, dig_pin_translation)
pin1_medial = pin1_medial + pin_TransRot_medial

def viewRotate(R, basis):
    out = np.array([])
    for i in range(0,len(basis)):
        out = np.append(out, np.matmul(R, basis[i]).reshape((1,3)))
    return out.reshape((len(basis), -1))
# find new glasses basis viewing orientation still in global system
# n.b. we only focus on z axis (i.e. normal of viewing plane)
glasses_basis_straight = viewRotate(R_straight, glasses_basis)
glasses_basis_flexup = viewRotate(R_flexup, glasses_basis)
glasses_basis_flexdown = viewRotate(R_flexdown, glasses_basis)
glasses_basis_lateral = viewRotate(R_lateral, glasses_basis)
glasses_basis_medial = viewRotate(R_medial, glasses_basis)

# fig = pl.figure()
# ax = pl.axes(projection='3d')
# ax.quiver(0, 0, 0, x_vec_g[0], x_vec_g[1], x_vec_g[2], color="red")
# ax.quiver(0, 0, 0, y_vec_g[0], y_vec_g[1], y_vec_g[2], color="green")
# ax.quiver(0, 0, 0, z_vec_g[0], z_vec_g[1], z_vec_g[2], color="blue")
# ax.quiver(0, 0, 0, glasses_basis_straight[2,0], glasses_basis_straight[2,1], glasses_basis_straight[2,2], color="purple")
# ax.set_xlim(-1,1)
# ax.set_ylim(1,-1)
# ax.set_zlim(-1,1)
# pl.show()

### --------------------BUILD PROXY COORDINATE SYSTEM--------------------------
vec1 = dig_probe[1] - dig_probe[0]; # EPICONDYLAR MED - LAT
origin = dig_probe[0] + vec1/2      # set origin as midpoint between epicondylar
x_vec = vec1/np.linalg.norm(vec1)    #set x_vec to vec1

# set refpoint2 to midpoint between proximal digitisation points
refpoint2 = dig_probe[2] + (dig_probe[3] - dig_probe[2])/2
vec2 = refpoint2-origin; # PROXIMAL LAT - DISTAL LAT

z_vec = vec2/np.linalg.norm(vec2)
y_vec = np.cross(z_vec, x_vec)
y_vec = y_vec/np.linalg.norm(y_vec) # unit

# Proxy Coordinate system basis vectors
proxy_basis = np.array([x_vec, y_vec, z_vec])

# find relative vector from proxy origin to pin1
pin_rel_vec_straight = pin1_straight - origin;
pin_rel_vec_flexup = pin1_flexup - origin;
pin_rel_vec_flexdown = pin1_flexdown - origin;
pin_rel_vec_lateral = pin1_lateral - origin;
pin_rel_vec_medial = pin1_medial - origin;

probe_rel_vec_1 = probe_1 - origin;
probe_rel_vec_2 = probe_2 - origin;
probe_rel_vec_3 = probe_3 - origin;
probe_rel_vec_4 = probe_4 - origin;
probe_rel_vec_5 = probe_5 - origin;
probe_rel_vec_6 = probe_6 - origin;
probe_rel_vec_7 = probe_7 - origin;
probe_rel_vec_8 = probe_8 - origin;
probe_rel_vec_9 = probe_9 - origin;


# n.b. still in terms of tracker global coordinate system
# Transform into proxy coordinate system
def vec_in_new_coord_system(vec, basis):
    vec_rel_proxy = []
    for i in range(0, len(vec)):
        newproxy = np.dot(basis, vec[i])
        vec_rel_proxy += [newproxy]
    return np.array(vec_rel_proxy).reshape((-1, 3))


pin_proxy_straight= vec_in_new_coord_system(pin_rel_vec_straight, proxy_basis)
pin_proxy_flexup = vec_in_new_coord_system(pin_rel_vec_flexup, proxy_basis)
pin_proxy_flexdown = vec_in_new_coord_system(pin_rel_vec_flexdown, proxy_basis)
pin_proxy_lateral = vec_in_new_coord_system(pin_rel_vec_lateral, proxy_basis)
pin_proxy_medial = vec_in_new_coord_system(pin_rel_vec_medial, proxy_basis)
probe_proxy_1 = vec_in_new_coord_system(probe_rel_vec_1, proxy_basis)
probe_proxy_2 = vec_in_new_coord_system(probe_rel_vec_2, proxy_basis)
probe_proxy_3 = vec_in_new_coord_system(probe_rel_vec_3, proxy_basis)
probe_proxy_4 = vec_in_new_coord_system(probe_rel_vec_4, proxy_basis)
probe_proxy_5 = vec_in_new_coord_system(probe_rel_vec_5, proxy_basis)
probe_proxy_6 = vec_in_new_coord_system(probe_rel_vec_6, proxy_basis)
probe_proxy_7 = vec_in_new_coord_system(probe_rel_vec_7, proxy_basis)
probe_proxy_8 = vec_in_new_coord_system(probe_rel_vec_8, proxy_basis)
probe_proxy_9 = vec_in_new_coord_system(probe_rel_vec_9, proxy_basis)

# convert glass basis into proxy coordinate system
glasses_basis_proxy_1 = vec_in_new_coord_system(glasses_basis_straight, proxy_basis)
glasses_basis_proxy_2 = vec_in_new_coord_system(glasses_basis_flexup, proxy_basis)
glasses_basis_proxy_3 = vec_in_new_coord_system(glasses_basis_flexdown, proxy_basis)
glasses_basis_proxy_4 = vec_in_new_coord_system(glasses_basis_lateral, proxy_basis)
glasses_basis_proxy_5 = vec_in_new_coord_system(glasses_basis_medial, proxy_basis)


### ------------------- RELATE PROXY TO HIP COORD SYSTEM --------------------
hip2proxy = np.array([[-54.61, 24.79, -433.9]]) # n.b. in proxy system coords
y_rot = -np.arctan(54.66/433.72)    # clockwise rotation about y axis
x_rot = -np.arctan(13.88/433.72)    # clockwise rotation about x axis
trans_angle = np.array([x_rot, y_rot, 0])

# keep relative to proxy coordinate system
original_basis = np.array([[1,0,0],[0,1,0],[0,0,1]])
# rotate into hip basis vector (relative to proxy coord systems)
hip_basis = np.matmul(coordTransform(trans_angle), original_basis)
# normalise hip basis vectors
hip_basis[0] = hip_basis[0]/np.linalg.norm(hip_basis[0])
hip_basis[1] = hip_basis[1]/np.linalg.norm(hip_basis[1])
hip_basis[2] = hip_basis[2]/np.linalg.norm(hip_basis[2])
# N.B. hip basis vectors are in proxy coord system values

# SANITY CHECK
# test = np.array([[54.66, -13.88, 433.72]])
# print(vec_in_new_coord_system(test, hip_basis))    # should go to only z component

# FIRST: find vector from hip origin to tracker points in proxy coords
# SECOND: convert tracker vector to hip coordinate system
# THird: convert glasses basis from proxy system to hip system
pin_hip_straight = vec_in_new_coord_system(pin_proxy_straight + hip2proxy, hip_basis)
pin_hip_flexup = vec_in_new_coord_system(pin_proxy_flexup + hip2proxy, hip_basis)
pin_hip_flexdown = vec_in_new_coord_system(pin_proxy_flexdown + hip2proxy, hip_basis)
pin_hip_lateral = vec_in_new_coord_system(pin_proxy_lateral + hip2proxy, hip_basis)
pin_hip_medial = vec_in_new_coord_system(pin_proxy_medial + hip2proxy, hip_basis)
glasses_basis_hip_1 = vec_in_new_coord_system(glasses_basis_proxy_1, hip_basis)
glasses_basis_hip_2 = vec_in_new_coord_system(glasses_basis_proxy_2, hip_basis)
glasses_basis_hip_3 = vec_in_new_coord_system(glasses_basis_proxy_3, hip_basis)
glasses_basis_hip_4 = vec_in_new_coord_system(glasses_basis_proxy_4, hip_basis)
glasses_basis_hip_5 = vec_in_new_coord_system(glasses_basis_proxy_5, hip_basis)
#print(glasses_basis_hip_1)
probe_hip_1 = vec_in_new_coord_system(probe_proxy_1 + hip2proxy, hip_basis)
probe_hip_2 = vec_in_new_coord_system(probe_proxy_2 + hip2proxy, hip_basis)
probe_hip_3 = vec_in_new_coord_system(probe_proxy_3 + hip2proxy, hip_basis)
probe_hip_4 = vec_in_new_coord_system(probe_proxy_4 + hip2proxy, hip_basis)
probe_hip_5 = vec_in_new_coord_system(probe_proxy_5 + hip2proxy, hip_basis)
probe_hip_6 = vec_in_new_coord_system(probe_proxy_6 + hip2proxy, hip_basis)
probe_hip_7 = vec_in_new_coord_system(probe_proxy_7 + hip2proxy, hip_basis)
probe_hip_8 = vec_in_new_coord_system(probe_proxy_8 + hip2proxy, hip_basis)
probe_hip_9 = vec_in_new_coord_system(probe_proxy_9 + hip2proxy, hip_basis)


# PLOT DATA relative to hip origin
# fig = pl.figure()
# ax = pl.axes(projection='3d')
# ax.scatter(pin_hip_straight[:,0], pin_hip_straight[:,1], pin_hip_straight[:,2], label="Pin position 1", c='blue')
# ax.quiver(pin_hip_straight[:,0], pin_hip_straight[:,1], pin_hip_straight[:,2], glasses_basis_hip_1[2,0]*50, glasses_basis_hip_1[2,1]*50, glasses_basis_hip_1[2,2]*50,color="blue")
# ax.scatter(pin_hip_flexup[:,0], pin_hip_flexup[:,1], pin_hip_flexup[:,2], label="Pin position 2", c='orange')
# ax.quiver(pin_hip_flexup[:,0], pin_hip_flexup[:,1], pin_hip_flexup[:,2], glasses_basis_hip_2[2,0]*50, glasses_basis_hip_2[2,1]*50, glasses_basis_hip_2[2,2]*50,color="orange")
# ax.scatter(pin_hip_flexdown[:,0], pin_hip_flexdown[:,1], pin_hip_flexdown[:,2], label="Pin position 3")
# ax.quiver(pin_hip_flexdown[:,0], pin_hip_flexdown[:,1], pin_hip_flexdown[:,2], glasses_basis_hip_3[2,0]*50, glasses_basis_hip_3[2,1]*50, glasses_basis_hip_3[2,2]*50)
# ax.scatter(pin_hip_lateral[:,0], pin_hip_lateral[:,1], pin_hip_lateral[:,2], label="Pin position 4")
# ax.quiver(pin_hip_lateral[:,0], pin_hip_lateral[:,1], pin_hip_lateral[:,2], glasses_basis_hip_4[2,0]*50, glasses_basis_hip_4[2,1]*50, glasses_basis_hip_4[2,2]*50)
# ax.scatter(pin_hip_medial[:,0], pin_hip_medial[:,1], pin_hip_medial[:,2], label="Pin position 5")
# ax.quiver(pin_hip_medial[:,0], pin_hip_medial[:,1], pin_hip_medial[:,2], glasses_basis_hip_5[2,0]*50, glasses_basis_hip_5[2,1]*50, glasses_basis_hip_5[2,2]*50)
# ax.scatter(probe_hip_1[:,0], probe_hip_1[:,1], probe_hip_1[:,2], label="Whitesides 1", c='purple')
# ax.scatter(probe_hip_2[:,0], probe_hip_2[:,1], probe_hip_2[:,2], label="Whitesides 2", c='red')
# ax.scatter(probe_hip_3[:,0], probe_hip_3[:,1], probe_hip_3[:,2], label="Whitesides 3")
# ax.scatter(probe_hip_4[:,0], probe_hip_4[:,1], probe_hip_4[:,2], label="Whitesides 4")
# ax.scatter(probe_hip_5[:,0], probe_hip_5[:,1], probe_hip_5[:,2], label="Whitesides 5")
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.legend()
# pl.show()
#
# pin_axes = []
# fig = pl.figure()
# ax = pl.axes(projection='3d')
# #ax.quiver(0, 0, 0,1, 1,1)
# ax.quiver(0, 0, 0, glasses_basis_hip_2[0, 0], glasses_basis_hip_2[0,1], glasses_basis_hip_2[0,2],color="red")
# ax.quiver(0, 0, 0, glasses_basis_hip_2[1, 0], glasses_basis_hip_2[1,1], glasses_basis_hip_2[1,2],color="green")
# ax.quiver(0, 0, 0, glasses_basis_hip_2[2, 0], glasses_basis_hip_2[2,1], glasses_basis_hip_2[2,2],color="blue")
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# pl.show()



## SIMPLE ORIENTATION FIX -- assume eyes are always positioned toward distal femur
hip2epimid_inhip = np.array([[0,0,-437.37]])    # length of mechanical axis (i.e. z-coordinate of midpoint of epicondylar axis)
normal_straight = (hip2epimid_inhip - pin_hip_straight)/np.linalg.norm(hip2epimid_inhip - pin_hip_straight)
normal_flexup = (hip2epimid_inhip - pin_hip_flexup)/np.linalg.norm(hip2epimid_inhip - pin_hip_flexup)
normal_flexdown = (hip2epimid_inhip - pin_hip_flexdown)/np.linalg.norm(hip2epimid_inhip - pin_hip_flexdown)
normal_lateral = (hip2epimid_inhip - pin_hip_lateral)/np.linalg.norm(hip2epimid_inhip - pin_hip_lateral)
normal_medial = (hip2epimid_inhip - pin_hip_medial)/np.linalg.norm(hip2epimid_inhip - pin_hip_medial)

# Find real view orientation (no rotation ONLY flexion & varus valgus)
def realViewOrient(vec):
    varvalg = np.arctan(vec[0,0]/np.abs(vec[0,2]))*180/np.pi    # +ve --> medial for left femur
    flexext = np.arctan(vec[0,1]/np.abs(vec[0,2]))*180/np.pi    # +ve --> flex up
    return np.array([varvalg, flexext])   # output in degrees

# N.B. use vector from distal femur centre to glasses (so swap directions)
RVO_straight = realViewOrient(-hip2epimid_inhip + pin_hip_straight)
print("Real Straight View orientation:", RVO_straight)
print("Real FlexUp View orientation:", realViewOrient(-hip2epimid_inhip + pin_hip_flexup))
print("Real FlexDown View orientation:", realViewOrient(-hip2epimid_inhip + pin_hip_flexdown))
print("Real Lateral View orientation:", realViewOrient(-hip2epimid_inhip + pin_hip_lateral))
print("Real Medial View orientation:", realViewOrient(-hip2epimid_inhip + pin_hip_medial))
print("\nRELATIVE TO STRAIGHT VIEW")
print("Real Straight View orientation:", RVO_straight-RVO_straight)
print("Real FlexUp View orientation:", realViewOrient(-hip2epimid_inhip + pin_hip_flexup)-RVO_straight)
print("Real FlexDown View orientation:", realViewOrient(-hip2epimid_inhip + pin_hip_flexdown)-RVO_straight)
print("Real Lateral View orientation:", realViewOrient(-hip2epimid_inhip + pin_hip_lateral)-RVO_straight)
print("Real Medial View orientation:", realViewOrient(-hip2epimid_inhip + pin_hip_medial)-RVO_straight)

# fig = pl.figure()
# ax = pl.axes(projection='3d')
# ax.scatter(pin_hip_straight[:,0], pin_hip_straight[:,1], pin_hip_straight[:,2], label="Straight On avg position", c='blue')
# ax.scatter(pin_hip_flexup[:,0], pin_hip_flexup[:,1], pin_hip_flexup[:,2], label="FlexUp avg position", c='orange')
# ax.scatter(pin_hip_flexdown[:,0], pin_hip_flexdown[:,1], pin_hip_flexdown[:,2], label="FlexDown avg position")
# ax.scatter(pin_hip_lateral[:,0], pin_hip_lateral[:,1], pin_hip_lateral[:,2], label="Lateral avg position")
# ax.scatter(pin_hip_medial[:,0], pin_hip_medial[:,1], pin_hip_medial[:,2], label="Medial avg position")
# # viewing plane normal vectors quiver plot
# #ax.quiver(pin_hip_straight[:,0], pin_hip_straight[:,1], pin_hip_straight[:,2], normal_straight[0,0]*50, normal_straight[0,1]*50, normal_straight[0,2]*50,color="blue")
# #ax.quiver(pin_hip_flexup[:,0], pin_hip_flexup[:,1], pin_hip_flexup[:,2], normal_2[0,0]*50, normal_2[0,1]*50, normal_2[0,2]*50,color="orange")
# #ax.quiver(pin_hip_flexdown[:,0], pin_hip_flexdown[:,1], pin_hip_flexdown[:,2], normal_3[0,0]*50, normal_3[0,1]*50, normal_3[0,2]*50)
# #ax.quiver(pin_hip_lateral[:,0], pin_hip_lateral[:,1], pin_hip_lateral[:,2], normal_4[0,0]*50, normal_4[0,1]*50, normal_4[0,2]*50)
# #ax.quiver(pin_hip_medial[:,0], pin_hip_medial[:,1], pin_hip_medial[:,2], normal_5[0,0]*50, normal_5[0,1]*50, normal_5[0,2]*50)
# ax.scatter(probe_hip_1[:,0], probe_hip_1[:,1], probe_hip_1[:,2], label="Whitesides 1", c='purple')
# ax.scatter(probe_hip_2[:,0], probe_hip_2[:,1], probe_hip_2[:,2], label="Whitesides 2", c='red')
# ax.scatter(probe_hip_3[:,0], probe_hip_3[:,1], probe_hip_3[:,2], label="Whitesides 3")
# ax.scatter(probe_hip_4[:,0], probe_hip_4[:,1], probe_hip_4[:,2], label="Whitesides 4")
# ax.scatter(probe_hip_5[:,0], probe_hip_5[:,1], probe_hip_5[:,2], label="Whitesides 5")
# ax.scatter(probe_hip_6[:,0], probe_hip_6[:,1], probe_hip_6[:,2], label="Whitesides 6")
# ax.scatter(probe_hip_7[:,0], probe_hip_7[:,1], probe_hip_7[:,2], label="Whitesides 7")
# ax.scatter(probe_hip_8[:,0], probe_hip_8[:,1], probe_hip_8[:,2], label="Whitesides 8")
# ax.scatter(probe_hip_9[:,0], probe_hip_9[:,1], probe_hip_9[:,2], label="Whitesides 9")
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.legend()
# pl.show()

# project drawings into the viewing plane
proj_exp_1 = np.cross(normal_straight, np.cross(probe_hip_1, normal_straight))
proj_exp_2 = np.cross(normal_flexup, np.cross(probe_hip_2, normal_flexup))
proj_exp_3 = np.cross(normal_flexdown, np.cross(probe_hip_3, normal_flexdown))
proj_exp_4 = np.cross(normal_lateral, np.cross(probe_hip_4, normal_lateral))
proj_exp_5 = np.cross(normal_medial, np.cross(probe_hip_5, normal_medial))
proj_exp_6 = np.cross(normal_straight, np.cross(probe_hip_6, normal_straight))
proj_exp_7 = np.cross(normal_flexup, np.cross(probe_hip_7, normal_flexup))
proj_exp_8 = np.cross(normal_flexdown, np.cross(probe_hip_8, normal_flexdown))
proj_exp_9 = np.cross(normal_medial, np.cross(probe_hip_9, normal_medial))

proj_exp_straight = np.concatenate((proj_exp_1, proj_exp_6), axis=0)
proj_exp_flexup = np.concatenate((proj_exp_2, proj_exp_7), axis=0)
proj_exp_flexdown = np.concatenate((proj_exp_3, proj_exp_8), axis=0)
proj_exp_lateral = np.concatenate((proj_exp_4, proj_exp_4), axis=0)
proj_exp_medial = np.concatenate((proj_exp_5, proj_exp_9), axis=0)


c_straight, m_straight = poly.polyfit(proj_exp_straight[:,0], proj_exp_straight[:,1], 1)
x_straight = np.arange(np.amin(proj_exp_straight[:,0]), np.amax(proj_exp_straight[:,0]), 0.01)
c_flexup, m_flexup = poly.polyfit(proj_exp_flexup[:,0], proj_exp_flexup[:,1], 1)
x_flexup = np.arange(np.amin(proj_exp_flexup[:,0]), np.amax(proj_exp_flexup[:,0]), 0.1)
c_flexdown, m_flexdown = poly.polyfit(proj_exp_flexdown[:,0], proj_exp_flexdown[:,1], 1)
x_flexdown = np.arange(np.amin(proj_exp_flexdown[:,0]), np.amax(proj_exp_flexdown[:,0]), 0.1)
c_lateral, m_lateral = poly.polyfit(proj_exp_lateral[:,0], proj_exp_lateral[:,1], 1)
x_lateral = np.arange(np.amin(proj_exp_lateral[:,0]), np.amax(proj_exp_lateral[:,0]), 0.1)
c_medial, m_medial = poly.polyfit(proj_exp_medial[:,0], proj_exp_medial[:,1], 1)
x_medial = np.arange(np.amin(proj_exp_medial[:,0]), np.amax(proj_exp_medial[:,0]), 0.1)

# pl.scatter(proj_exp_1[:,0], proj_exp_1[:,1])    # straight on #1
# pl.scatter(proj_exp_6[:,0], proj_exp_6[:,1])    # straight on #2
markstyle = "x"
alp = 0.7
lw = 0.8
size = 5
pl.scatter(proj_exp_straight[:,0], proj_exp_straight[:,1], label="Straight on Projections", s=size, alpha=alp, marker=markstyle,linewidths=lw)    # straight on combined
pl.plot(x_straight, m_straight*x_straight + c_straight, c="black")
# pl.scatter(proj_exp_2[:,0], proj_exp_2[:,1])    # flexup #1
# pl.scatter(proj_exp_7[:,0], proj_exp_7[:,1])    # flexup #2
pl.scatter(proj_exp_flexup[:,0], proj_exp_flexup[:,1], label="Flex Up Projections",s=size, alpha=alp, marker=markstyle,linewidths=lw)
pl.plot(x_flexup, m_flexup*x_flexup + c_flexup, c="black")
# pl.scatter(proj_exp_3[:,0], proj_exp_3[:,1])    # flexdown #1
# pl.scatter(proj_exp_8[:,0], proj_exp_8[:,1])    # flexdown #2
pl.scatter(proj_exp_flexdown[:,0], proj_exp_flexdown[:,1], label="Flex Down Projections", s=size,alpha=alp, marker=markstyle,linewidths=lw)
pl.plot(x_flexdown, m_flexdown*x_flexdown + c_flexdown, c="black")
pl.scatter(proj_exp_lateral[:,0], proj_exp_lateral[:,1], label="Lateral Projections", s=size,alpha=alp, marker=markstyle,linewidths=lw)    # lateral
pl.plot(x_lateral, m_lateral*x_lateral + c_lateral, c="black")
# pl.scatter(proj_exp_5[:,0], proj_exp_5[:,1])    # medial #1
# pl.scatter(proj_exp_9[:,0], proj_exp_9[:,1])    # medial #2
pl.scatter(proj_exp_medial[:,0], proj_exp_medial[:,1], label="Medial Projections", s=size,alpha=alp, marker=markstyle,linewidths=lw)
pl.plot(x_medial, m_medial*x_medial + c_medial, c="black")
pl.legend()
pl.show()
# pl.scatter(proj_exp_medial[:,0], proj_exp_medial[:,1])    # medial
# pl.plot(x_medial, m_medial*x_medial + c_medial)
# pl.show()

# Angles of lines in viewing plane
theta_straight =np.arctan(m_straight)*180/np.pi
theta_flexup = np.arctan(m_flexup)*180/np.pi
theta_flexdown = np.arctan(m_flexdown)*180/np.pi
theta_lateral = np.arctan(m_lateral)*180/np.pi
theta_medial = np.arctan(m_medial)*180/np.pi

# Angle between lines compared to the straight on line
theta_flexup_rel = np.arctan((m_straight - m_flexup)/(1+ (m_straight + m_flexup)))*180/np.pi
theta_flexdown_rel = np.arctan((m_straight - m_flexdown)/(1+ (m_straight + m_flexdown)))*180/np.pi
theta_lateral_rel = np.arctan((m_straight - m_lateral)/(1+ (m_straight + m_lateral)))*180/np.pi
theta_medial_rel = np.arctan((m_straight - m_medial)/(1+ (m_straight + m_medial)))*180/np.pi
print("\nLine Angles:", np.array([theta_straight, theta_flexup, theta_flexdown, theta_lateral, theta_medial]))
print("Line Relative Angles (to straight):", np.array([theta_flexup_rel, theta_flexdown_rel, theta_lateral_rel, theta_medial_rel]))
