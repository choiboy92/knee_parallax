import csv
from itertools import islice
import numpy as np
import math
import matplotlib.pyplot as pl


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

digprobe_filenames = ['FDL005.csv','FDM006.csv', 'FPL007.csv', 'FPM008.csv'];
expprobe_filenames = ['StraightOn009.csv', 'FlexUp010.csv', 'FlexDown011.csv', 'Lateral012.csv', 'Medial013.csv'];

# EXTRACT DIGITISATION DATA
headslice = 5;
dig_frame = 100;
data_probe = [];
for item in digprobe_filenames:
    data_probe += [parse_csv(dig_frame, item, headslice)];
data_probe = np.array(data_probe);  # data_probe[file, frame num, column]
# cols = calib q0, qx, qy, qz, x, y, z, error, pin1 q0, qx, qy, qz, x, y, z
# calib x, y, z are the ones we are meant to use for digitisation

# average digitisation columns (only calib x, y, z)
dig_probe = [];
for i in range(0, len(digprobe_filenames)):
    dig_probe += [np.average(data_probe[i, :, 4:7], axis=0)];
dig_probe = np.array(dig_probe);    # dig_probe = [file, cols]
# e.g. dig_probe[0,0] = x pos for FL001 digitisation
#      dig_probe[0,1] = y pos for FL001 digitisation, etc.

# EXTRACT EXPERIMENT DATA
exp1_probe = parse_csv(1000, expprobe_filenames[0], headslice)
exp2_probe = parse_csv(1000, expprobe_filenames[1], headslice)[:800]
exp3_probe = parse_csv(500, expprobe_filenames[2], headslice)
exp4_probe = parse_csv(500, expprobe_filenames[3], headslice)
exp5_probe = parse_csv(500, expprobe_filenames[4], headslice)[:332]
# need to slice some data to cut off unecessary readings

# pin1 indicates the tracker pin we use (in this case for PEN)
# therefore we use pin1 x, y, z values (n.b. these are relative to tracker)
# so we need to convert to find relative to coordinate system from digitisation
pin1_009 = exp1_probe[:, 12:15]
pin1_009_quat = exp1_probe[:, 8:12]
probe_009 = exp1_probe[:, 4:7]

pin1_010 = exp2_probe[:, 12:15]
pin1_010_quat = exp2_probe[:, 8:12]
probe_010 = exp2_probe[:, 4:7]

pin1_011 = exp3_probe[:, 12:15]
pin1_011_quat = exp3_probe[:, 8:12]
probe_011 = exp3_probe[:, 4:7]

pin1_012 = exp4_probe[:, 12:15]
pin1_012_quat = exp4_probe[:, 8:12]
probe_012 = exp4_probe[:, 4:7]

pin1_013 = exp5_probe[:, 12:15]
pin1_013_quat = exp5_probe[:, 8:12]
probe_013 = exp5_probe[:, 4:7]

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
pin_rel_vec_009 = pin1_009 - origin;
pin_rel_vec_010 = pin1_010 - origin;
pin_rel_vec_011 = pin1_011 - origin;
pin_rel_vec_012 = pin1_012 - origin;
pin_rel_vec_013 = pin1_013 - origin;

probe_rel_vec_009 = probe_009 - origin;
probe_rel_vec_010 = probe_010 - origin;
probe_rel_vec_011 = probe_011 - origin;
probe_rel_vec_012 = probe_012 - origin;
probe_rel_vec_013 = probe_013 - origin;


# n.b. still in terms of tracker global coordinate system
# Transform into proxy coordinate system
def vec_in_new_coord_system(vec, basis):
    vec_rel_proxy = []
    for i in range(0, len(vec)):
        newproxy = np.dot(basis, vec[i])
        vec_rel_proxy += [newproxy]
    return np.array(vec_rel_proxy).reshape((-1, 3))


pin_proxy_009 = vec_in_new_coord_system(pin_rel_vec_009, proxy_basis)
pin_proxy_010 = vec_in_new_coord_system(pin_rel_vec_010, proxy_basis)
pin_proxy_011 = vec_in_new_coord_system(pin_rel_vec_011, proxy_basis)
pin_proxy_012 = vec_in_new_coord_system(pin_rel_vec_012, proxy_basis)
pin_proxy_013 = vec_in_new_coord_system(pin_rel_vec_013, proxy_basis)
probe_proxy_009 = vec_in_new_coord_system(probe_rel_vec_009, proxy_basis)
probe_proxy_010 = vec_in_new_coord_system(probe_rel_vec_010, proxy_basis)
probe_proxy_011 = vec_in_new_coord_system(probe_rel_vec_011, proxy_basis)
probe_proxy_012 = vec_in_new_coord_system(probe_rel_vec_012, proxy_basis)
probe_proxy_013 = vec_in_new_coord_system(probe_rel_vec_013, proxy_basis)


### ------------------- RELATE PROXY TO HIP COORD SYSTEM --------------------
hip2proxy = np.array([[-54.61, 24.79, -433.9]]) # n.b. in proxy system coords
y_rot = -np.arctan(54.66/433.72)    # clockwise rotation about y axis
x_rot = -np.arctan(13.88/433.72)    # clockwise rotation about x axis
trans_angle = np.array([x_rot, y_rot, 0])

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
pin_hip_009 = vec_in_new_coord_system(pin_proxy_009 + hip2proxy, hip_basis)
pin_hip_010 = vec_in_new_coord_system(pin_proxy_010 + hip2proxy, hip_basis)
pin_hip_011 = vec_in_new_coord_system(pin_proxy_011 + hip2proxy, hip_basis)
pin_hip_012 = vec_in_new_coord_system(pin_proxy_012 + hip2proxy, hip_basis)
pin_hip_013 = vec_in_new_coord_system(pin_proxy_013 + hip2proxy, hip_basis)
probe_hip_009 = vec_in_new_coord_system(probe_proxy_009 + hip2proxy, hip_basis)
probe_hip_010 = vec_in_new_coord_system(probe_proxy_010 + hip2proxy, hip_basis)
probe_hip_011 = vec_in_new_coord_system(probe_proxy_011 + hip2proxy, hip_basis)
probe_hip_012 = vec_in_new_coord_system(probe_proxy_012 + hip2proxy, hip_basis)
probe_hip_013 = vec_in_new_coord_system(probe_proxy_013 + hip2proxy, hip_basis)

# PLOT DATA relative to hip origin
fig = pl.figure()
ax = pl.axes(projection='3d')
ax.scatter(pin_hip_009[:,0], pin_hip_009[:,1], pin_hip_009[:,2], label="Pin position 009", c='blue')
ax.scatter(pin_hip_010[:,0], pin_hip_010[:,1], pin_hip_010[:,2], label="Pin position 010", c='orange')
ax.scatter(pin_hip_011[:,0], pin_hip_011[:,1], pin_hip_011[:,2], label="Pin position 011")
ax.scatter(pin_hip_012[:,0], pin_hip_012[:,1], pin_hip_012[:,2], label="Pin position 012")
ax.scatter(pin_hip_013[:,0], pin_hip_013[:,1], pin_hip_013[:,2], label="Pin position 013")
ax.scatter(probe_hip_009[:,0], probe_hip_009[:,1], probe_hip_009[:,2], label="Whitesides 009", c='purple')
ax.scatter(probe_hip_010[:,0], probe_hip_010[:,1], probe_hip_010[:,2], label="Whitesides 010", c='red')
ax.scatter(probe_hip_011[:,0], probe_hip_011[:,1], probe_hip_011[:,2], label="Whitesides 011")
ax.scatter(probe_hip_012[:,0], probe_hip_012[:,1], probe_hip_012[:,2], label="Whitesides 012")
ax.scatter(probe_hip_013[:,0], probe_hip_013[:,1], probe_hip_013[:,2], label="Whitesides 013")
ax.legend()
pl.show()

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

# pin1_006_euler = quaternion2euler(pin1_006_quat)*np.pi/180
# pin_axes = []
# fig = pl.figure()
# ax = pl.axes(projection='3d')
# ax.quiver(0, 0, 0, pin_axes[0], pin_axes[1], pin_axes[2])
# ax.set_xlim([-1, 0.5])
# ax.set_ylim([-1, 1.5])
# ax.set_zlim([-10, 8])
# pl.show()
# pin_axes = np.matmul(coordTransform(pin1_006_euler[0]), original_basis)
# print(pin_axes)
# pin_axes = vec_in_new_coord_system(pin_axes, proxy_basis)
# print(pin_axes)
# pin_axes = vec_in_new_coord_system(pin_axes, hip_basis)
# print(pin_axes)
