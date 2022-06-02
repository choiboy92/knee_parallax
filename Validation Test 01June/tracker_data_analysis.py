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

# digprobe_filenames = ['FDL005.csv','FDM006.csv', 'FPL007.csv', 'FPM008.csv'];
# expprobe_filenames = ['StraightOn009.csv', 'FlexUp010.csv', 'FlexDown011.csv', 'Lateral012.csv', 'Medial013.csv'];
digprobe_filenames = ['FDL014.csv','FDM015.csv', 'FPL016.csv', 'FPM017.csv'];
digglasses_filenames = ['GL002.csv','GM003.csv', 'GR004.csv'];
expprobe_filenames = ['StraightOn019.csv', 'FlexUp039.csv', 'FlexDown023.csv', 'Lateral037.csv', 'Medial043.csv'];


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
exp5_probe = parse_csv(1000, expprobe_filenames[4], headslice)#[:332]
# need to slice some data to cut off unecessary readings

# pin1 indicates the tracker pin we use (in this case for GLASSES)
# therefore we use pin1 x, y, z values (n.b. these are relative to tracker)
# so we need to convert to find relative to global coordinate system from digitisation
# N.B. take avg of pin position/orientation, reshape so 1 row in matrix

pin1_009 = np.average(exp1_probe[:, 12:15], axis=0).reshape((1,-1))
pin1_009_euler = np.average(quaternion2euler(exp1_probe[:, 8:12]), axis=0).reshape((1,-1))
probe_009 = exp1_probe[:, 4:7]

pin1_010 = np.average(exp2_probe[:, 12:15], axis=0).reshape((1,-1))
pin1_010_euler = np.average(quaternion2euler(exp2_probe[:, 8:12]), axis=0).reshape((1,-1))
probe_010 = exp2_probe[:, 4:7]

pin1_011 = np.average(exp3_probe[:, 12:15], axis=0).reshape((1,-1))
pin1_011_euler = np.average(quaternion2euler(exp3_probe[:, 8:12]), axis=0).reshape((1,-1))
probe_011 = exp3_probe[:, 4:7]

pin1_012 = np.average(exp4_probe[:, 12:15], axis=0).reshape((1,-1))
pin1_012_euler = np.average(quaternion2euler(exp4_probe[:, 8:12]), axis=0).reshape((1,-1))
probe_012 = exp4_probe[:, 4:7]

pin1_013 = np.average(exp5_probe[:, 12:15], axis=0).reshape((1,-1))
pin1_013_euler = np.average(quaternion2euler(exp5_probe[:, 8:12]), axis=0).reshape((1,-1))
probe_013 = exp5_probe[:, 4:7]

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
euler_rot_009 = pin1_009_euler - dig_pin_euler
R_009 = coordTransform(euler_rot_009[0]*np.pi/180)
pin_TransRot_009 = np.matmul(R_009, dig_pin_translation)    # rotate trans vector
pin1_009 = pin1_009 + pin_TransRot_009  # add trans vector to find glasses front position
# repeat for other experiments
euler_rot_010 = pin1_010_euler - dig_pin_euler
R_010 = coordTransform(euler_rot_010[0]*np.pi/180)
pin_TransRot_010 = np.matmul(R_010, dig_pin_translation)
pin1_010 = pin1_010 + pin_TransRot_010

euler_rot_011 = pin1_011_euler - dig_pin_euler
R_011 = coordTransform(euler_rot_011[0]*np.pi/180)
pin_TransRot_011 = np.matmul(R_011, dig_pin_translation)
pin1_011 = pin1_011 + pin_TransRot_011

euler_rot_012 = pin1_012_euler - dig_pin_euler
R_012 = coordTransform(euler_rot_012[0]*np.pi/180)
pin_TransRot_012 = np.matmul(R_012, dig_pin_translation)
pin1_012 = pin1_012 + pin_TransRot_012

euler_rot_013 = pin1_013_euler - dig_pin_euler
R_013 = coordTransform(euler_rot_013[0]*np.pi/180)
pin_TransRot_013 = np.matmul(R_013, dig_pin_translation)
pin1_013 = pin1_013 + pin_TransRot_013

def viewRotate(R, basis):
    out = np.array([])
    for i in range(0,len(basis)):
        out = np.append(out, np.matmul(R, basis[i]).reshape((1,3)))
    return out.reshape((len(basis), -1))
# find new glasses basis viewing orientation still in global system
# n.b. we only focus on z axis (i.e. normal of viewing plane)
glasses_basis_009 = viewRotate(R_009, glasses_basis)
glasses_basis_010 = viewRotate(R_010, glasses_basis)
glasses_basis_011 = viewRotate(R_011, glasses_basis)
glasses_basis_012 = viewRotate(R_012, glasses_basis)
glasses_basis_013 = viewRotate(R_013, glasses_basis)

# fig = pl.figure()
# ax = pl.axes(projection='3d')
# ax.quiver(0, 0, 0, x_vec_g[0], x_vec_g[1], x_vec_g[2], color="red")
# ax.quiver(0, 0, 0, y_vec_g[0], y_vec_g[1], y_vec_g[2], color="green")
# ax.quiver(0, 0, 0, z_vec_g[0], z_vec_g[1], z_vec_g[2], color="blue")
# ax.quiver(0, 0, 0, glasses_basis_009[2,0], glasses_basis_009[2,1], glasses_basis_009[2,2], color="purple")
# ax.set_xlim(-1,1)
# ax.set_ylim(1,-1)
# ax.set_zlim(-1,1)
# pl.show()
print(glasses_basis_009)
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
# convert glass basis into proxy coordinate system
glasses_basis_proxy_009 = vec_in_new_coord_system(glasses_basis_009, proxy_basis)
glasses_basis_proxy_010 = vec_in_new_coord_system(glasses_basis_010, proxy_basis)
glasses_basis_proxy_011 = vec_in_new_coord_system(glasses_basis_011, proxy_basis)
glasses_basis_proxy_012 = vec_in_new_coord_system(glasses_basis_012, proxy_basis)
glasses_basis_proxy_013 = vec_in_new_coord_system(glasses_basis_013, proxy_basis)


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
glasses_basis_hip_009 = vec_in_new_coord_system(glasses_basis_proxy_009, hip_basis)
glasses_basis_hip_010 = vec_in_new_coord_system(glasses_basis_proxy_010, hip_basis)
glasses_basis_hip_011 = vec_in_new_coord_system(glasses_basis_proxy_011, hip_basis)
glasses_basis_hip_012 = vec_in_new_coord_system(glasses_basis_proxy_012, hip_basis)
glasses_basis_hip_013 = vec_in_new_coord_system(glasses_basis_proxy_013, hip_basis)
print(glasses_basis_hip_009)

# PLOT DATA relative to hip origin
fig = pl.figure()
ax = pl.axes(projection='3d')
ax.scatter(pin_hip_009[:,0], pin_hip_009[:,1], pin_hip_009[:,2], label="Pin position 009", c='blue')
ax.quiver(pin_hip_009[:,0], pin_hip_009[:,1], pin_hip_009[:,2], glasses_basis_hip_009[2,0]*50, glasses_basis_hip_009[2,1]*50, glasses_basis_hip_009[2,2]*50,color="blue")
ax.scatter(pin_hip_010[:,0], pin_hip_010[:,1], pin_hip_010[:,2], label="Pin position 010", c='orange')
ax.quiver(pin_hip_010[:,0], pin_hip_010[:,1], pin_hip_010[:,2], glasses_basis_hip_010[2,0]*50, glasses_basis_hip_010[2,1]*50, glasses_basis_hip_010[2,2]*50,color="orange")
ax.scatter(pin_hip_011[:,0], pin_hip_011[:,1], pin_hip_011[:,2], label="Pin position 011")
ax.quiver(pin_hip_011[:,0], pin_hip_011[:,1], pin_hip_011[:,2], glasses_basis_hip_011[2,0]*50, glasses_basis_hip_011[2,1]*50, glasses_basis_hip_011[2,2]*50)
ax.scatter(pin_hip_012[:,0], pin_hip_012[:,1], pin_hip_012[:,2], label="Pin position 012")
ax.quiver(pin_hip_012[:,0], pin_hip_012[:,1], pin_hip_012[:,2], glasses_basis_hip_012[2,0]*50, glasses_basis_hip_012[2,1]*50, glasses_basis_hip_012[2,2]*50)
ax.scatter(pin_hip_013[:,0], pin_hip_013[:,1], pin_hip_013[:,2], label="Pin position 013")
ax.quiver(pin_hip_013[:,0], pin_hip_013[:,1], pin_hip_013[:,2], glasses_basis_hip_013[2,0]*50, glasses_basis_hip_013[2,1]*50, glasses_basis_hip_013[2,2]*50)
ax.scatter(probe_hip_009[:,0], probe_hip_009[:,1], probe_hip_009[:,2], label="Whitesides 009", c='purple')
ax.scatter(probe_hip_010[:,0], probe_hip_010[:,1], probe_hip_010[:,2], label="Whitesides 010", c='red')
ax.scatter(probe_hip_011[:,0], probe_hip_011[:,1], probe_hip_011[:,2], label="Whitesides 011")
ax.scatter(probe_hip_012[:,0], probe_hip_012[:,1], probe_hip_012[:,2], label="Whitesides 012")
ax.scatter(probe_hip_013[:,0], probe_hip_013[:,1], probe_hip_013[:,2], label="Whitesides 013")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
pl.show()


pin_axes = []
fig = pl.figure()
ax = pl.axes(projection='3d')
#ax.quiver(0, 0, 0,1, 1,1)
ax.quiver(0, 0, 0, glasses_basis_hip_010[0, 0], glasses_basis_hip_010[0,1], glasses_basis_hip_010[0,2],color="red")
ax.quiver(0, 0, 0, glasses_basis_hip_010[1, 0], glasses_basis_hip_010[1,1], glasses_basis_hip_010[1,2],color="green")
ax.quiver(0, 0, 0, glasses_basis_hip_010[2, 0], glasses_basis_hip_010[2,1], glasses_basis_hip_010[2,2],color="blue")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
pl.show()
