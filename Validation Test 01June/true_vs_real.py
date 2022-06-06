# PROJECT TRUE REFERENCES INTO REAL VIEWING POSITION & ORIENTATION
# (real viewing position is measured from glasses position & orientation)
#
# COMPARE TRUE PROJECTED vs REAL PROJECTED
#

import tracker_data_analysis as tda
import numpy as np
import matplotlib.pyplot as pl

def viewing_plane_transform(angle_val):
  flexion = angle_val[0]*np.pi/180
  var_val = angle_val[1]*np.pi/180
  int_ext = angle_val[2]*np.pi/180

  Rx = np.array([[1, 0, 0],
                [0, np.cos(flexion), -np.sin(flexion)],
                [0, np.sin(flexion), np.cos(flexion)]])
  Ry = np.array([[np.cos(var_val), 0, np.sin(var_val)],
                [0, 1, 0],
                [-np.sin(var_val), 0, np.cos(var_val)]])
  Rz = np.array([[np.cos(int_ext), -np.sin(int_ext), 0],
                [np.sin(int_ext), np.cos(int_ext), 0],
                [0, 0, 1]])
  R = Rx.dot(Ry).dot(Rz)
  return R

# Coordinates of references in drawing plane
whitesides_ant = np.array([-0.4435, 45.8136, -0.0914])/10
whitesides_post = np.array([3.6296, -14.9831, 0.7476])/10
whitesides = whitesides_ant - whitesides_post

epicondylar_med = np.array([44.4236, -0.0000, 9.1508])/10
epicondylar_lat = np.array([-44.4236, 0.0000, -9.1508])/10
epicondylar = epicondylar_med - epicondylar_lat

posteriorcondylar_med = np.array([26.692, 23.254, 6.852])/10
posteriorcondylar_lat = np.array([-28.787, 23.737, 4.3025])/10
posteriorcondylar = posteriorcondylar_med - posteriorcondylar_lat
theta_og = np.abs(np.arccos(np.dot(epicondylar, whitesides)/(np.linalg.norm(epicondylar)*np.linalg.norm(whitesides)))*(180/np.pi))

# Set camera and viewing positions
straight_coord = -tda.hip2epimid_inhip + tda.pin_hip_straight
straight_flexup = -tda.hip2epimid_inhip + tda.pin_hip_flexup
straight_flexdown = -tda.hip2epimid_inhip + tda.pin_hip_flexdown
straight_lateral = -tda.hip2epimid_inhip + tda.pin_hip_lateral
straight_medial = -tda.hip2epimid_inhip + tda.pin_hip_medial

# Find projected vectors of true Whitesides onto REAL view plane
# !!focus only on Whiteside for now!!!
# proj_epicondylar = np.cross(tda.normal_unit, np.cross(epicondylar, tda.normal_unit))
# proj_postcond = np.cross(tda.normal_unit, np.cross(posteriorcondylar, tda.normal_unit))
proj_whiteside_straight = np.cross(tda.normal_straight, np.cross(whitesides, tda.normal_straight))
proj_whiteside_flexup = np.cross(tda.normal_flexup, np.cross(whitesides, tda.normal_flexup))
proj_whiteside_flexdown = np.cross(tda.normal_flexdown, np.cross(whitesides, tda.normal_flexdown))
proj_whiteside_lateral = np.cross(tda.normal_lateral, np.cross(whitesides, tda.normal_lateral))
proj_whiteside_medial = np.cross(tda.normal_medial, np.cross(whitesides, tda.normal_medial))

# Projected True and Real Whitesides Line Plot
pl.plot([0,proj_whiteside_straight[0,0]],[0, proj_whiteside_straight[0,1]], label="Straight", color="blue",linestyle = 'dashed')
pl.plot([0,proj_whiteside_flexup[0,0]],[0, proj_whiteside_flexup[0,1]], label="Flexion", color="green",linestyle = 'dashed')
pl.plot([0,proj_whiteside_flexdown[0,0]],[0, proj_whiteside_flexdown[0,1]], label="Extension", color="orange",linestyle = 'dashed')
pl.plot([0,proj_whiteside_lateral[0,0]],[0, proj_whiteside_lateral[0,1]], label="Lateral", color="red",linestyle = 'dashed')
pl.plot([0,proj_whiteside_medial[0,0]],[0, proj_whiteside_medial[0,1]], label="Medial", color="purple",linestyle = 'dashed')
x_max = proj_whiteside_medial[0,0]
pl.plot([0,x_max],[0, tda.m_straight*x_max], label="Real Straight", color="blue")
pl.plot([0,x_max],[0, tda.m_flexup*x_max], label="Real Flexion", color="green")
pl.plot([0,x_max],[0, tda.m_flexdown*x_max], label="Real Extension", color="orange")
pl.plot([0,x_max],[0, tda.m_lateral*x_max], label="Real Lateral", color="red")
#pl.plot([0,x_max],[0, tda.m_medial*x_max], label="Real Medial", color="purple",linestyle = 'dashed')
pl.xlabel("x")
pl.ylabel("y")
pl.legend(loc=0)
pl.show()

ref = proj_whiteside_straight[0,:2]
PWFU = proj_whiteside_flexup[0,:2]
PWFD = proj_whiteside_flexdown[0,:2]
PWL = proj_whiteside_lateral[0,:2]
PWM = proj_whiteside_medial[0,:2]

# Relative Angles of Projected True Whiteside
theta_flexup_rel = np.arccos(np.dot(ref, PWFU)/(np.linalg.norm(ref)*np.linalg.norm(PWFU)))*(180/np.pi)
theta_flexdown_rel = np.arccos(np.dot(ref, PWFD)/(np.linalg.norm(ref)*np.linalg.norm(PWFD)))*(180/np.pi)
theta_lateral_rel = np.arccos(np.dot(ref, PWL)/(np.linalg.norm(ref)*np.linalg.norm(PWL)))*(180/np.pi)
theta_medial_rel = np.arccos(np.dot(ref, PWM)/(np.linalg.norm(ref)*np.linalg.norm(PWM)))*(180/np.pi)
print("Relative Angles of Projected True Whiteside:", np.array([theta_flexup_rel, theta_flexdown_rel, theta_lateral_rel, theta_medial_rel]))

# Compare angles between real and projected true (from proj to real)
theta_PWS = np.arctan(proj_whiteside_straight[0,1]/proj_whiteside_straight[0,0])*(180/np.pi)
theta_PWFU = np.arctan(proj_whiteside_flexup[0,1]/proj_whiteside_flexup[0,0])*(180/np.pi)
theta_PWFD = np.arctan(proj_whiteside_flexdown[0,1]/proj_whiteside_flexdown[0,0])*(180/np.pi)
theta_PWL = np.arctan(proj_whiteside_lateral[0,1]/proj_whiteside_lateral[0,0])*(180/np.pi)
theta_PWM = np.arctan(proj_whiteside_medial[0,1]/proj_whiteside_medial[0,0])*(180/np.pi)
theta_PWS_rel = np.abs(theta_PWFU)-np.abs(tda.theta_straight)
theta_PWFU_rel = np.abs(theta_PWFU)-np.abs(tda.theta_flexup)
theta_PWFD_rel = np.abs(theta_PWFD)-np.abs(tda.theta_flexdown)
theta_PWL_rel = np.abs(theta_PWL)-np.abs(tda.theta_lateral)
theta_PWM_rel = -(180-np.abs(tda.theta_medial) - np.abs(theta_PWM)) # as theta_medial measure from +ve x-axis
print("Compare Real vs Projected True lines:", np.array([theta_PWS_rel,theta_PWFU_rel, theta_PWFD_rel, theta_PWL_rel, theta_PWM_rel]))
