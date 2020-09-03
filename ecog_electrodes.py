from mayavi import mlab
import numpy as np
import trimesh
from copy import deepcopy
from functions import cluster_centers, center_electrodes, sort_electrodes, translate_electrodes_onto_the_brain_surface, brain_plot, plot_gamma_motor, plot_gamma_touch, plot_alpha_motor, plot_alpha_touch, plot_accuracy, make_T, move_points, plot_spline


subject_name = 'Name'
directory = r'.../ecog_electrodes/example/'
subjects_dir = directory+'subjects'
point_cloud_file_from_meshlab = directory+'Segmentation.xyz'

# in 3dslicer segmentation have to be saved in RAS
n_clusters = 64  # number of electrodes
elec_0 = cluster_centers(point_cloud_file_from_meshlab, n_clusters)
# you also can use stl file for electrodes visualization
# mesh = trimesh.load(directory+'Segmentation.stl')
# meshes = mesh.split()
# elec_0 = [i.center_mass for i in meshes]

# The center of fresurfer's volume file is not at (0,0,0).
# Obtain the offset with
# mri_info --cras /tmp/subjects/Name/mri/orig.mgz
# t for Yacyuk
t = np.array([-1.59642, -13.9633, -14.8403])

# I have a RAS point on the surface (tkrR tkrA tkrS)
# and want to compute the MNI305 RAS that corresponds to this point:
# MNI305RAS = TalXFMNoriginv(Torig)*[tkrR tkrA tkrS 1]'
# TalXFM: /tmp/subjects/Name/mri/transforms/talairach.xfm --> m1
# Norig: mri_info --vox2ras /tmp/subjects/Name/mri/orig.mgz --> m2
# Torig: mri_info --vox2ras-tkr /tmp/subjects/Name/mri/orig.mgz --> same for all subjects
# m1,m2 for Yacyuk
m1 = np.array([[1.070332, 0.003227, 0.008803, -0.094025],
               [0.029746, 1.024466, 0.336834, 15.886017],
               [-0.005421, -0.293166, 1.068191, -20.031723],
               [0.0, 0.0, 0.0, 1.0]])
m2 = np.array([[-1.00000,   0.00000,    -0.00000,   126.40357],
               [-0.00000,   0.00000,    1.00000,  -141.96332],
               [-0.00000,  -1.00000,    0.00000,   113.15971],
               [0.00000,   0.00000,    0.00000,   1.00000]])
elec_centered = center_electrodes(m1, m2, elec_0, t)


# sort electrodes
elec_sorted = sort_electrodes(elec_centered)

# translate electrodes onto the brain surface
# wait: it takes some time
elec_up = translate_electrodes_onto_the_brain_surface(
    subject_name, subjects_dir, elec_sorted)

# to export data to BrainStorm freesurfer's offset sould be added
l_BS = deepcopy(elec_up) + t
np.savetxt(directory+subject_name + '_electrodes_for_BS.txt', l_BS)

# if you already have _electrodes_for_BS.txt file
elec_up = np.loadtxt(directory+subject_name + '_electrodes_for_BS.txt')
elec_up -= t

# chouse point of view
mlabViewX = 190
mlabViewY = 60

# checkpoint
# plot electrodes as numbers
# maybe you'll have to swape axes or revers electrodes order
textBool = 1
brain_plot(subject_name, subjects_dir, elec_up, mlabViewX, mlabViewY, textBool)

# plot electrodes as spheres
textBool = 0
brain_plot(subject_name, subjects_dir, elec_up, mlabViewX, mlabViewY, textBool)

# plot gamma, alpha, accuracy
# save png & 3d data or not
saveDataBool = 0
dir_n = directory+'res_230319/'
plot_gamma_motor(subject_name, subjects_dir, dir_n, elec_up,
                 mlabViewX, mlabViewY, saveDataBool)
file_n = directory+'res_touch.mat'
plot_gamma_touch(subject_name, subjects_dir, file_n,
                 elec_up, mlabViewX, mlabViewY, saveDataBool)
plot_alpha_motor(subject_name, subjects_dir, dir_n, elec_up,
                 mlabViewX, mlabViewY, saveDataBool)
plot_alpha_touch(subject_name, subjects_dir, file_n,
                 elec_up, mlabViewX, mlabViewY, saveDataBool)
dir_n = directory+'pet67_paper/'
plot_accuracy(subject_name, subjects_dir, dir_n, elec_up,
              mlabViewX, mlabViewY, saveDataBool)

# TPS
# source control points
x, y = np.linspace(0, 0.021, 8), np.linspace(
    0, 0.021, 8)  # construct electrodes grid
x, y = np.meshgrid(x, y)
xs = x.flatten()
ys = y.flatten()
control_points = np.vstack([xs, ys, np.zeros(64)]).T

# make T
T = make_T(control_points)

# target control points
points = deepcopy(elec_up)/1000
xt = points[:, 0]
yt = points[:, 1]
zt = points[:, 2]

# solve cx, cy, cz (coefficients for x, y, z)
xtAug = np.concatenate([xt, np.zeros(4)])
ytAug = np.concatenate([yt, np.zeros(4)])
ztAug = np.concatenate([zt, np.zeros(4)])
cx = np.dot(np.linalg.pinv(T), xtAug)  # [K+4]
cy = np.dot(np.linalg.pinv(T), ytAug)
cz = np.dot(np.linalg.pinv(T), ztAug)
C = np.array((cx, cy, cz))

# dense grid
N = 50
x = np.linspace(0, .021, N)
y = np.linspace(0, .021, N)
x, y = np.meshgrid(x, y)
xgs, ygs = x.flatten(), y.flatten()
grid_points = np.vstack([xgs, ygs, np.zeros(N*N)]).T

# transform
p_moved = move_points(grid_points, control_points)  # [N x (K+4)]
xgt = np.dot(p_moved, cx.T)
ygt = np.dot(p_moved, cy.T)
zgt = np.dot(p_moved, cz.T)

plot_spline(subject_name, subjects_dir, xgt, ygt,
            zgt, elec_up, mlabViewX, mlabViewY)

mlab.show()
