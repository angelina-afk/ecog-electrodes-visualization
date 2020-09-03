
import numpy as np
import numpy.linalg as nl
# import ipdb
import mne
from mne.viz import plot_alignment
from scipy.io import loadmat
from scipy.spatial.distance import pdist, cdist, squareform
from scipy import linalg
from itertools import combinations
from sklearn import cluster
from mayavi import mlab
from surfer import Brain
from os import listdir
from scipy.spatial import ConvexHull
from copy import deepcopy

def cluster_centers (point_cloud_file_from_meshlab, n_clusters):
    kmeans = cluster.KMeans(n_clusters=n_clusters,
                            algorithm='elkan',
                            random_state=0).fit(np.loadtxt(point_cloud_file_from_meshlab))
    return kmeans.cluster_centers_  

def center_electrodes(m1,m2,elec_0,t):
    m3 = np.array([[ -1.00000,    0.00000,    0.00000, 128.00000],
                   [   0.00000,    0.00000,    1.00000, -128.00000],
                   [   0.00000,   -1.00000,    0.00000,  128.00000],
                   [   0.00000,    0.00000,    0.00000,    1.00000]])
    m4 = np.linalg.pinv(m3)
    el_MNI305RAS = np.dot(m1,np.dot(m2,np.dot(m4,np.c_[elec_0, np.ones(64)].T))).T[:,:-1]
    return el_MNI305RAS - t 

def sort_electrodes(elec_centered):
    #change coordinate system for sorting along grid sides 
    el_mean = np.mean(elec_centered,0)
    pl=np.array(sorted(elec_centered, key = lambda x: cdist([el_mean],[x]),reverse=True)[:3])
    vec=np.array(sorted([x[1]-x[0] for x in combinations(pl,2)],key=np.linalg.norm)[:2])
    vec=np.vstack((vec,np.cross(vec[0],vec[1])))
    X = np.linalg.solve(np.eye(3,3),vec)
    A1=np.dot(X,elec_centered.T).T
    #sort
    el_sorted = np.array(sorted(A1, key=lambda x: (-x[0]))).reshape(8,8,3)
    el_sorted=np.array([sorted(y, key=lambda x: (-x[1])) for y in el_sorted] ).swapaxes(0,1).reshape(64,3)
    return np.dot(np.linalg.inv(X),np.array(el_sorted).T).T

def translate_electrodes_onto_the_brain_surface(subject_name, subjects_dir, elec_sorted):
    #find brain surface points
    brain = Brain(subject_id=subject_name, subjects_dir=subjects_dir, surf='pial', hemi ='lh',offscreen=True)
    surf = brain.geo['lh']
    src= np.array([surf.x,surf.y,surf.z]).T
    #use ConvexHull to chek if a point is inside the brain or not
    elec_up = np.zeros([64,3])
    hull = ConvexHull(src)
    for i, e in enumerate(elec_sorted):
        a = 1
        n = deepcopy(e)
        while a==1:
            new_hull = ConvexHull(np.concatenate((hull.points, [e])))
            if np.array_equal(new_hull.vertices, hull.vertices):  
                a = 1
                e += n*0.01
            else:
                a=0
        elec_up[i] = e
    return elec_up

def brain_plot(subject_name, subjects_dir, electrodes, mlabViewX, mlabViewY, textBool):
    #make info file
    ch_names = [str(x) for x in np.arange(64)]
    dig_ch_pos = dict(zip(ch_names, electrodes/1000)) 
    mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
    info = mne.create_info(ch_names, 1000., ch_types='ecog', montage=mon)
    #plot electrodes as spheres or as numbers
    if textBool:
        fig = mlab.figure('brain text',size=(800, 700))
        #plot subject's brain
        plot_alignment(info,
                   subject=subject_name, 
                   subjects_dir=subjects_dir, 
                   surfaces=['pial'],
                   ecog=False,
                   fig=fig)
        fig.scene.disable_render = True

        for i, x in enumerate(electrodes):
                mlab.text3d(x[0]/1000, 
                            x[1]/1000, 
                            x[2]/1000, 
                            str(i+1), 
                            scale=0.001)

                fig.scene.disable_render = False # Super duper trick
    else:
        fig = mlab.figure('brain spheres',size=(800, 700))
        #plot subject's brain
        plot_alignment(info,
                   subject=subject_name, 
                   subjects_dir=subjects_dir, 
                   surfaces=['pial'],
                   ecog=False,
                   fig=fig)
        mlab.points3d(electrodes[:,0]/1000, 
                      electrodes[:,1]/1000, 
                      electrodes[:,2]/1000,
                      color = (1,0,0),
                      scale_factor=0.002)
    
    #chouse point of view
    mlab.view(mlabViewX, mlabViewY) 

def plot_gamma_motor(subject_name, subjects_dir, dir_n, elec_up, mlabViewX, mlabViewY, saveDataBool):
    #make info file
    ch_names = [str(x) for x in np.arange(64)]
    dig_ch_pos = dict(zip(ch_names, elec_up/1000)) 
    mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
    info = mne.create_info(ch_names, 1000., ch_types='ecog', montage=mon)
    
    name = listdir(dir_n)
    for nam in name:
        if nam != '.DS_Store':
            gama_t=loadmat(dir_n + nam)
            res_re = gama_t['res']
            s = np.mean(res_re[:,15:,0],axis=1) #gamma
            s=np.interp(s, (s.min(), s.max()), (0, 1))
            fig = mlab.figure('gamma_motor_'+nam,size=(800, 700))
            plot_alignment(info,
                               subject=subject_name, 
                               subjects_dir=subjects_dir, 
                               surfaces=['pial'],
                               ecog=False,
                               fig=fig)
            obj=mlab.points3d(elec_up[:,0]/1000, 
                              elec_up[:,1]/1000, 
                              elec_up[:,2]/1000, 
                              s, 
                              colormap='rainbow', 
                              scale_mode='none',
                              scale_factor=0.002)
            mlab.colorbar(object=obj, 
                          title='gamma', 
                          orientation='vertical', 
                          nb_labels=8, 
                          nb_colors=None, 
                          label_fmt=None)
            mlab.title(nam[:-4],
                       size=0.2,
                       height=0.015)

            #chouse point of view
            mlab.view(mlabViewX, mlabViewY) 
            if saveDataBool:
                mlab.savefig(nam[:-4] + '.png')
                mlab.savefig(nam[:-4] + '.x3d')

def plot_gamma_touch(subject_name, subjects_dir, file_n, elec_up, mlabViewX, mlabViewY, saveDataBool):
    #make info file
    ch_names = [str(x) for x in np.arange(64)]
    dig_ch_pos = dict(zip(ch_names, elec_up/1000)) 
    mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
    info = mne.create_info(ch_names, 1000., ch_types='ecog', montage=mon)
    
    data_t = loadmat(file_n)
    res_re = data_t['res']
    for i in range(5):
        s = np.mean(res_re[:,15:,i],axis=1) #gamma
        s=np.interp(s, (s.min(), s.max()), (0, 1))
        fig = mlab.figure('gamma_touch_'+str(i),size=(800, 700))
        plot_alignment(info,
                            subject=subject_name, 
                            subjects_dir=subjects_dir, 
                            surfaces=['pial'],
                            ecog=False,
                            fig=fig)
        obj=mlab.points3d(elec_up[:,0]/1000, 
                          elec_up[:,1]/1000, 
                          elec_up[:,2]/1000, 
                          s, colormap='rainbow', 
                          scale_mode='none',
                          scale_factor=0.002)
        mlab.colorbar(object=obj, 
                      title='gamma', 
                      orientation='vertical', 
                      nb_labels=8, 
                      nb_colors=None, 
                      label_fmt=None)
        mlab.title('touch '+str(i+1),
                   size=0.2,
                   height=0.015)
        #chouse point of view
        mlab.view(mlabViewX, mlabViewY)         
        if saveDataBool:
            mlab.savefig('touch_'+str(i+1) + '.png')
            mlab.savefig('touch_'+str(i+1) + '.x3d')

def plot_alpha_motor(subject_name, subjects_dir, dir_n, elec_up, mlabViewX, mlabViewY, saveDataBool):
    #make info file
    ch_names = [str(x) for x in np.arange(64)]
    dig_ch_pos = dict(zip(ch_names, elec_up/1000)) 
    mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
    info = mne.create_info(ch_names, 1000., ch_types='ecog', montage=mon)
    
    name = listdir(dir_n)
    for nam in name:
        if nam != '.DS_Store':
            gama_t=loadmat(dir_n + nam)
            res_re = gama_t['res']
            s = np.mean(res_re[:,:5,0],axis=1) #alpha
            s=np.interp(s, (s.min(), s.max()), (0, 1))
            fig = mlab.figure('alpha_motor_'+nam,size=(800, 700))
            plot_alignment(info,
                               subject=subject_name, 
                               subjects_dir=subjects_dir, 
                               surfaces=['pial'],
                               ecog=False,
                               fig=fig)
            obj=mlab.points3d(elec_up[:,0]/1000, 
                              elec_up[:,1]/1000, 
                              elec_up[:,2]/1000, 
                              s, 
                              colormap='rainbow', 
                              scale_mode='none',
                              scale_factor=0.002)
            mlab.colorbar(object=obj, 
                          title='alpha', 
                          orientation='vertical', 
                          nb_labels=8, 
                          nb_colors=None, 
                          label_fmt=None)
            mlab.title(nam[:-4],
                       size=0.2,
                       height=0.015)
            #chouse point of view
            mlab.view(mlabViewX, mlabViewY) 
            if saveDataBool:
                mlab.savefig('alpha_'+nam[:-4] + '.png')
                mlab.savefig('alpha_'+nam[:-4] + '.x3d')

def plot_alpha_touch(subject_name, subjects_dir, file_n, elec_up, mlabViewX, mlabViewY, saveDataBool):
    #make info file
    ch_names = [str(x) for x in np.arange(64)]
    dig_ch_pos = dict(zip(ch_names, elec_up/1000)) 
    mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
    info = mne.create_info(ch_names, 1000., ch_types='ecog', montage=mon)
    
    data_t = loadmat(file_n)
    res_re = data_t['res']
    for i in range(5):
        s = np.mean(res_re[:,:5,i],axis=1) #alpha
        s=np.interp(s, (s.min(), s.max()), (0, 1))
        fig = mlab.figure('alpha_touch_'+str(i),size=(800, 700))
        plot_alignment(info,
                           subject=subject_name, 
                           subjects_dir=subjects_dir, 
                           surfaces=['pial'],
                           ecog=False,
                           fig=fig)
        obj=mlab.points3d(elec_up[:,0]/1000, 
                          elec_up[:,1]/1000, 
                          elec_up[:,2]/1000, 
                          s, 
                          colormap='rainbow', 
                          scale_mode='none',
                          scale_factor=0.002)
        mlab.colorbar(object=obj, 
                      title='alpha', 
                      orientation='vertical', 
                      nb_labels=8, 
                      nb_colors=None, 
                      label_fmt=None)
        mlab.title('touch '+ str(i+1),
                   size=0.2,
                   height=0.015)
        #chouse point of view
        mlab.view(mlabViewX, mlabViewY) 
        if saveDataBool:
            mlab.savefig('alpha_touch_'+str(i+1) + '.png')
            mlab.savefig('alpha_touch_'+str(i+1) + '.x3d')

def plot_accuracy(subject_name, subjects_dir, dir_n, elec_up, mlabViewX, mlabViewY, saveDataBool):
    #make info file
    ch_names = [str(x) for x in np.arange(64)]
    dig_ch_pos = dict(zip(ch_names, elec_up/1000)) 
    mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
    info = mne.create_info(ch_names, 1000., ch_types='ecog', montage=mon)
    
    name = listdir(dir_n)
    for nam in name:
        if nam != '.DS_Store':
            s=np.loadtxt(dir_n + nam)
            s = np.nan_to_num(s[:,1])
            fig = mlab.figure('accuracy_'+nam,size=(800, 700))
            plot_alignment(info,
                               subject=subject_name, 
                               subjects_dir=subjects_dir, 
                               surfaces=['pial'],
                               ecog=False,
                               fig=fig)
            obj=mlab.points3d(elec_up[:,0]/1000, 
                              elec_up[:,1]/1000, 
                              elec_up[:,2]/1000, 
                              s, 
                              colormap='rainbow', 
                              scale_mode='none',
                              scale_factor=0.002)
            mlab.colorbar(object=obj, 
                          title='accuracy', 
                          orientation='vertical', 
                          nb_labels=8, 
                          nb_colors=None, 
                          label_fmt=None)
            mlab.title(nam[:-4],
                       size=0.2,
                       height=0.015)
            #chouse point of view
            mlab.view(mlabViewX, mlabViewY) 
            if saveDataBool:
                mlab.savefig(nam[:-4] + '.png')
                mlab.savefig(nam[:-4] + '.x3d')

def make_T(control_points):
    # control_points: [K x 2] control points
    # T: [(K+4) x (K+4)]
    K = control_points.shape[0]
    T = np.zeros((K+4, K+4))
    T[:K, 0] = 1
    T[:K, 1:4] = control_points
    T[K, 4:] = 1
    T[K+1:, 4:] = control_points.T
    R = squareform(pdist(control_points, metric='euclidean'))
    R = R * R
    R[R == 0] = 1 # a trick to make R ln(R) 0
    R = R * np.log(R)
    np.fill_diagonal(R, 0)
    T[:K, 4:] = R
    return T

def move_points(points, control_points):
    # points: [N x 2], input points
    # control_points: [K x 2], control points
    # p_moved: [N x (4+K)], moved input points
    N, K = points.shape[0], control_points.shape[0]
    p_moved = np.zeros((N, K+4))
    p_moved[:,0] = 1
    p_moved[:,1:4] = points
    R = cdist(points, control_points, 'euclidean')
    R = R * R
    R[R == 0] = 1
    R = R * np.log(R)
    p_moved[:,4:] = R
    return p_moved

def plot_spline(subject_name, subjects_dir, xgt, ygt, zgt, electrodes, mlabViewX, mlabViewY):
    #make info file
    ch_names = [str(x) for x in np.arange(64)]
    dig_ch_pos = dict(zip(ch_names, electrodes/1000)) 
    mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
    info = mne.create_info(ch_names, 1000., ch_types='ecog', montage=mon)
    fig = mlab.figure('TPS',size=(800, 700))
    plot_alignment(info,
                   subject=subject_name, 
                   subjects_dir=subjects_dir, 
                   surfaces=['pial'],
                   ecog=False,
                   fig=fig)
    pts = mlab.points3d(xgt, 
                          ygt, 
                          zgt, scale_mode='none', scale_factor=0.0003)
    # Create and visualize the mesh
    mesh = mlab.pipeline.delaunay2d(pts)
    surf = mlab.pipeline.surface(mesh, representation='surface',colormap='rainbow')
    mlab.points3d(electrodes[:,0]/1000, 
                  electrodes[:,1]/1000, 
                  electrodes[:,2]/1000, 
                  color=(1,0,0), 
                  scale_factor=0.002)
    
    mlab.view(mlabViewX, mlabViewY)