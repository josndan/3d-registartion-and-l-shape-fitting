import copy
import numpy as np
import open3d as o3d
from ouster_viz import Visualizer
from stitching import stitch_scans, apply_transformations
import ouster_utils
from os.path import join
from ouster.sdk import viz
from ouster import client

# frames = [i for i in range(90,111)] #90 to 110, 210 to 400
# frames = [90,95,100,105,110]
parent = join ('..','data', 'Boston_OS1-128_2023-01-02', 'Back_1_Subaru', 'split_files')

frames = [i for i in range(0,1200,4)]
# frames = [i for i in range(0,20,2)] #3599

# frames = [i for i in range(210,212,1)]

pcap = ouster_utils.Pcap(parent, '2', metadata="meta")

scans = pcap.get_scans_at_frames(frames)

cropped_scans = ouster_utils.Pcap.clean_pcd_in_scans(scans,radius=(2,float("inf")),z_threshold=(-0.1,float("inf"))) # z_threshold around -0.5

transformations = stitch_scans(cropped_scans,max_correspondence_distance=1,relative_fitness = 1e-7,
                                           relative_rmse = 1e-7,
                                           max_iteration = 1e3)

stiched_scans = apply_transformations(transformations,cropped_scans)

Visualizer.visualize_scans(stiched_scans,unique_color=True,use_pcd=True)


# Visualizer.play_scans(scans,fps=2)


# print(np.asarray(pcds[0].points).T.shape)

# plot_xyz(np.asarray(pcds[0].points))
# Visualizer.visualize_pcds(pcds,unique_color=True)

# Visualizer.play_pcds(pcds)

# threshold = 0.02

# stitched_pcds = reg_utils.stitch_pcds(pcds, threshold=threshold)
# Visualizer.visualize_pcds(stitched_pcds)

# trans_init = np.eye(4)

# source, target = pcds

# evaluation = o3d.pipelines.registration.evaluate_registration(
#     source, target, threshold, trans_init)

# print(evaluation)

# print("Apply point-to-point ICP")
# reg_p2p = o3d.pipelines.registration.registration_icp(
#     source, target, threshold, trans_init,
#     o3d.pipelines.registration.TransformationEstimationPointToPoint())
# print(reg_p2p)
# print("Transformation is:")
# print(reg_p2p.transformation)

# reg_utils.draw_registration_result(source, target, reg_p2p.transformation)


# voxel_size = 0.5

# reg_utils.draw_registration_result(source, target, np.identity(4))

# reg_utils.perform_ransac(source, target,voxel_size=voxel_size)
