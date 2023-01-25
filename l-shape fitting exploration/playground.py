from ouster import client
from ouster import pcap
from ouster.sdk import viz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from os.path import join
import open3d as o3d
from pytransform3d.transformations import transform_from
from ouster_viz import Visualizer
from lidar_utils import find_radius, remove_ground, radial_filter, get_points_in_plane,best_fit_plane,fit_plane,plane_filter,get_points_between_z
from pytransform3d.transformations import transform_from

from PythonRobotics.Mapping.rectangle_fitting import rectangle_fitting

def get_channel():
    metadata_path = join('..','..', 'data','from_car','OS1-64_2022-11-16','processed','Section2','meta.json')
    pcap_path = join('..','..', 'data','from_car','OS1-64_2022-11-16','processed','Section2','1.pcap')
    # metadata_path = join('..', 'data','Sample_Data','meta.json')
    # pcap_path = join('..', 'data','Sample_Data','data.pcap')

    with open(metadata_path, 'r') as f:
        metadata = client.SensorInfo(f.read())

    source = pcap.Pcap(pcap_path, metadata)
    xyzlut = client.XYZLut(metadata)
    scans = iter(client.Scans(source))
    num_frames = 1
    for i in range(num_frames):
        scan = next(scans)
        if i% 10 == 0:
            print(i)

    print(scan.frame_id)
    range_ = scan.field(client.ChanField.RANGE)
    signal = scan.field(client.ChanField.SIGNAL)

    return metadata, xyzlut, range_, signal

def get_car_coordinates(x1,y1,w,h):
    # x1 = 260
    # y1 = 70
    # h = 50
    # w = 140
    # x1,y1,w,h = [0.3168402910232544, 0.578125, 0.046875, 0.125]

    # resize_factor = (1.144901610017889, 1.144901610017889) #h,w
    global metadata, signal

    img_h = signal.shape[0]
    img_w = signal.shape[1]

    x1 -= w/2
    y1 -= h/2
    
    x1 *= img_w
    y1 *= img_h
    w  *= img_w
    h  *= img_h

    x1 = int(x1)
    y1 = int(y1)
    w = int(w)
    h = int(h)

    x2 = x1 + w
    y2 = y1 + h
    
    # fig,ax = plt.subplots()
    # rect = patches.Rectangle((x1,y1),w,h,linewidth=1,edgecolor='r',facecolor='none')
    # ax.imshow(signal)
    # ax.add_patch(rect)

    # plt.show()

    return x1,y1,x2,y2


#get data
left_clip = 182
right_clip = 266

metadata, xyzlut, range_, signal = get_channel()

signal = client.destagger(metadata,signal)[:,left_clip:-right_clip]

xyz = xyzlut(range_)
xyz = xyz[:,left_clip:-right_clip,:]

x1,y1,x2,y2 = get_car_coordinates(*[0.5373263955116272, 0.6328125, 0.1197916641831398, 0.484375])
bbox_1 = xyz[y1:y2,x1:x2]
bbox_1 = bbox_1.reshape((-1,3))

x1,y1,x2,y2 = get_car_coordinates(*[0.3168402910232544, 0.578125, 0.046875, 0.125])
bbox_2 = xyz[y1:y2,x1:x2]
bbox_2 = bbox_2.reshape((-1,3))

xyz = xyz.reshape((-1,3))

z_threshold = np.percentile(xyz[:,2],25)
print(z_threshold)
# ground = get_points_between_z(xyz,z_threshold=(z_threshold,float("-inf")))

# pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ground))
# pcd = pcd.voxel_down_sample(voxel_size=0.4)
# ground = np.asarray(pcd.points)


bbox_1 = get_points_between_z(bbox_1 ,z_threshold=(float("inf"),z_threshold))

bbox_2 = get_points_between_z(bbox_2,z_threshold=(float("inf"),z_threshold))

bbox_1 = np.concatenate((bbox_1,bbox_2),axis=0)

# plane_eqn = fit_plane(bbox_1)
# plane_eqn = best_fit_plane(bbox_1)

# bbox_1_n = plane_filter(bbox_1,plane_eqn,threshold_distance=0.2)
xyz = radial_filter(xyz.reshape((-1,3)),(3,float("inf")))

print("Before downsampling ",bbox_1.shape)

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bbox_1))
pcd = pcd.voxel_down_sample(voxel_size=0.4)
pcd, indx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.25)

pcd = pcd.select_by_index(indx)

bbox_1 = np.asarray(pcd.points)
print("After downsampling ",bbox_1.shape)
# bbox_xyz[:,2] = 0

ox = bbox_1[:,0]
oy = bbox_1[:,1]
l_shape_fitting =  rectangle_fitting.LShapeFitting()
rects, id_sets = l_shape_fitting.fitting(ox,oy)


plt.cla()
# for stopping simulation with the esc key.
plt.gcf().canvas.mpl_connect(
    'key_release_event',
    lambda event: [exit(0) if event.key == 'escape' else None])
plt.axis("equal")
plt.plot(0.0, 0.0, "*r")

# Plot range observation
for ids in id_sets:
    x = [ox[i] for i in range(len(ox)) if i in ids]
    y = [oy[i] for i in range(len(ox)) if i in ids]

    for (ix, iy) in zip(x, y):
        plt.plot([0.0, ix], [0.0, iy], "-og")

    plt.plot([ox[i] for i in range(len(ox)) if i in ids],
                [oy[i] for i in range(len(ox)) if i in ids],
                "o")
for rect in rects:
    rect.plot()

poses = []
for rect in rects:

    rect.calc_rect_contour()

    corner_x, corner_y = rect.rect_c_x, rect.rect_c_y

    c_x = (corner_x[0] + corner_x[2])/2
    c_y = (corner_y[0] + corner_y[2])/2

    p1,p2 = np.asarray([corner_x[0],corner_y[0]]),np.asarray([corner_x[1],corner_y[1]])
    x_len = np.linalg.norm(p1-p2)

    p1,p2 = np.asarray([corner_x[0],corner_y[0]]),np.asarray([corner_x[3],corner_y[3]])
    y_len = np.linalg.norm(p1-p2)

    theta = np.arccos(rect.a[0])

    radius = find_radius([[corner_x[0],corner_y[0]],
                    [corner_x[1],corner_y[1]],
                    [corner_x[2],corner_y[2]],
                    [corner_x[3],corner_y[3]] ])
    

    obj_points = radial_filter(bbox_1, radius,lambda x : x[:,:2])
    print("obj_points",obj_points.shape)

    if obj_points.shape[0] > 3:
        o3d_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(obj_points),robust=True)

        rot_mat = o3d.geometry.get_rotation_matrix_from_xyz((0,0,theta))

        c_z = o3d_bbox.get_center()[2]
        z_len = o3d_bbox.extent[2]

        pose_matrix = transform_from(rot_mat, [c_x,c_y,c_z])

        scale = [x_len,y_len,z_len,1]
        pose_matrix = pose_matrix @ np.diag(scale)

        poses.append(pose_matrix)

with Visualizer() as viz:
    viz.add_xyz(xyz)
    for pose in poses:
        viz.add_bbox(pose)
    viz.run()


# plt.pause(100)
