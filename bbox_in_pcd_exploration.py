from ouster import client
from ouster import pcap
from ouster.sdk import viz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from os.path import join
import open3d as o3d
from pytransform3d.transformations import transform_from

def get_channel():
    metadata_path = join('..', 'data','Sample_Data','meta.json')
    pcap_path = join('..', 'data','Sample_Data','data.pcap')

    with open(metadata_path, 'r') as f:
        metadata = client.SensorInfo(f.read())

    source = pcap.Pcap(pcap_path, metadata)
    xyzlut = client.XYZLut(metadata)
    scans = iter(client.Scans(source))
    scan = next(scans)
    range = scan.field(client.ChanField.RANGE)
    signal = scan.field(client.ChanField.SIGNAL)

    return metadata, xyzlut, range, signal

metadata, xyzlut, range, signal = get_channel()

def get_car_coordinates():
    x1 = 260
    y1 = 70
    h = 50
    w = 140
    x2 = x1 + w 
    y2 = y1 + h
    
    destaggered_range = client.destagger(metadata,range)

    fig,ax = plt.subplots()
    rect = patches.Rectangle((x1,y1),w,h,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.imshow(destaggered_range)
    plt.show()

    return x1,y1,x2,y2

def plot_xyz(xyz,pose,bbox_xyz):

    def get_axis():
        x_ = np.array([1, 0, 0]).reshape((-1, 1))
        y_ = np.array([0, 1, 0]).reshape((-1, 1))
        z_ = np.array([0, 0, 1]).reshape((-1, 1))

        axis_n = 100
        line = np.linspace(0, 1, axis_n).reshape((1, -1))

        # basis vector to point cloud
        axis_points = np.hstack((x_ @ line, y_ @ line, z_ @ line)).transpose()

        # colors for basis vectors
        axis_color_mask = np.vstack((np.full(
            (axis_n, 4), [1, 0.1, 0.1, 1]), np.full((axis_n, 4), [0.1, 1, 0.1, 1]),
                                    np.full((axis_n, 4), [0.1, 0.1, 1, 1])))

        cloud_axis = viz.Cloud(axis_points.shape[0])

        cloud_axis.set_xyz(axis_points)
        cloud_axis.set_key(np.full(axis_points.shape[0], 0.5))
        cloud_axis.set_mask(axis_color_mask)
        cloud_axis.set_point_size(3)
        
        return cloud_axis

    point_viz = viz.PointViz("Testing")
    viz.add_default_controls(point_viz)
    axis = get_axis()
    point_viz.add(axis)

    cloud_xyz = viz.Cloud(xyz.shape[0] * xyz.shape[1])
    cloud_xyz.set_xyz(np.reshape(xyz, (-1, 3)))
    # cloud_xyz.set_key(ranges.ravel()) #This is for coloring not required

    n = bbox_xyz.shape[0] * bbox_xyz.shape[1]
    cloud_bbox_xyz = viz.Cloud(bbox_xyz.shape[0] * bbox_xyz.shape[1])
    cloud_bbox_xyz.set_xyz(np.reshape(bbox_xyz.T, (-1, 3)))
    cloud_bbox_xyz.set_mask(np.full((n,4),[1,0,0,0.5]))

    bbox = viz.Cuboid(pose, (0.5, 0.5, 0.5))

    point_viz.add(cloud_xyz)
    point_viz.add(bbox)
    point_viz.add(cloud_bbox_xyz)


    point_viz.update()
    point_viz.run()

xyz = xyzlut(range)

print(xyz.shape)

x1,y1,x2,y2 = get_car_coordinates()

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz.reshape((-1,3)))

# pcd.paint_uniform_color([1, 0.706, 0])

bbox_xyz = xyz[y1:y2,x1:x2]

bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(xyz[y1:y2,x1:x2].reshape((-1,3))),robust=True)
bbox.color = [0, 0,0]
bbox_center = bbox.get_center()

# viewer = o3d.visualization.Visualizer()
# viewer.create_window()

# viewer.add_geometry(pcd)
# viewer.add_geometry(bbox)

# axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        # size=10, origin=[0, 0, 0])

# box = o3d.geometry.TriangleMesh.create_box(*bbox.extent).translate(bbox_center,relative=False)
# box_center = box.get_center()

# box.rotate(bbox.R)

pose_matrix = transform_from(bbox.R, bbox_center)

scale = list(bbox.extent)
scale.append(1)

pose_matrix = pose_matrix @ np.diag(scale)

plot_xyz(xyz,pose_matrix,bbox_xyz)

# viewer.add_geometry(box)

# viewer.add_geometry(axis)
# opt = viewer.get_render_option()
# opt.point_size = 1.5
# opt.background_color = np.asarray([0.5, 0.5, 0.5])
# viewer.run()
# viewer.destroy_window()


# # car_points = xyz[y1:y2,x1:x2]
# car_points = xyz[:,512:]
# car_points.shape

# n = car_points.reshape((-1,3))
# n.shape


# point_viz = viz.PointViz("Testing")
# viz.add_default_controls(point_viz)
# axis = get_axis()
# point_viz.add(axis)

# cloud_xyz = viz.Cloud(n.shape[0])
# cloud_xyz.set_xyz(n)
# # cloud_xyz.set_key(ranges[y1:y2,x1:x2].ravel()/np.max(ranges[y1:y2,x1:x2])) #This is for coloring not required
# # cloud_xyz.set_key(ranges[:,1:].ravel()) #This is for coloring not required

# point_viz.add(cloud_xyz)

# point_viz.update()
# point_viz.run()

