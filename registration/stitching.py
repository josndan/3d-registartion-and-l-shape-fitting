import numpy as np
import copy
import open3d as o3d

def stitch_scans(scans,max_correspondence_distance = 0.05,trans_init=None,relative_fitness = 1e-6,
                                           relative_rmse = 1e-6,
                                           max_iteration = 100):

    if trans_init is None:
        trans_init = np.eye(4)

    print("Apply point-to-point ICP")
    print('-'*80)

    target = scans[0]
    transformations = []
    for i, source in enumerate(scans[1:]):
        # source = copy.deepcopy(source)
       
        print(f"Before Stitching frame {i} and {i+1}")
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source.pcd, target.pcd, max_correspondence_distance, trans_init)
        print(evaluation)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source.pcd, target.pcd, max_correspondence_distance, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness,
                                           relative_rmse,
                                           int(max_iteration)))

        transformations.append(reg_p2p.transformation)
        # source.pcd = source.pcd.transform(reg_p2p.transformation)
        
        print(f"After Stitching frame {i} and {i+1}")
        print(reg_p2p)
        print('-'*80)

        target = source

    return transformations

def apply_transformations(transformations,scans):
    
    res = [scans[0]]

    for i,source in enumerate(scans[1:]):

        for transformation in reversed(transformations[:i+1]):
            source.pcd = source.pcd.transform(transformation)
        
        res.append(source)

    return res