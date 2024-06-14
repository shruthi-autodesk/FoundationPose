# Author: Sai Shruthi Balaji

# The intent of this script is to capture an RGB and depth image using the realsense camera,
# Use CNOS to get a segmentation mask for the object in a zero-shot manner,
# And use FoundationPose to get the pose of the object in a zero-shot manner.

# Intended for testing and evaluating these zero-shot models.

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import ipdb
import trimesh
import pyrender
import argparse
import subprocess
import matplotlib.pyplot as plt
import cnos

from cnos.poses.pyrender import render as cnos_render
from cnos.utils.trimesh_utils import as_mesh
from cnos.scripts.inference_custom import run_inference as cnos_run_inference
from cnos.scripts.inference_custom import visualize as cnos_visualize

# FP imports
from foundationpose.estimater import *
from foundationpose.datareader import *


def realsense_capture():
    pipeline = rs.pipeline()
    config = rs.config()
    cfg = pipeline.start(config)

    # Reset devices
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev.hardware_reset()

    # config = rs.config()

    profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
    print("Camera intrinsics: ", intr)

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return depth_image, color_image

def get_cnos_data(mesh: trimesh.Trimesh):
    radius = 0.4
    lighting_intensity = 1.0
    pose_path = "/home/shruthi/Documents/Code/cnos/src/cnos/poses/predefined_poses/obj_poses_level0.npy"
    poses = np.load(pose_path)
    poses[:, :3, 3] = poses[:, :3, 3] / 1000.0
    poses[:, :3, 3] = poses[:, :3, 3] * radius
    intrinsics = np.array(
        [[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]
    )

    # load mesh to scale
    

    # for skateboard_base, axle
    # mesh.apply_scale(0.1)

    # for ondrive, strut
    mesh.apply_scale(0.01)

    # for skateboard hanger
    # mesh.apply_scale(0.001)

    pyrender_mesh = pyrender.Mesh.from_trimesh(as_mesh(mesh), smooth=False)
    return pyrender_mesh, mesh, poses, intrinsics, lighting_intensity

def run_foundation_pose(mesh, scene_dir):
    set_logging_format()
    set_seed(0)

    debug = 3
    debug_dir = "debug"
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    reader = YcbineoatReader(
        video_dir=scene_dir,
        shorter_side=None,
        zfar=np.inf
    )

    for i in range(len(reader.color_files)):
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        if i==0:
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5)

            if debug>=3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth>=0.1
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=2)

        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

        if debug>=1:
            center_pose = pose@np.linalg.inv(to_origin)

            # Write code to visualize contour of CAD model and not bounding box
            pyrender_mesh = pyrender.Mesh.from_trimesh(as_mesh(mesh), smooth=False)
            cad_renders = cnos_render(
                output_dir='debug',
                mesh=pyrender_mesh,
                obj_poses=np.expand_dims(pose, axis=0),
                intrinsic=reader.K,
                img_size=(480, 640),
                light_itensity=1.0
            )
            # cad_renders[0].show()
            composite = cv2.addWeighted(color, 0.5, np.array(cad_renders[0])[:, :, :3], 1.0, 0)
            cv2.imwrite("debug/composite.png", composite)
            cv2.imshow("composite", composite)
            cv2.waitKey(50000)

            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imwrite('debug/result.png', vis[...,::-1])
            # cv2.imshow('1', vis[...,::-1])
            # cv2.waitKey(1)

        if debug>=2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", 
        "--mode", 
        default="single", 
        choices=["single", "multi"], 
        help="Pose estimation in single frame or pose estimation + tracking in multiple frames"
    )

    parser.add_argument(
        "--cad-path",
        default = "/home/shruthi/Documents/Code/cnos/models/ondrive.ply",
        help = "Path to the CAD model that the camera is seeing"
    )

    parser.add_argument(
        "--exp-name",
        default = "ondrive",
        help = "Name of the experiment being run (which will be the output directory name)"
    )

    args = parser.parse_args()

    # Capture RGB and depth image
    depth, rgb = realsense_capture()

    # Load and pre-process some data for CNOS
    mesh = trimesh.load_mesh(args.cad_path)
    pyrender_mesh, mesh, poses, intrinsics, intensity = get_cnos_data(mesh)

    # Render the template from CAD models
    output_dir = "experiments/" + args.exp_name + "/renders"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + "/cnos_results", exist_ok=True)
    cnos_render(
        output_dir=output_dir,
        mesh=pyrender_mesh,
        obj_poses=poses,
        intrinsic=intrinsics,
        img_size=(480, 640),
        light_itensity=intensity
    )

    # Save RGB image, Depth Image and CAD model into the directory structure that foundationpose wants
    rgb_path = "experiments/" + args.exp_name + "/rgb"
    depth_path = "experiments/" + args.exp_name + "/depth"
    mesh_path = "experiments/" + args.exp_name + "/mesh"
    mask_path = "experiments/" + args.exp_name + "/masks"
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(mesh_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)

    cv2.imwrite(rgb_path + "/0.png", rgb)
    cv2.imwrite(depth_path + "/0.png", depth)
    # byte_array = trimesh.exchange.ply.export_ply(
    #     mesh, 
    #     encoding='binary', 
    #     include_attributes=True
    # )
    # output_file = open(mesh_path + "/skateboard_hanger.ply", "wb+")
    # output_file.write(byte_array)
    # output_file.close()
    np.savetxt("experiments/" + args.exp_name + "/cam_K.txt", intrinsics)

    # Run CNOS to get Segmentation mask
    result_image, results = cnos_run_inference(
        template_dir=output_dir,
        rgb_path=rgb_path + "/0.png",
        num_max_dets=1,
        stability_score_thresh=0.2,
        conf_threshold=0.2
    )

    # Save mask
    cv2.imwrite(mask_path + "/0.png", results["segmentation"][0]*255.)

    # FoundationPose
    run_foundation_pose(mesh, scene_dir="experiments/" + args.exp_name)