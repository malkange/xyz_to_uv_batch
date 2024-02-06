import os
import pickle

from pyquaternion import Quaternion
from PIL import Image
from torchvision.transforms import transforms
from nuscenes import NuScenes
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import *

# nuscene initialization
version = 'v1.0-mini'                                                               ### fix ###
data_path = 'D:/Dataset/nuscenes/v1.0-mini'                                         ### fix ###
nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

# hz 
open_asynchronous_compensation = True

# image & camera setting
IMAGE_SIZE = (900, 1600)
CAM_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# pickle -> nusc_infos(sample's information)
sample_pkl_path = 'D:/Dataset/nuscenes/lcps_pkl_files/nuscenes_pkl'                 ### fix ###
imageset = os.path.join(sample_pkl_path, "nuscenes_infos_train_mini.pkl")           ### fix ###
with open(imageset, 'rb') as f:
    data = pickle.load(f)
nusc_infos = data['infos']


# nuscenes_infos_val_mini.pkl : 81 samples(len(nusc_infos))
# nuscenes_infos_train_mini.pkl : 323 samples(len(nusc_infos))
for i in range(len(nusc_infos)):
    # create directory
    dir_name = f"save/data/sample_{i}"                                               ### fix ###
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    # pick one sample(anotated snapshot of a scene at a particular timestamp)
    # lenth = 14 -> See the info.txt file
    info = nusc_infos[i]
    """
    my_scene = nusc.scene[0]
    first_sample_token = my_scene['first_sample_token']
    nusc.render_sample(first_sample_token)
    exit()
    """
    
    # get point cloud from lidar top
    lidar_path = os.path.join(data_path,'samples','LIDAR_TOP', info['lidar_path'].split('/')[-1])
        # D:/Dataset/nuscenes/v1.0-mini\samples\LIDAR_TOP\n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin
    points = np.fromfile(os.path.join(data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        # count = -1 : all file
        # (173440,) -> (34688, 5) ... why 5 dim?
    points_ids = np.linspace(0, len(points)-1, len(points)).astype(int)

    # sample -> LIDAR_TOP token -> Lidar meta data(data collected from a lidar)
    lidar_sd_token = nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        # nusc.get('sample', info['token']) : sample(1 frame scene)'s meta data
        # ['data' ] : sample data(data collected from a particcular censor)
        # ['LIDAR_TOP']
    lidar_channel = nusc.get("sample_data", lidar_sd_token)

    # initialization before projection calculation
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ),
            ])
    imgs_per_sample = np.empty((0, 900, 1600, 3))                   # channel num, 900, 1600, 3
    points_ids_per_sample = np.empty((0,1))                         # points num, 1
    points_uvs_per_sample = np.empty((0,2))                         # points num, 2

    # projection calculation
    for idx, channel in enumerate(CAM_CHANNELS):
        # sample_data -> filename -> image
        cam_token = info['cams'][channel]['sample_data_token']
        cam_channel = nusc.get('sample_data', cam_token)
        #img = Image.open(os.path.join(nusc.dataroot, cam_channel['filename'])).convert('RGB')
        img = Image.open(os.path.join(data_path, cam_channel['filename'])).convert('RGB')
            # nusc.dataroot : D:/Dataset/nuscenes/v1.0-mini
            # .convert('RGB') : not necessary
        
        # image per channel -> image per sample
        imgs_per_channel = np.array(transform(img)).transpose(1, 2, 0).astype('float32')
        imgs_per_channel = np.expand_dims(imgs_per_channel, axis = 0)
        imgs_per_sample = np.vstack((imgs_per_sample, imgs_per_channel))
        
        ### Points live in the point sensor frame. So they need to be transformed via global to the image plane. ###
        # PCD Transform object
        pcd_trans_tool = PCDTransformTool(points[:, :3])
        
        # First step : transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        # lidar position -> ego position
        cs_record = nusc.get('calibrated_sensor', lidar_channel['calibrated_sensor_token'])
        pcd_trans_tool.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pcd_trans_tool.translate(np.array(cs_record['translation']))


        # todo add codes for asynchronous compensation after testing the code
        if open_asynchronous_compensation:
            # Second step: transform from ego to the global frame at timestamp of the first frame in the sequence pack.
            poserecord = nusc.get('ego_pose', lidar_channel['ego_pose_token'])
            pcd_trans_tool.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
            pcd_trans_tool.translate(np.array(poserecord['translation']))

            # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
            poserecord = nusc.get('ego_pose', cam_channel['ego_pose_token'])
            pcd_trans_tool.translate(-np.array(poserecord['translation']))
            pcd_trans_tool.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        
        # Fourth step: transform from ego into the camera.
        # ego position -> camera position
        cs_record = nusc.get('calibrated_sensor', cam_channel['calibrated_sensor_token'])
        pcd_trans_tool.translate(-np.array(cs_record['translation']))
        pcd_trans_tool.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
        # remove points based on z-axis ... reference paper
        mask = np.ones(points.shape[0], dtype=bool)
        mask = np.logical_and(mask, pcd_trans_tool.pcd[2, :] > 1)               
        
        # Fifth step: project from 3d coordinate to 2d coordinate
        # projection
        pcd_trans_tool.pcd2image(np.array(cs_record['camera_intrinsic']))
            # intrinsic camera calibration matrix
            # [[1266, 0.0, 816]
            # [0.0, 1266, 491]
            # [0.0, 0.0, 1.0]]
            # output shape : (3, 34688)
        pixel_coord = pcd_trans_tool.pcd[:2, :]
        pixel_coord_copy = pixel_coord.copy()
            # output shape : (2, 34688)
            
        # remove off-image points
        pixel_coord[0, :] = pixel_coord[0, :] / (img.size[0] - 1.0) * 2.0 - 1.0  # width
        pixel_coord[1, :] = pixel_coord[1, :] / (img.size[1] - 1.0) * 2.0 - 1.0  # height
        mask = np.logical_and(mask, pixel_coord[0, :] > -1)
        mask = np.logical_and(mask, pixel_coord[0, :] < 1)
        mask = np.logical_and(mask, pixel_coord[1, :] > -1)
        mask = np.logical_and(mask, pixel_coord[1, :] < 1)

        # uv and id
        points_ids_per_channel = points_ids[mask] # (3056,)
        points_ids_per_channel = points_ids_per_channel.reshape((-1,1)) # (3056,1)
        points_uvs_per_channel = pixel_coord_copy.T[mask] # (3056, 2)

        # uv and id information concat (channel -> sample)
        points_ids_per_sample = np.concatenate((points_ids_per_sample, points_ids_per_channel), axis = 0)
        points_uvs_per_sample = np.concatenate((points_uvs_per_sample, points_uvs_per_channel), axis = 0)
        
        # check projection onto image
        plt.imshow(img)
        plt.savefig(dir_name + '/' + str(idx+1) + '_' + channel + '.png')                                          ### fix ###
        plt.scatter(points_uvs_per_channel[:,0], points_uvs_per_channel[:,1], marker='o', color="red", s=10)
        plt.savefig(dir_name + '/' + str(idx+1) + '_' + channel + '_points.png')                                   ### fix ###
        plt.clf()

        #save points per channel
        np.save(dir_name + '/' + str(idx+1) + '_' + channel + '_ids.npy', points_ids_per_channel)                  ### fix ###
        np.save(dir_name + '/' + str(idx+1) + '_' + channel + '_uvs.npy', points_uvs_per_channel)                  ### fix ###
            
    #save points per sample
    np.save(dir_name + '/' + 'sample_ids.npy', points_ids_per_sample)                                           ### fix ###
    np.save(dir_name + '/' + 'sample_uvs.npy', points_uvs_per_sample)                                           ### fix ###
    np.save(dir_name + '/' + 'sample_imgs.npy', points_uvs_per_sample)                                          ### fix ###
