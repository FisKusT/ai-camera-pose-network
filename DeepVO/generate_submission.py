import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from DeepVO.data_helper import ImageSequenceDataset
from DeepVO.params import par

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", help="path to test set")
args = parser.parse_args()

# get df format for submission
submission = pd.read_csv("submission_format.csv")

test_images = sorted(os.listdir(args.datapath))
assert submission['Filename'].isin(test_images).all(), "Not all submission images are in test set"
assert all(
    [file in submission['Filename'].values for file in test_images]), "Not all test images are in submission format"

# read test set
test_df = pd.DataFrame(columns=['seq_len', 'image_path'])
seq_len = par.seq_len[0]
for i in range(len(test_images) - seq_len + 1):
    test_df.loc[i] = [seq_len, [os.path.join(args.datapath, test_images[i + j]) for j in range(seq_len)]]

# create test loader
test_dataset = ImageSequenceDataset(test_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds,
                                    par.minus_point_5, test_mode=True)
test_dl = DataLoader(test_dataset, batch_size=1, num_workers=par.n_processors, pin_memory=par.pin_mem, shuffle=False)

# run model
model = torch.load(
    '/cortex/users/hilita/CameraPose/DeepVO/models/t1234_v_im184x608_s7x7_b16_rnn1000_optAdam_lr0.1.model_ep150.pth')

results = pd.DataFrame(columns=['Filename', 'Easting', 'Northing', 'Height', 'Roll', 'Pitch', 'Yaw'])
for i, (seq_len, test_seq, pose_seq) in enumerate(test_dl):
    preds = model(test_seq.cuda()).cpu().detach().numpy()

    for j in range(preds.shape[1]):
        results.loc[i * int(seq_len) + j, :] = [test_df.loc[i]['image_path'][j].split("/")[-1]] + list(preds[0, j])

    a = 1

# edit submission df
final_df = submission[['Filename', 'TrajectoryId', 'Timestamp']].merge(results.groupby('Filename').mean(), how='left',
                                                                       left_on='Filename', right_index=True)

# TODO: handle last row
final_df = final_df.fillna(method='ffill')


# try to add intrinsic params
def euler_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def create_projection_matrix(row, intrinsic_matrix):
    roll, pitch, yaw = np.deg2rad(row[["Roll", "Pitch", "Yaw"]].astype(float).values)  # Convert to radians
    R = euler_to_rotation_matrix(roll, pitch, yaw)  # Convert to rotation matrix
    T = np.array([row["Easting"], row["Northing"], row["Height"]]).reshape(3, 1)  # Translation vector
    RT = np.hstack((R, T))  # [R | T]
    P = np.dot(intrinsic_matrix, RT)  # P = K[R | T]
    return P


def rotation_matrix_to_euler_angles(R):
    import math
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.rad2deg(np.array([x, y, z]))  # Convert to degrees


def decompose_projection_matrix(P):
    M = P[:, :3]  # 3x3 matrix from P
    T = P[:, 3]  # translation vector from P

    # SVD to get rotation matrix
    U, S, Vt = np.linalg.svd(M)
    R = np.dot(U, Vt)

    # Ensure correct handedness
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(U, Vt)

    # Normalize T
    T = T / S[0]

    roll, pitch, yaw = rotation_matrix_to_euler_angles(R)  # Convert rotation matrix to Euler angles
    easting, northing, height = T[0], T[1], T[2]
    return pd.Series([easting, northing, height, roll, pitch, yaw])


# Intrinsic camera parameters
fx = 935.6461822571149
fy = 935.7779926708049
cx = 1501.8278990534407
cy = 1016.1713538034546

# Create the intrinsic matrix
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Assuming df is your DataFrame
# Create projection matrix for each row
final_df['ProjectionMatrix'] = final_df.apply(lambda row: create_projection_matrix(row, K), axis=1)

# Decompose projection matrix back to the rotation and translation parameters
final_df[['Easting', 'Northing', 'Height', 'Roll', 'Pitch', 'Yaw']] = final_df['ProjectionMatrix'].apply(
    decompose_projection_matrix)

# save submission df
final_df.to_csv('submission.csv', index=False)
