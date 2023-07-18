import cv2
import json
import matplotlib.pyplot as plt
import os
import numpy as np


# load train images
DATA_DIR = '/home/nlp/ron.eliav/pose3d/data'

example_image = cv2.imread(os.path.join(DATA_DIR, 'train_images-1/180502_032924483_Camera_0.jpg'))

# load calibration data
with open('intrinsic_parameters.json', 'r') as f:
    intrinsic_parameters = json.load(f)

fx, fy, cx, cy = intrinsic_parameters['fx'], intrinsic_parameters['fy'], intrinsic_parameters['Cx'], \
                 intrinsic_parameters['Cy']

# Define the intrinsic camera matrix K
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# Assuming there's no lens distortion
distortion_coefficients = np.zeros((4,1))


# The function cv2.undistort() requires the inverse of camera matrix K.
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, distortion_coefficients, example_image.shape[1::-1], 1, example_image.shape[1::-1])

# Calibrate the image
calibrated_image = cv2.undistort(example_image, K, distortion_coefficients, None, new_camera_matrix)

# rotate example image
example_image = cv2.rotate(example_image, cv2.ROTATE_90_CLOCKWISE, example_image)
calibrated_image = cv2.rotate(calibrated_image, cv2.ROTATE_90_CLOCKWISE, calibrated_image)

# plot image and calibrated side by side
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(example_image)
ax2.imshow(calibrated_image)

plt.show()
