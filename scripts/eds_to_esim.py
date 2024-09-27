import os
import sys
import argparse
import yaml
import json

import easydict
import h5py
import numpy as np
import torch
import cv2

# insert the project / script parent directory into the module search path
PROJECT_DIR = os.path.join(sys.path[0], '..')
sys.path.insert(1, PROJECT_DIR)
import deblur_e_nerf as den


T_CCOMMON_COPENGL = np.array([[1,  0,  0, 0],
                              [0, -1,  0, 0],
                              [0,  0, -1, 0],
                              [0,  0,  0, 1]], dtype=np.float32)
S_TO_NS = int(1e+9)
MS_TO_NS = int(1e+6)
US_TO_NS = int(1e+3)

MV_TO_V = 1e-3

# calibration constants
CALIBRATION_CONFIG_FILENAME = (
    "camchain-mediajaviJAVISdatasetshwdscalibratione2kalibr.yaml"
)
RGB_CAMERA_ID = "cam0"
EVENT_CAMERA_ID = "cam1"

# raw dataset file/folder names
RAW_EVENTS_FILENAME = "events.h5"
RAW_EVENT_CAMERA_POSES_FILENAME = "stamped_groundtruth.txt"
DISTORTED_IMAGES_FOLDER_NAME = "images"
IMAGE_TIMESTAMPS_EXPOSURES_GAINS_FILENAME = "times.txt"

# preprocessed dataset file/folder names
PREPROCESSED_EVENTS_FILENAME = "raw_events.npz"
PREPROCESSED_EVENT_CAMERA_POSES_FILENAME = "camera_poses.npz"
PREPROCESSED_EVENT_CAMERA_CALIBRATION_FILENAME = "camera_calibration.npz"
POSED_UNDISTORTED_IMAGES_FOLDER_NAME = "views"
STAGE = "train"
STAGE_TRANSFORMS_FILENAME_FORMAT_STR = "transforms_{}.json"

# event camera biases
"""
NOTE:
    Biases for the Prophesee Gen 3.1 sensor (PPS3MVCD) are expressed in mV.

    References:
        1. https://docs.prophesee.ai/stable/hw/manuals/biases.html
        2. https://docs.prophesee.ai/stable/hw/sensors/gen31.html
"""
BIAS_DIFF_OFF = 194
BIAS_DIFF_ON = 414
BIAS_DIFF = 300
BIAS_FO = 1480
BIAS_PR = 1250
BIAS_REFR = 1500
BIAS_HPF = 1500

# assumed event camera parameters
ASSUMED_NEGATIVE_CONTRAST_THRESHOLD = 0.25

# DVS128 fast biases
ASSUMED_INPUT_TIME_CONST_EFF_IT_PROD = (35e-12 * 25e-3) / 2000e-12
ASSUMED_MILLER_TIME_CONST_EFF_IT_PROD = (0.6e-12 * 25e-3) / 2000e-12
ASSUMED_BLACK_LEVEL = 4e-12 / 2000e-12
ASSUMED_AMPLIFIER_GAIN = 140
ASSUMED_CLOSED_LOOP_GAIN = 1 / 0.7
ASSUMED_OUTPUT_TIME_CONST = 25e-6
ASSUMED_LOWER_CUTOFF_FREQ = 0.01
ASSUMED_SF_CUTOFF_FREQ = 16400
ASSUMED_DIFF_AMP_CUTOFF_FREQ = 82000

NULL_BAYER_PATTERN = ""     # ie. monochrome camera


def main(args):
    # create the preprocessed dataset directory, if necessary
    os.makedirs(args.preprocessed_dataset_path, exist_ok=True)

    # load the EDS calibration results from the config file
    eds_calibration_path = os.path.join(
        args.calibration_path, CALIBRATION_CONFIG_FILENAME
    )
    with open(eds_calibration_path) as f:
        eds_calibration = easydict.EasyDict(yaml.full_load(f))
    rgb_calibration = eds_calibration[RGB_CAMERA_ID]
    event_calibration = eds_calibration[EVENT_CAMERA_ID]

    # derive & save the event camera calibration parameters into an npz file
    assert (event_calibration.camera_model == "pinhole")
    preprocessed_event_calibration_path = os.path.join(
        args.preprocessed_dataset_path,
        PREPROCESSED_EVENT_CAMERA_CALIBRATION_FILENAME
    )
    event_intrinsics = event_calibration.intrinsics                             # (4)
    event_intrinsics = np.array(                                                # (3, 3)
        [[ event_intrinsics[0], 0,                   event_intrinsics[2] ],
         [ 0,                   event_intrinsics[1], event_intrinsics[3] ], 
         [ 0,                   0,                   1                   ]],
        dtype=np.float32
    )
    event_distortion_params = np.array(
        event_calibration.distortion_coeffs, dtype=np.float32
    )
    event_distortion_model = np.array({
        "radtan": "plumb_bob",
        "equi": "equidistant",
        "fov": "fov",
        "none": "plumb_bob"
    }[event_calibration.distortion_model])
    event_img_width, event_img_height = event_calibration.resolution
    event_img_width = np.array(event_img_width, dtype=np.uint16)
    event_img_height = np.array(event_img_height, dtype=np.uint16)

    positive_to_negative_contrast_threshold_ratio = np.array(
        (BIAS_DIFF_ON - BIAS_DIFF) / (BIAS_DIFF - BIAS_DIFF_OFF),
        dtype=np.float32
    )
    neg_contrast_threshold = np.array(
        ASSUMED_NEGATIVE_CONTRAST_THRESHOLD, dtype=np.float32
    )
    pos_contrast_threshold = positive_to_negative_contrast_threshold_ratio \
                             * neg_contrast_threshold
    refractory_period = np.array(bias_refr_voltage_to_ns(BIAS_REFR * MV_TO_V),
                                 dtype=np.float32)
    bayer_pattern = NULL_BAYER_PATTERN

    # `tau_in * it_eff = C_p * V_T / I_p_to_it_ratio`
    input_time_const_eff_it_prod = np.array(
        ASSUMED_INPUT_TIME_CONST_EFF_IT_PROD, dtype=np.float32
    )
    # `tau_mil * it_eff = C_mil * V_T / I_p_to_it_ratio`
    miller_time_const_eff_it_prod = np.array(
        ASSUMED_MILLER_TIME_CONST_EFF_IT_PROD, dtype=np.float32
    )
    # `black_level = I_dark / I_p_to_it_ratio`
    black_level = np.array([ ASSUMED_BLACK_LEVEL ], dtype=np.float32)
    amplifier_gain = np.array(ASSUMED_AMPLIFIER_GAIN, dtype=np.float32)
    # `A_cl = 1 / kappa`
    closed_loop_gain = np.array(ASSUMED_CLOSED_LOOP_GAIN, dtype=np.float32)
    output_time_const = np.array(ASSUMED_OUTPUT_TIME_CONST, dtype=np.float32)
    lower_cutoff_freq = np.array(ASSUMED_LOWER_CUTOFF_FREQ, dtype=np.float32)
    sf_cutoff_freq = np.array(ASSUMED_SF_CUTOFF_FREQ, dtype=np.float32)
    diff_amp_cutoff_freq = np.array(ASSUMED_DIFF_AMP_CUTOFF_FREQ,
                                    dtype=np.float32)

    np.savez(
        preprocessed_event_calibration_path,
        intrinsics=event_intrinsics,
        distortion_params=event_distortion_params,
        distortion_model=event_distortion_model,
        img_height=event_img_height,
        img_width=event_img_width,
        pos_contrast_threshold=pos_contrast_threshold,
        neg_contrast_threshold=neg_contrast_threshold,
        refractory_period=refractory_period,
        bayer_pattern=bayer_pattern,
        input_time_const_eff_it_prod=input_time_const_eff_it_prod,
        miller_time_const_eff_it_prod=miller_time_const_eff_it_prod,
        black_level=black_level,
        amplifier_gain=amplifier_gain,
        closed_loop_gain=closed_loop_gain,
        output_time_const=output_time_const,
        lower_cutoff_freq=lower_cutoff_freq,
        sf_cutoff_freq=sf_cutoff_freq,
        diff_amp_cutoff_freq=diff_amp_cutoff_freq
    )

    # convert & save event camera poses into an npz file
    raw_event_poses_path = os.path.join(
        args.raw_dataset_path, RAW_EVENT_CAMERA_POSES_FILENAME
    )
    preprocessed_event_poses_path = os.path.join(
        args.preprocessed_dataset_path,
        PREPROCESSED_EVENT_CAMERA_POSES_FILENAME
    )
    raw_event_poses = np.loadtxt(raw_event_poses_path)                          # (P, 8)

    T_wc_timestamp = S_TO_NS * raw_event_poses[:, 0]                            # (P)
    T_wc_timestamp = T_wc_timestamp.astype(np.int64)
    T_wc_position = raw_event_poses[:, 1:4].astype(np.float32)                  # (P, 3)
    T_wc_orientation = raw_event_poses[:, 4:8].astype(np.float32)               # (P, 4)

    # trim the sequence to the interval-of-interest
    is_valid_timestamp = (args.start_timestamp <= T_wc_timestamp) \
                         & (T_wc_timestamp < args.end_timestamp)                # (P)
    T_wc_timestamp = T_wc_timestamp[is_valid_timestamp]                         # (P')
    init_T_wc_timestamp = T_wc_timestamp[0]
    T_wc_timestamp = T_wc_timestamp - init_T_wc_timestamp

    T_wc_position = T_wc_position[is_valid_timestamp, :]                        # (P', 3)
    T_wc_orientation = T_wc_orientation[is_valid_timestamp, :]                  # (P', 4)

    np.savez(
        preprocessed_event_poses_path,
        T_wc_position=T_wc_position,
        T_wc_orientation=T_wc_orientation,
        T_wc_timestamp=T_wc_timestamp
    )

    # convert & save events into an npz file
    raw_events_path = os.path.join(
        args.raw_dataset_path, RAW_EVENTS_FILENAME
    )
    preprocessed_events_path = os.path.join(
        args.preprocessed_dataset_path, PREPROCESSED_EVENTS_FILENAME
    )
    with h5py.File(raw_events_path, "r") as f:
        event_position = np.stack((f['x'], f['y']), axis=1)                     # (N, 2)
        event_timestamp = US_TO_NS * np.array(f['t'])                           # (N)
        event_timestamp = event_timestamp - init_T_wc_timestamp
        event_polarity = np.array(f['p'], dtype=bool)                           # (N)

    # filter out events that occur outside of the T_wc relative pose timestamps
    event_position, event_timestamp, event_polarity = filter_event(
        event_position, event_timestamp, event_polarity, T_wc_timestamp
    )
    np.savez(
        preprocessed_events_path,
        position=event_position,
        timestamp=event_timestamp,
        polarity=event_polarity,
    )

    # derive & save RGB cam intrinsics, poses, exposure time & gain into a json
    assert (rgb_calibration.camera_model == "pinhole")
    assert rgb_calibration.distortion_model in ( "radtan", "none" )

    rgb_intrinsics = rgb_calibration.intrinsics                                 # (4)
    rgb_intrinsics = np.array(                                                  # (3, 3)
        [[ rgb_intrinsics[0], 0,                 rgb_intrinsics[2] ],
         [ 0,                 rgb_intrinsics[1], rgb_intrinsics[3] ], 
         [ 0,                 0,                 1                 ]],
        dtype=np.float32
    )
    rgb_distortion_params = np.array(rgb_calibration.distortion_coeffs,
                                     dtype=np.float32)
    rgb_img_width, rgb_img_height = rgb_calibration.resolution
    new_rgb_intrinsics, rgb_roi = cv2.getOptimalNewCameraMatrix(                # (3, 3), (4)
        rgb_intrinsics, rgb_distortion_params,
        (rgb_img_width, rgb_img_height), alpha=0
    )
    assert (rgb_roi == (0, 0, rgb_img_width - 1, rgb_img_height - 1))

    # linearly interpolate event camera poses at the image timestamps
    image_timestamps_exposures_gain_path = os.path.join(
        args.raw_dataset_path, IMAGE_TIMESTAMPS_EXPOSURES_GAINS_FILENAME
    )
    image_timestamp = np.loadtxt(                                               # (I)
        image_timestamps_exposures_gain_path, usecols=1
    )
    image_timestamp = S_TO_NS * image_timestamp                                 # (I)
    image_timestamp = image_timestamp.astype(np.int64)
    image_timestamp = image_timestamp - init_T_wc_timestamp

    is_valid_image = (0 <= image_timestamp) \
                     & (image_timestamp <= T_wc_timestamp[-1])                  # [I]
    image_timestamp = image_timestamp[is_valid_image]                           # [I']

    event_trajectory = den.models.trajectories.LinearTrajectory(
        easydict.EasyDict({
            "camera_poses": {
                "T_wc_position": torch.from_numpy(T_wc_position),
                "T_wc_orientation": torch.from_numpy(T_wc_orientation),
                "T_wc_timestamp": torch.from_numpy(T_wc_timestamp)
            }
        })
    )
    with torch.no_grad():
        T_w_event_position, T_w_event_orientation = event_trajectory(           # (I', 3), (I', 3, 3)
            torch.from_numpy(image_timestamp)
        )

    # derive the RGB camera poses from the event camera poses
    T_w_event = np.zeros((len(T_w_event_position), 4, 4), dtype=np.float32)     # (I', 4, 4)
    T_w_event[:, :3, 3] = T_w_event_position.numpy()
    T_w_event[:, :3, :3] = T_w_event_orientation.numpy()
    T_w_event[:, 3, 3] = 1
    T_event_rgb = np.array(                                                     # (4, 4)
        event_calibration.T_cn_cnm1, dtype=np.float32
    )
    T_w_rgb = T_w_event @ T_event_rgb                                           # (I', 4, 4)

    # convert the RGB camera poses from a common to the OpenGL convention
    T_w_rgb = T_w_rgb @ T_CCOMMON_COPENGL                                       # (I', 4, 4)

    image_exposure = np.loadtxt(                                                # (I)
        image_timestamps_exposures_gain_path, usecols=2
    )
    image_exposure = MS_TO_NS * image_exposure                                  # (I)
    image_exposure = image_exposure.astype(np.int64)
    image_exposure = image_exposure[is_valid_image]                             # [I']

    image_gain = np.loadtxt(image_timestamps_exposures_gain_path, usecols=3)    # (I)
    image_gain = db_to_linear(image_gain)                                       # (I)
    image_gain = image_gain.astype(np.float32)
    image_gain = image_gain[is_valid_image]                                     # [I']

    posed_undistorted_images_path = os.path.join(
        args.preprocessed_dataset_path, POSED_UNDISTORTED_IMAGES_FOLDER_NAME
    )
    transforms_path = os.path.join(
        posed_undistorted_images_path,
        STAGE_TRANSFORMS_FILENAME_FORMAT_STR.format(STAGE)
    )
    image_filename = np.loadtxt(                                                # (I)
        image_timestamps_exposures_gain_path, dtype=str, usecols=4
    )
    image_filename = image_filename[is_valid_image]                             # (I')
    transforms = {
        "intrinsics": new_rgb_intrinsics.tolist(),
        "frames": [ { "file_path": os.path.join(
                                    ".", STAGE, os.path.splitext(filename)[0]
                                   ),
                      "exposure_time": exposure_time.item(),
                      "gain": gain.item(),
                      "transform_matrix": tf_matrix.tolist() }
                    for filename, exposure_time, gain, tf_matrix
                    in zip(image_filename, image_exposure,
                           image_gain, T_w_rgb) ]
    }
    os.mkdir(posed_undistorted_images_path)
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=4)

    # undistort the RGB images & save them
    distorted_images_path = os.path.join(
        args.raw_dataset_path, DISTORTED_IMAGES_FOLDER_NAME
    )
    stage_undistorted_images_path = os.path.join(
        posed_undistorted_images_path, STAGE
    )
    os.mkdir(stage_undistorted_images_path)
    for filename in image_filename:
        distorted_image_path = os.path.join(distorted_images_path, filename)
        distorted_image = cv2.imread(
            distorted_image_path, cv2.IMREAD_UNCHANGED
        )
        undistorted_image = cv2.undistort(
            distorted_image, rgb_intrinsics, rgb_distortion_params,
            newCameraMatrix=new_rgb_intrinsics
        )
        undistorted_image_path = os.path.join(
            stage_undistorted_images_path, filename
        )
        cv2.imwrite(undistorted_image_path, undistorted_image)


def bias_refr_voltage_to_ns(voltage):
    refractory_period_s = 4e-23 * np.exp(27.64 * voltage)
    refractory_period_ns = S_TO_NS * refractory_period_s
    return refractory_period_ns


def filter_event(
    event_position,
    event_timestamp,
    event_polarity,
    T_wc_timestamp
):
    valid_indices = (T_wc_timestamp[0] <= event_timestamp) \
                    & (event_timestamp <= T_wc_timestamp[-1])
    event_position = event_position[valid_indices, :].copy(order="C")
    event_timestamp = event_timestamp[valid_indices].copy(order="C")
    event_polarity = event_polarity[valid_indices].copy(order="C")

    return event_position, event_timestamp, event_polarity


def db_to_linear(db_values):
    return 10 ** (db_values / 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Script for converting EDS datasets to"
                     " pre-processed ESIM format.")
    )
    parser.add_argument(
        "calibration_path", type=str,
        help=("Path to the EDS calibration results folder"
              " (ie. `01_calib_results`).")
    )
    parser.add_argument(
        "raw_dataset_path", type=str, help="Path to the raw EDS dataset."
    )
    parser.add_argument(
        "preprocessed_dataset_path", type=str,
        help="Desired path to the pre-processed EDS dataset."
    )
    parser.add_argument(
        "--start_timestamp", type=int, default=0,
        help="Trim the sequence to start at the given timestamp (inclusive)."
    )
    parser.add_argument(
        "--end_timestamp", type=int, default=float("inf"),
        help="Trim the sequence to end at the given timestamp (exclusive)."
    )
    args = parser.parse_args()

    main(args)
