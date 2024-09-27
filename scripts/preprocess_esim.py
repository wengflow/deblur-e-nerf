import argparse
import os
import absl.flags
import tqdm
import rosbag
import numpy as np
import cv2

INTERM_COLOR_SPACE_ID_TO_NAME = {
    0: "display",
    1: "linear"
}
BAYER_PATTERN = "RGGB"
NULL_BAYER_PATTERN = ""     # ie. monochrome camera
FROM_MILLI = 1e-3
FROM_MICRO = 1e-6
FROM_FEMTO = 1e-9

TOPICS = [ "/cam0/events", "/cam0/pose", "/cam0/camera_info", "/imu" ]
EVENTS_FILENAME = "raw_events.npz"
CAMERA_POSES_FILENAME = "camera_poses.npz"
CAMERA_CALIBRATION_FILENAME = "camera_calibration.npz"
RENDERER_PARAMS_FILENAME = "renderer_params.npz"

# default ESIM configs
GAUSSIAN_BLUR_KSIZE = np.array([21, 21])
absl.flags.DEFINE_integer("renderer_type", 0,
                          ("0: Planar renderer, 1: Panorama renderer, "
                           "2: OpenGL renderer, 3: UnrealCV renderer"))
absl.flags.DEFINE_string("renderer_texture", "",
                         ("Path to image which will be used to texture the"
                          " plane"))
absl.flags.DEFINE_float("renderer_hfov_cam_source_deg", 130.0,
                        ("Horizontal FoV of the source camera (that captured"
                         " the image on the plane)"))
absl.flags.DEFINE_integer("renderer_preprocess_median_blur", 0,
                          "Kernel size of the preprocessing median blur.")
absl.flags.DEFINE_float("renderer_preprocess_gaussian_blur", 0,
                        "Amount of preprocessing Gaussian blur.")
absl.flags.DEFINE_float("renderer_plane_x", 0.0,
                        ("x position of the center of the plane, in world"
                         " coordinates"))
absl.flags.DEFINE_float("renderer_plane_y", 0.0,
                        ("y position of the center of the plane, in world"
                         " coordinates"))
absl.flags.DEFINE_float("renderer_plane_z", -1.0,
                        ("z position of the center of the plane, in world"
                         " coordinates"))
absl.flags.DEFINE_float("renderer_plane_qw", 0.0,
                        ("w component of the quaternion q_W_P (orientation of"
                         " the plane with respect to the world)"))
absl.flags.DEFINE_float("renderer_plane_qx", 1.0,
                        ("x component of the quaternion q_W_P (orientation of"
                         " the plane with respect to the world)"))
absl.flags.DEFINE_float("renderer_plane_qy", 0.0,
                        ("y component of the quaternion q_W_P (orientation of"
                         " the plane with respect to the world)"))
absl.flags.DEFINE_float("renderer_plane_qz", 0.0,
                        ("z component of the quaternion q_W_P (orientation of"
                         " the plane with respect to the world)"))

absl.flags.DEFINE_integer("blender_interm_color_space", 0,
                          ("Color space of the intermediate output RGBA image."
                           " 0: Display (Filmic sRGB by default), 1: Linear."))

absl.flags.DEFINE_float("contrast_threshold_pos", 1.0,
                        "Contrast threshold (positive)")
absl.flags.DEFINE_float("contrast_threshold_neg", 1.0,
                        "Contrast threshold  (negative)")
absl.flags.DEFINE_integer("refractory_period_ns", 0,
                          ("Refractory period (time during which a pixel"
                           " cannot fire events just after it fired one), in"
                           " nanoseconds"))

absl.flags.DEFINE_float("I_p_to_intensity_ratio_fa", float("inf"),
                        ("Ratio of the signal photocurrent `I_p`, in fA, to"
                         " image pixel intensity `it`, `I_p_to_it_ratio`"))
absl.flags.DEFINE_float("dark_current_fa", 0.0,
                        ("Photodiode dark current `I_dark`, in fA. The"
                         " photocurrent `I = I_p + I_dark`. When"
                         " `I_p_to_it_ratio` approaches infinity, then"
                         " `I_dark` is effectively 0 / dark current-equivalent"
                         " image pixel intensity (i.e. black level)"
                         " `black_level = I_dark / I_p_to_it_ratio` is 0."))
absl.flags.DEFINE_float("amplifier_gain", float("inf"),
                        "Amplifier gain of the photoreceptor circuit `A_amp`")
absl.flags.DEFINE_float("back_gate_coeff", 0.7,
                        ("Back-gate coefficient `kappa`. The closed-loop gain"
                         " of the photoreceptor circuit `A_cl = 1 / kappa`,"
                         " and the total loop gain of the photoreceptor"
                         " circuit `A_loop = A_amp / A_cl`."))
absl.flags.DEFINE_float("thermal_voltage_mv", 25,
                        "Thermal voltage `V_T`, in mV")
absl.flags.DEFINE_float("photodiode_cap_ff", 0.0,
                        ("(Lumped) Parasitic capacitance on the photodiode"
                         " / input node of the photoreceptor circuit `C_p`, in"
                         " fF. The time constant associated to the input node"
                         " of the photoreceptor circuit"
                         " `tau_in = C_p * V_T / I = Q_in / I`."))
absl.flags.DEFINE_float("miller_cap_ff", 0.0,
                        ("Miller capacitance in the photoreceptor circuit"
                         " `C_mil`, in fF. In the absence of a cascode"
                         " transistor, `C_mil = C_fb + C_n`, where `C_fb` is"
                         " the Miller capacitance from the gate to the source"
                         " of the feedback transistor M_fb, and `C_n` is the"
                         " Miller capacitance from the gate to the drain of"
                         " the inverting amplifier transistor M_n. Else,"
                         " `C_mil = C_fb`. The time constant associated to the"
                         " Miller capacitance `tau_mil = C_mil * V_T / I"
                         " = Q_mil / I`."))
absl.flags.DEFINE_float("output_time_const_us", 0.0,
                        ("Time constant `tau_out` associated to the output"
                         " node of the photoreceptor circuit / photoreceptor"
                         " bias current `I_pr`, in microseconds"))
absl.flags.DEFINE_float("lower_cutoff_freq_hz", 0.0,
                        ("Lower cutoff frequency of the pixel circuit / high"
                         "-pass filter present in certain event cameras"
                         " `f_c_lower`, in Hz"))
absl.flags.DEFINE_float("sf_cutoff_freq_hz", float("inf"),
                        ("(Upper) Cutoff frequency of the source follower"
                         " buffer `f_c_sf`, associated to the source follower"
                         " buffer bias current `I_sf`, in Hz"))
absl.flags.DEFINE_float("diff_amp_cutoff_freq_hz", float("inf"),
                        ("(Upper) Cutoff frequency of the differencing/change"
                         " amplifier `f_c_diff`, in Hz"))

absl.flags.DEFINE_float("log_eps", 0.001,
                        ("Epsilon value used to convert images"
                         " to log: L = log(eps + I / 255.0)."))
absl.flags.DEFINE_bool("simulate_color_events", False,
                       ("Whether to simulate color events or not"
                        " (default: false)"))

def main(args):
    pos_contrast_threshold, neg_contrast_threshold, \
    refractory_period, bayer_pattern, input_time_const_eff_it_prod, \
    miller_time_const_eff_it_prod, black_level, amplifier_gain, \
    closed_loop_gain, output_time_const, lower_cutoff_freq, sf_cutoff_freq, \
    diff_amp_cutoff_freq = preprocess_conf(
        args.conf_path, args.dataset_path
    )
    preprocess_rosbag(
        args.rosbag_path, args.dataset_path, pos_contrast_threshold,
        neg_contrast_threshold, refractory_period, bayer_pattern,
        input_time_const_eff_it_prod, miller_time_const_eff_it_prod,
        black_level, amplifier_gain, closed_loop_gain, output_time_const,
        lower_cutoff_freq, sf_cutoff_freq, diff_amp_cutoff_freq
    )


def preprocess_conf(conf_path, dataset_path):
    # parse conf file
    FLAGS = absl.flags.FLAGS
    FLAGS(argv=[ "", "--flagfile=" + conf_path ], known_only=True)

    # extract & save renderer parameters into an npz file
    renderer_params_path = os.path.join(dataset_path, RENDERER_PARAMS_FILENAME)
    if FLAGS.renderer_type == 0:    # Planar
        # derive the camera intrinsics used to virtually capture the planar img
        planar_img = cv2.imread(FLAGS.renderer_texture)
        planar_img_height, planar_img_width = planar_img.shape[:2]
        planar_img_f = (planar_img_width / 2) / np.tan(
            np.deg2rad(FLAGS.renderer_hfov_cam_source_deg / 2)
        )
        planar_intrinsics = np.array(
            [[ planar_img_f, 0,            planar_img_width / 2 ],
             [ 0,            planar_img_f, planar_img_height / 2], 
             [ 0,            0,            1                    ]],
            dtype=np.float32
        )

        np.savez(
            renderer_params_path,
            planar_img_filename=os.path.basename(FLAGS.renderer_texture),
            planar_intrinsics=planar_intrinsics,
            median_blur_ksize=np.array(FLAGS.renderer_preprocess_median_blur),
            gaussian_blur_ksize=GAUSSIAN_BLUR_KSIZE,
            gaussian_blur_sigma=np.array(
                                    FLAGS.renderer_preprocess_gaussian_blur
                                ),
            T_wp_position=np.array([ 
                              FLAGS.renderer_plane_x,
                              FLAGS.renderer_plane_y,
                              FLAGS.renderer_plane_z
                          ], dtype=np.float32),
            T_wp_orientation=np.array([ 
                                 FLAGS.renderer_plane_qx,
                                 FLAGS.renderer_plane_qy,
                                 FLAGS.renderer_plane_qz,
                                 FLAGS.renderer_plane_qw
                             ], dtype=np.float32)
        )
    elif FLAGS.renderer_type == 1:  # Panoramic
        raise NotImplementedError   # TODO
    elif FLAGS.renderer_type == 2:  # OpenGL
        raise NotImplementedError   # TODO
    elif FLAGS.renderer_type == 3:  # UnrealCV
        raise NotImplementedError   # TODO
    elif FLAGS.renderer_type == 4:  # Blender
        interm_color_space = (
            INTERM_COLOR_SPACE_ID_TO_NAME[FLAGS.blender_interm_color_space]
        )
        np.savez(
            renderer_params_path,
            interm_color_space=interm_color_space,
            log_eps=FLAGS.log_eps
        )
    else:
        raise NotImplementedError

    pos_contrast_threshold = np.array(
        FLAGS.contrast_threshold_pos, dtype=np.float32
    )
    neg_contrast_threshold = np.array(
        FLAGS.contrast_threshold_neg, dtype=np.float32
    )
    refractory_period = np.array(FLAGS.refractory_period_ns)
    if FLAGS.simulate_color_events:
        bayer_pattern = BAYER_PATTERN
        intensity_shape = 3
    else:
        bayer_pattern = NULL_BAYER_PATTERN
        intensity_shape = 1

    # `tau_in * it_eff = C_p * V_T / I_p_to_it_ratio`
    input_time_const_eff_it_prod = np.array(
        FLAGS.photodiode_cap_ff * (FROM_MILLI * FLAGS.thermal_voltage_mv)
        / FLAGS.I_p_to_intensity_ratio_fa,
        dtype=np.float32
    )
    # `tau_mil * it_eff = C_mil * V_T / I_p_to_it_ratio`
    miller_time_const_eff_it_prod = np.array(
        FLAGS.miller_cap_ff * (FROM_MILLI * FLAGS.thermal_voltage_mv)
        / FLAGS.I_p_to_intensity_ratio_fa,
        dtype=np.float32
    )
    black_level = np.full(
        intensity_shape,
        FLAGS.dark_current_fa / FLAGS.I_p_to_intensity_ratio_fa,
        dtype=np.float32
    )
    amplifier_gain = np.array(FLAGS.amplifier_gain, dtype=np.float32)
    closed_loop_gain = np.array(1 / FLAGS.back_gate_coeff, dtype=np.float32)
    output_time_const = np.array(FROM_MICRO * FLAGS.output_time_const_us,
                                 dtype=np.float32)
    lower_cutoff_freq = np.array(FLAGS.lower_cutoff_freq_hz, dtype=np.float32)
    sf_cutoff_freq = np.array(FLAGS.sf_cutoff_freq_hz, dtype=np.float32)
    diff_amp_cutoff_freq = np.array(FLAGS.diff_amp_cutoff_freq_hz,
                                    dtype=np.float32)

    return pos_contrast_threshold, neg_contrast_threshold, \
           refractory_period, bayer_pattern, input_time_const_eff_it_prod, \
           miller_time_const_eff_it_prod, black_level, amplifier_gain, \
           closed_loop_gain, output_time_const, lower_cutoff_freq, \
           sf_cutoff_freq, diff_amp_cutoff_freq


def preprocess_rosbag(
    rosbag_path,
    dataset_path,
    pos_contrast_threshold,
    neg_contrast_threshold,
    refractory_period,
    bayer_pattern,
    input_time_const_eff_it_prod,
    miller_time_const_eff_it_prod,
    black_level,
    amplifier_gain,
    closed_loop_gain,
    output_time_const,
    lower_cutoff_freq,
    sf_cutoff_freq,
    diff_amp_cutoff_freq
):
    # read rosbag
    bag = rosbag.Bag(rosbag_path)

    # initialize dataset
    event_position = []
    event_timestamp = []
    event_polarity = []
    img_height = None
    img_width = None
    T_wc_position = []
    T_wc_orientation = []
    T_wc_timestamp = []
    imu_timestamp = []
    intrinsics = None
    distortion_params = None
    distortion_model = None

    # Extract relevant data
    print("Extracting relevant data from the rosbag...")
    for topic, msg, t in tqdm.tqdm(
        bag.read_messages(topics=TOPICS),
        total=bag.get_message_count(topic_filters=TOPICS)
    ):
        if topic == TOPICS[0]:
            img_height, img_width = event_callback(
                msg, event_position, event_timestamp, event_polarity
            )
        elif topic == TOPICS[1]:
            pose_callback(msg, T_wc_position, T_wc_orientation, T_wc_timestamp)
        elif topic == TOPICS[2]:
            intrinsics, distortion_params, distortion_model = (
                camera_info_callback(msg)
            )
        elif topic == TOPICS[3]:
            imu_callback(msg, imu_timestamp)
        else:
            raise NotImplementedError

    # cast dataset to numpy arrays
    event_position = np.array(event_position, dtype=np.uint16)
    event_timestamp = np.array(event_timestamp)
    event_polarity = np.array(event_polarity)
    T_wc_position = np.array(T_wc_position, dtype=np.float32)
    T_wc_orientation = np.array(T_wc_orientation, dtype=np.float32)
    T_wc_timestamp = np.array(T_wc_timestamp)
    imu_timestamp = np.array(imu_timestamp)

    # filter out non-IMU-synced T_wc relative poses
    print("Filtering out non-IMU-synced T_wc relative poses...")
    T_wc_position, T_wc_orientation, T_wc_timestamp = filter_T_wc(
        T_wc_position, T_wc_orientation, T_wc_timestamp, imu_timestamp
    )

    # filter out events that occur outside of the T_wc relative pose timestamps
    event_position, event_timestamp, event_polarity = filter_event(
        event_position, event_timestamp, event_polarity, T_wc_timestamp
    )

    # save dataset to npz file
    print("Saving dataset...")
    events_path = os.path.join(dataset_path, EVENTS_FILENAME)
    camera_poses_path = os.path.join(dataset_path, CAMERA_POSES_FILENAME)
    camera_calibration_path = os.path.join(
        dataset_path, CAMERA_CALIBRATION_FILENAME
    )
    np.savez(
        events_path,
        position=event_position,
        timestamp=event_timestamp,
        polarity=event_polarity,
    )
    np.savez(
        camera_poses_path,
        T_wc_position=T_wc_position,
        T_wc_orientation=T_wc_orientation,
        T_wc_timestamp=T_wc_timestamp
    )
    np.savez(
        camera_calibration_path,
        intrinsics=intrinsics,
        distortion_params=distortion_params,
        distortion_model=distortion_model,
        img_height=img_height,
        img_width=img_width,
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
    print("Done!")
    

def event_callback(msg, event_position, event_timestamp, event_polarity):
    for event in msg.events:
        event_position.append((event.x, event.y))
        event_timestamp.append(event.ts.to_nsec())
        event_polarity.append(event.polarity)
    img_height = np.array(msg.height, dtype=np.uint16)
    img_width = np.array(msg.width, dtype=np.uint16)
    return img_height, img_width


def pose_callback(msg, T_wc_position, T_wc_orientation, T_wc_timestamp):
    T_wc_position.append(
        (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
    )
    T_wc_orientation.append(
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w)
    )
    T_wc_timestamp.append(msg.header.stamp.to_nsec())


def camera_info_callback(msg):
    intrinsics = np.array(msg.K, dtype=np.float32)          # (9)
    intrinsics = intrinsics.reshape(3, 3).copy(order="C")   # (3, 3)
    distortion_params = np.array(msg.D, dtype=np.float32)   # (4) or ()
    distortion_model = np.array(msg.distortion_model)

    return intrinsics, distortion_params, distortion_model


def imu_callback(msg, imu_ts):
    imu_ts.append(msg.header.stamp.to_nsec())


def filter_T_wc(
    T_wc_position,
    T_wc_orientation,
    T_wc_timestamp,
    imu_timestamp
):
    _, valid_indices, _ = np.intersect1d(
        T_wc_timestamp, imu_timestamp, assume_unique=True, return_indices=True
    )
    T_wc_position = T_wc_position[valid_indices, :].copy(order="C")
    T_wc_orientation = T_wc_orientation[valid_indices, :].copy(order="C")

    return T_wc_position, T_wc_orientation, imu_timestamp


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Script for pre-processing ESIM .conf file &"
                     " ESIM-generated rosbag into a dataset")
    )
    parser.add_argument(
        "conf_path", type=str, help="Path to the ESIM conf file."
    )
    parser.add_argument(
        "rosbag_path", type=str, help="Path to the ESIM-generated rosbag."
    )
    parser.add_argument(
        "dataset_path", type=str,
        help="Desired path to the pre-processed dataset."
    )
    args = parser.parse_args()

    main(args)
