import easydict
import torch
import roma
from ..data import datasets
from ..utils import tensor_ops


class LinearTrajectory(torch.nn.Module):
    def __init__(self, camera_poses):
        super().__init__()

        # define camera pose samples & sampling timestamps as a buffer
        self.register_buffer(
            "T_wc_position",                                                    # (C, 3)
            camera_poses.camera_poses.T_wc_position, persistent=False
        )
        self.register_buffer(
            "T_wc_orientation_quat",                                            # (C, 4)
            camera_poses.camera_poses.T_wc_orientation, persistent=False
        )
        self.register_buffer(
            "T_wc_timestamp",                                                   # (C)
            camera_poses.camera_poses.T_wc_timestamp.contiguous(),              # contiguous required for efficient `torch.searchsorted()`
            persistent=False
        )
        self.register_buffer(                                                   # (C-1)
            "bin_width", self.T_wc_timestamp.diff(), persistent=False
        )
        
    def forward(self, input_timestamp):
        """
        Args:
            input_timestamp (torch.Tensor):
                Input timestamp(s) of shape (...) with dtype in the same
                units as `self.T_wc_timestamp` & contiguous
        Returns:
            input_T_wc_position (torch.Tensor):
                Linearly interpolated position of camera poses associated to
                the input timestamps
            input_T_wc_orientation (torch.Tensor):
                Linearly interpolated orientation of camera poses associated to
                the input timestamps
        """
        # deduce the camera pose bins that input timestamps should fall into
        bin_left = easydict.EasyDict()
        bin_right = easydict.EasyDict()

        bin_right.index = torch.searchsorted(                                   # (...)
            sorted_sequence=self.T_wc_timestamp, input=input_timestamp
        )
        is_corner_case = (input_timestamp == self.T_wc_timestamp[0])            # (...)
        bin_left.index = torch.where(                                           # (...)
            is_corner_case, bin_right.index, bin_right.index - 1
        )
        assert (
            (bin_left.index >= 0)
            & (bin_right.index < len(self.T_wc_timestamp))
        ).all()

        # linearly interpolate camera positions
        weight = (input_timestamp - self.T_wc_timestamp[bin_left.index]) \
                 / self.bin_width[bin_left.index]                               # (...)
        weight = weight.to(self.T_wc_position.dtype)

        bin_left.T_wc_position = self.T_wc_position[bin_left.index, :]          # (..., 3)
        bin_right.T_wc_position = self.T_wc_position[bin_right.index, :]        # (..., 3)
        input_T_wc_position = torch.lerp(                                       # (..., 3)
            input=bin_left.T_wc_position,
            end=bin_right.T_wc_position,
            weight=weight.unsqueeze(dim=-1)                                     # (..., 1)
        )

        # apply slerp to interpolate orientation
        bin_left.T_wc_orientation_quat = (                                      # (..., 4)
            self.T_wc_orientation_quat[bin_left.index, :]
        )
        bin_right.T_wc_orientation_quat = (                                     # (..., 4)
            self.T_wc_orientation_quat[bin_right.index, :]
        )

        input_T_wc_orientation_quat = tensor_ops.unitquat_slerp(                # (..., 4)
            q0=bin_left.T_wc_orientation_quat,
            q1=bin_right.T_wc_orientation_quat,
            steps=weight, shortest_path=True
        )
        input_T_wc_orientation = roma.unitquat_to_rotmat(                       # (..., 3, 3)
            input_T_wc_orientation_quat
        )

        return input_T_wc_position, input_T_wc_orientation
