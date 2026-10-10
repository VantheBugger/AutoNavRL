from dataclasses import dataclass

import numpy as np


ACKERMANN = 0
LATERAL = 1
PARALLEL = 2
MODE_NAMES = ("ackermann", "lateral", "parallel")


@dataclass
class FourWISCommand:
    forward: float
    lateral: float
    yaw_rate: float
    mode: int
    mode_name: str
    mode_switched: bool
    switch_penalty: float
    steer_angle: float


class FourWISActionMapper:
    """
    Map a compact continuous RL action into a 4WIS body twist.

    The policy action is [speed, steer_or_direction, mode_selector].
    The output is [forward, lateral, yaw_rate], which IRSim can simulate with
    its omni_angular kinematics. This keeps the training loop stable while
    explicitly exposing 4WIS motion modes to the policy.
    """

    def __init__(
        self,
        wheelbase=0.7,
        track_width=0.5,
        max_speed=0.8,
        max_lateral_speed=0.6,
        max_yaw_rate=1.2,
        max_steer=np.deg2rad(30.0),
        switch_penalty=0.35,
    ):
        self.wheelbase = float(wheelbase)
        self.track_width = float(track_width)
        self.max_speed = float(max_speed)
        self.max_lateral_speed = float(max_lateral_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.max_steer = float(max_steer)
        self.switch_penalty = float(switch_penalty)
        self.current_mode = ACKERMANN
        self.mode_switches = 0
        self.mode_counts = np.zeros(len(MODE_NAMES), dtype=np.int64)

    def reset(self):
        self.current_mode = ACKERMANN
        self.mode_switches = 0
        self.mode_counts[:] = 0

    @staticmethod
    def selector_to_mode(selector):
        selector = float(np.clip(selector, -1.0, 1.0))
        if selector < -1.0 / 3.0:
            return ACKERMANN
        if selector < 1.0 / 3.0:
            return LATERAL
        return PARALLEL

    def map(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size < 3:
            raise ValueError("4WIS action must be [speed, steer_or_direction, mode_selector]")

        speed_cmd = float(np.clip(action[0], -1.0, 1.0))
        aux_cmd = float(np.clip(action[1], -1.0, 1.0))
        mode = self.selector_to_mode(action[2])
        switched = mode != self.current_mode
        if switched:
            self.mode_switches += 1
        self.current_mode = mode
        self.mode_counts[mode] += 1

        steer_angle = aux_cmd * self.max_steer
        penalty = self.switch_penalty if switched else 0.0

        if mode == ACKERMANN:
            forward = speed_cmd * self.max_speed
            lateral = 0.0
            yaw_rate = forward * np.tan(steer_angle) / max(self.wheelbase, 1e-6)
        elif mode == LATERAL:
            forward = 0.0
            lateral = speed_cmd * self.max_lateral_speed
            yaw_rate = lateral * np.tan(steer_angle) / max(self.track_width, 1e-6)
        else:
            direction = aux_cmd * np.pi
            speed = speed_cmd * self.max_speed
            forward = speed * np.cos(direction)
            lateral = speed * np.sin(direction)
            yaw_rate = 0.0

        yaw_rate = float(np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate))
        return FourWISCommand(
            forward=float(forward),
            lateral=float(lateral),
            yaw_rate=yaw_rate,
            mode=mode,
            mode_name=MODE_NAMES[mode],
            mode_switched=switched,
            switch_penalty=penalty,
            steer_angle=float(steer_angle),
        )

    def mode_one_hot(self):
        one_hot = np.zeros(len(MODE_NAMES), dtype=np.float32)
        one_hot[self.current_mode] = 1.0
        return one_hot

    def mode_ratios(self):
        total = int(self.mode_counts.sum())
        if total == 0:
            return {f"{name}_ratio": 0.0 for name in MODE_NAMES}
        return {
            f"{name}_ratio": float(self.mode_counts[i] / total)
            for i, name in enumerate(MODE_NAMES)
        }
