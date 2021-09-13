import dataclasses
import functools
import numpy as np
import yaml


@dataclasses.dataclass(eq=True, frozen=True)
class KinematicsConfig:
    abduction_offset: float
    upper_link_length: float
    lower_link_length: float
    hip_x_offset: float
    hip_y_offset: float

    @classmethod
    def from_yaml(cls, config_path):
        with open(config_path, "r") as sim_config_yaml:
            config = yaml.safe_load(sim_config_yaml)
            return KinematicsConfig(
                hip_x_offset=config["hip_x_offset"],
                hip_y_offset=config["hip_y_offset"],
                lower_link_length=config["lower_link_length"],
                upper_link_length=config["upper_link_length"],
                abduction_offset=config["abduction_offset"],
            )

    @functools.lru_cache(maxsize=128)
    def leg_origins(self):
        return np.array(
            [
                [
                    self.hip_x_offset,
                    self.hip_x_offset,
                    -self.hip_x_offset,
                    -self.hip_x_offset,
                ],
                [
                    -self.hip_y_offset,
                    self.hip_y_offset,
                    -self.hip_y_offset,
                    self.hip_y_offset,
                ],
                [0, 0, 0, 0],
            ]
        )

    @functools.lru_cache(maxsize=128)
    def abduction_offsets(self):
        return np.array(
            [
                -self.abduction_offset,
                self.abduction_offset,
                -self.abduction_offset,
                self.abduction_offset,
            ]
        )

    @functools.lru_cache(maxsize=128)
    def abduction_offset_i(self, leg_index: int):
        if leg_index == 0 or leg_index == 2:
            return -self.abduction_offset
        else:
            return self.abduction_offset
