from dataclasses import dataclass
import numpy as np
from loss_US import CustomLoss
import math

@dataclass()
class medium:
    sound_speed: float # [m/s]
    density:float #[kg/m^3]
    alpha_coeff: float #[dB(MHz^y cm)]
    alpha_power: float
    # BonA: float # nonlinearity; B/A characterises the  relative contribution of finite-amplitude effects to the sound speed



@dataclass()
class txdr_param:
    grid_spacing: float
    total_grids: np.ndarray
    size: np.ndarray
    phase_levels: float
    D_txdr: float
    single_txdr: bool



@dataclass()
class txdr_points:
    kerf: int
    width: int
    height: int
    txdr_width: int

@dataclass()
class txdr_output:
    points: txdr_points
    mask: np.ndarray
    ele_pos: np.ndarray
    center: np.ndarray
    phase_levels: int
    single_txdr: bool
    D_txdr_point: int


@dataclass()
class var_SGD:
    txdr_pts: txdr_points
    txdr_out: txdr_output
    iteration: int
    learning_rate: float
    loss: CustomLoss

@dataclass()
class prop_param:
    input_shape: tuple
    medium: medium
    Freq: float
    prop_distance: list
    grid_spacing: float
    txdr_output: txdr_output
    lens_SoS: float
    lens_impedance: float
    txdr_impedance: float
    prop_resize_ratio: int
    element_size: float
    PC_padding: int


