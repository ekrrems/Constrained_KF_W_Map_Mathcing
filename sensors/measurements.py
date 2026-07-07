import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass
class ImuMeasurement:
	timestamp: float
	acceleration: np.ndarray
	angular_velocity: np.ndarray

@dataclass
class ImageMeasurement:
	timestamp: float
	frame_index: int
	left_image_path: Path
	right_image_path: Path

@dataclass
class LidarMeasurement:
	timestamp: float
	frame_index: int
	scan_path: Path

SensorMeasurement = (
	ImuMeasurement
	| LidarMeasurement
	| ImageMeasurement
)