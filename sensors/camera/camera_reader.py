import numpy as np
import cv2

from sensors.measurements import ImageMeasurement

def combineImages(measurement: ImageMeasurement, scale: int = 1) -> np.array:
	leftImage = cv2.imread(measurement.left_image_path)
	rightImage = cv2.imread(measurement.right_image_path)

	for _ in range(scale):
		print(f"Scaled {scale} Time/s")
		leftImage = cv2.pyrDown(leftImage)
		rightImage = cv2.pyrDown(rightImage)

	assert (leftImage.shape) == (rightImage.shape), f"Left {(leftImage.shape)} and Right {rightImage.shape} images should have the same size"

	width = leftImage.shape[1]
	height = leftImage.shape[0]

	combinedArray = np.zeros((height, width*2, 3), dtype=np.uint8)

	combinedArray[0:height, 0:width] = leftImage
	combinedArray[0:height, width:width*2] = rightImage
	return combinedArray

