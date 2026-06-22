import os
import numpy as np
import cv2


class ImageHandler:
	def __init__(folderPath: str, frameStep: int = 2):
		leftImageFolder = os.path.join(folderPath, "image_00/data")
		rightImageFolderPath = os.path.join(folderPath, "image_01/data")

		left_files = sorted(
			file_name
			for file_name in os.listdir(leftImageFolder)
			if file_name.lower().endswith(".png")
		)

		right_files = sorted(
			file_name
			for file_name in os.listdir(rightImageFolderPath)
			if file_name.lower().endswith(".png")
		)

		self.frameStep = frameStep

		self.left_files = left_files[::frame_step]
		self.right_files = right_files[::frame_step]

		if len(left_files) != len(right_files):
			raise ValueError("Left and right folders contain different numbers of images.")

	# create image combination
	#add the features
	#show the features on top of the matches

	def run():
		pass




