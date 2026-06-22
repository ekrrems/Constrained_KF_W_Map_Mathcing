import os
import cv2
import numpy as np

class Featurehandler():
	def __init__():
		print("Feature are being generated")
		# self.keyframes = {}
		self.keypoints = {}

		self.orb = cv2.ORB_create(
			nfeatures=3000,
			scaleFactor=1.2,
			nlevels=8,
			edgeThreshold=31,
			fastThreshold=20,
		)

	def createKeyframes(image):
