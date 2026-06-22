# import os
# import cv2
# import numpy as np

# # get image feature matches
# # implement ransac
# #get the correct features
# # disparity calculation
# # depth of the points in there
# # 3d maps information

# def filter_rectified_stereo_matches(
#     keypoints_left,
#     keypoints_right,
#     matches,
#     max_vertical_difference=2.0,
#     min_disparity=1.0,
#     max_disparity=256.0,
# ):
# 	filtered_matches = []

# 	for match in matches:
# 		u_left, v_left = keypoints_left[match.queryIdx].pt
# 		u_right, v_right = keypoints_right[match.trainIdx].pt

# 		vertical_error = abs(v_left - v_right)
# 		disparity = u_left - u_right

# 		if (
# 			vertical_error <= max_vertical_difference
# 			and min_disparity <= disparity <= max_disparity
# 		):
# 			filtered_matches.append(match)

# 	return filtered_matches

# # Create two image sized array: you can put left and right
# dataPath = "/Users/ekremserdarozturk/Desktop/Projects/Datasets/KITTI_RAW/2011_10_03/2011_10_03_drive_0027_sync"
# leftImageFolder = os.path.join(dataPath, "image_00/data")
# rightImageFolder = os.path.join(dataPath, "image_01/data")
# # print(os.listdir(rightImageFolder))
# numberOfImages = len(os.listdir(rightImageFolder))

# print(numberOfImages)


# left_files = sorted(
# 	file_name
# 	for file_name in os.listdir(leftImageFolder)
# 	if file_name.lower().endswith(".png")
# )

# right_files = sorted(
# 	file_name
# 	for file_name in os.listdir(rightImageFolder)
# 	if file_name.lower().endswith(".png")
# )

# # Frame downsampling
# frame_step = 2

# left_files = left_files[::frame_step]
# right_files = right_files[::frame_step]

# # ORB Feature detector
# orb = cv2.ORB_create(
# 	nfeatures=3000,
# 	scaleFactor=1.2,
# 	nlevels=8,
# 	edgeThreshold=31,
# 	fastThreshold=20,
# )

# matcher = cv2.BFMatcher(
# 	cv2.NORM_HAMMING,
# 	crossCheck=False,
# )


# # Additional control
# if len(left_files) != len(right_files):
# 	raise ValueError("Left and right folders contain different numbers of images.")

# #get the size of the image
# for imageName in left_files:
# 	leftImagePath = os.path.join(leftImageFolder, imageName)
# 	rightImagePath = os.path.join(rightImageFolder, imageName)

# 	leftImage = cv2.imread(leftImagePath)
# 	rightImage = cv2.imread(rightImagePath)

# 	if leftImage.shape[:2] != rightImage.shape[:2]:
# 		raise ValueError("Left and right images are not in the same shape")

# 	# Get the features


# 	left_keypoints, left_descriptors = orb.detectAndCompute(
# 		leftImage,
# 		None,
# 	)

# 	right_keypoints, right_descriptors = orb.detectAndCompute(
# 		rightImage,
# 		None,
# 	)


# 	knn_matches = matcher.knnMatch(
# 		left_descriptors,
# 		right_descriptors,
# 		k=2,
# 	)

# 	good_matches = []

# 	for match_1, match_2 in knn_matches:
# 		if match_1.distance < 0.75 * match_2.distance:
# 			good_matches.append(match_1)

# 	stereo_matches = filter_rectified_stereo_matches(
# 		left_keypoints,
# 		right_keypoints,
# 		good_matches,
# 	)

# 	print("Left features:", len(left_keypoints))
# 	print("Right features:", len(right_keypoints))
# 	print("Descriptor matches:", len(good_matches))
# 	print("Geometrically valid stereo matches:", len(stereo_matches))


# 	# Create a size of the two images
# 	combinationImage = np.hstack((leftImage, rightImage))

# 	# cv2.imshow('Left Image', leftImage)
# 	cv2.imshow("combination", combinationImage)
# 	cv2.waitKey(1)

# cv2.destroyAllWindows()



import os
import cv2
import numpy as np
from utils import ImageHandler, FeatureHandler, OdometryHandler


imageHandler = imageHandler
