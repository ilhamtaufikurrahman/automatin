import numpy as np
import json
import os
import cv2

rgbPath = "D:/Gambar/rgb/"
jsonPath = "D:/Gambar/json/"
bwPath = "D:/Gambar/bw/"

jsonFiles = os.listdir(jsonPath)

for jF in jsonFiles:
	print(jF)

	with open(jsonPath+jF) as json_file:
		data = json.load(json_file)

	pcs = data['objects'][0]['points']['exterior']

	bwImg = np.zeros((324, 576, 3), np.uint8)
	polygon = np.array(pcs)
 
	cv2.fillConvexPoly(bwImg, polygon, (255, 255, 255))
	cv2.imwrite((bwPath+jF[:-4] + "jpg"),bwImg)


cv2.waitKey(0)
cv2.destroyAllWindows()