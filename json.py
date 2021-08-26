import numpy as np
import json
import os
import cv2

rgbPath = "D:/Gambar/rgb/"
jsonPath = "D:/Gambar/json/contoh.json"
bwPath = "D:/Gambar/bw/"

#mengambil path rgb
rgbFiles = os.listdir(rgbPath)
#mengambil file rgb
img0 = cv2.imread(os.path.join(rgbPath,rgbFiles[0]))
#mengambil width dan height
height, width = img0.shape[:2]

#buka file json
jsonFile = open(jsonPath)

point = []

#parsing data json
data = json.loads(jsonFile.read())

#mengambil points yang ada di json
for points in data['objects'][1]['points']['exterior']:
    point.append(points)

print(point)
#menambahkan background
img = np.zeros((height, width, 3), dtype = "uint8")

#membuat polygon dari points
polygon = np.array(point, np.int32)

#membuat polygon warna putih dalam background gambar
polyImage = cv2.fillConvexPoly(img, polygon, (255,255,255))

#menampilkan gambar
cv2.imshow("example", polyImage)

cv2.waitKey(0)
cv2.destroyAllWindows()