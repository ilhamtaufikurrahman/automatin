# import os, shutil

# rgbPath = "D:/Gambar/bw/"
# rgbFiles = os.listdir(rgbPath)
# path = 'D:/Gambar/per_folder_bw/'

# for item in rgbFiles:
#     short = item[:-4]
    
#     # print(item)
    
#     pathNew = path+short
    
#     bw = pathNew + '/images/'

#     shutil.move('D:/Gambar/per_folder_rgb/' + short + '/' + item, bw + item)

#     print('success move file ' + bw + '/images/')
#     # Membuat path baru
#     # short = item[:-4]
#     # newPath = path + short
#     # os.mkdir(newPath)
#     # print('Succesfuly make directory ' + newPath)

import os, shutil

rgbPath = "D:/PariIsland/rgb/"
rgbFiles = os.listdir(rgbPath)
path = 'D:/PariIsland/per_folder_bw/'

for item in rgbFiles:
    short = item[:-4]
    
    # print(item)
    
    pathNew = path+short
    
    bw = pathNew + '/images/'

    shutil.copy(rgbPath + item, bw + item)

    print('success copy file ' + bw + item)
    
    #Membuat path baru
    # short = item[:-4]
    # newPath = path + short
    # os.mkdir(newPath + '/images')
    # os.mkdir(newPath + '/mask')
    # print('Succesfuly make directory ' + newPath)