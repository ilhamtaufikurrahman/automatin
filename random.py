import random, os, cv2

path1 = "E:/cent/full"
path2 = "D:/Video/Autonomous/1201-top/Frame/1201-top/Total/1201-top/YDXJ0049 (8-7-2021 12-16-03 PM)"
path3 = "D:/Video/Autonomous/1201-top/Frame/1201-top/Total/1201-top/YDXJ0050 (8-7-2021 12-16-06 PM)"
path4 = "D:/Video/Autonomous/1201-top/Frame/1201-top/Total/1201-top/YDXJ0051 (8-7-2021 12-16-40 PM)"
path5 = "D:/Video/Autonomous/1201-top/Frame/1201-top/Total/1201-top/YDXJ0052 (8-7-2021 12-17-12 PM)"
path6 = "D:/Video/Autonomous/1201-top/Frame/1201-top/Total/1201-top/YDXJ0053 (8-7-2021 12-17-38 PM)"

pathSave = "E:/cent/300"

listPath = [path1]

for x in range(330):
    randomPath = random.choice(listPath)
    random_filename = random.choice([
        x for x in os.listdir(randomPath)
        if os.path.isfile(os.path.join(randomPath, x))
        ])

    print(random_filename)
    img = cv2.imread(randomPath+ "/" + random_filename)
    cv2.imwrite(pathSave + "/" + random_filename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# list = []

# for x in range(200):
#  list.append(x+1)

# for item in list:
#  print(item)