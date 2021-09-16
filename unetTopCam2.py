## codes: https://github.com/bnsreenu/python_for_microscopists/blob/master/076-077-078-Unet_nuclei_tutorial.py

import os
import numpy as np 
import cv2
import random
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda


from tqdm import tqdm
import matplotlib.pyplot as plt


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
n_classes = 2

TRAIN_PATH = r'/home/ngu0270181/unettopcam/nFold2/train/'
TEST_PATH = r'/home/ngu0270181/unettopcam/nFold2/test/'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

idx = 0
for name in train_ids:
    idx = idx + 1
    print(str(idx)+" "+name)


X_train = np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
Y_train = np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH, 1), dtype = np.bool)

print('Resizing training image and masks')
idx = 0
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    #path = TRAIN_PATH +'\\'+id_

    path = os.path.join(TRAIN_PATH, id_)
    imgFile = id_+'.jpg'
    imgPath = os.path.join(path,'images',imgFile)
    print(imgPath)
    img = cv2.imread(imgPath)
    img = cv2.resize(img,(IMG_WIDTH, IMG_HEIGHT) , interpolation = cv2.INTER_AREA)
    X_train[n] = img 
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)
    idx = idx + 1
    
    idy = 0
    for mask_file in next(os.walk(path+'//mask'))[2]:
        maskPath = os.path.join(path,'mask',mask_file)
        mask_ = cv2.imread(maskPath)
        mask_ = cv2.resize(mask_,(IMG_WIDTH, IMG_HEIGHT) , interpolation = cv2.INTER_AREA)
        mask_ = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
        #os.system('cls')
        #print(mask_.shape)

        mask_ = np.expand_dims(mask_, axis=-1)
        print("mask_.shape ",mask_.shape)
        mask = np.maximum(mask, mask_)
        idy = idy + 1
        print("idx %d   idy %d"%(idx,idy))

        Y_train[n] = mask
        #print(maskPath)


# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images') 
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    path = os.path.join(TEST_PATH, id_)
    imgFile = id_+'.jpg'
    imgPath = os.path.join(path,'images',imgFile)
    print(imgPath)

    img = cv2.imread(imgPath)
    sizes_test.append([img.shape[0],img.shape[1]])
    img = cv2.resize(img,(IMG_WIDTH, IMG_HEIGHT) , interpolation = cv2.INTER_AREA)
    X_test[n] = img 


print('Done!')



#image_x = random.randint(0, len(train_ids))
#cv2.imshow("X_train",X_train[image_x])
#plt.show()
#cv2.imshow("Y_train",np.squeeze(Y_train[image_x]))
#plt.show()


################################################################
#def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):

inputs = keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = keras.layers.Lambda(lambda x: x / 255)(inputs)

    #Contraction path
c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = keras.layers.Dropout(0.1)(c1)
c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    
c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = keras.layers.Dropout(0.1)(c2)
c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = keras.layers.MaxPooling2D((2, 2))(c2)
     
c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = keras.layers.Dropout(0.2)(c3)
c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = keras.layers.MaxPooling2D((2, 2))(c3)
     
c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = keras.layers.Dropout(0.2)(c4)
c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
     
c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = keras.layers.Dropout(0.3)(c5)
c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
#Expansive path 
u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = keras.layers.concatenate([u6, c4])
c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = keras.layers.Dropout(0.2)(c6)
c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = keras.layers.concatenate([u7, c3])
c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = keras.layers.Dropout(0.2)(c7)
c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = keras.layers.concatenate([u8, c2])
c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = keras.layers.Dropout(0.1)(c8)
c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = keras.layers.concatenate([u9, c1], axis=3)
c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = keras.layers.Dropout(0.1)(c9)
c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
#outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
#model = Model(inputs=[inputs], outputs=[outputs])    
    #NOTE: Compile the model in the main program to make it easy to test with various loss functions
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    
    #model.summary()    
#    return model
#os.system('cls')
#modelRd = multi_unet_model(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)



modelRd = keras.Model(inputs=[inputs], outputs=[outputs])
modelRd.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelRd.summary()


################################
#Modelcheckpoint
modelPath = r'/home/ngu0270181/unettopcam/model'
checkpointer = keras.callbacks.ModelCheckpoint(os.path.join(modelPath,'modelTopCamFold0.h5'), verbose=1, save_best_only=True)

callbacks = [
        keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
        keras.callbacks.TensorBoard(log_dir='logs')]

results = modelRd.fit(X_train, Y_train, validation_split=0.2, batch_size=16, epochs=50, callbacks=callbacks)

modelJsonFileNm = "unetarchi.json"
modelH5FileNm = "unetarchi.h5"
modelBestNm = "theBestModelOfUnet.hdf5"

model_json = modelRd.to_json()


with open(os.path.join(r'/home/ngu0270181/unettopcam/model',modelJsonFileNm), "w") as json_file:
    json_file.write(model_json)

modelRd.save_weights(os.path.join(r'/home/ngu0270181/unettopcam/model',modelH5FileNm))
print("Saved model to disk")



####################################
## idx = random.randint(0, len(X_train))


preds_train = modelRd.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val   = modelRd.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test  = modelRd.predict(X_test, verbose=1)


preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t   = (preds_val > 0.5).astype(np.uint8)
preds_test_t  = (preds_test > 0.5).astype(np.uint8)

print(type(preds_test))
print(preds_test.shape)
print(preds_test[0,10,10])
print(preds_test_t[0,10,10])
print(np.max(preds_test[0,:,:]))

imr = np.zeros((IMG_WIDTH,IMG_HEIGHT),dtype=np.uint8)
imTest = np.zeros((IMG_WIDTH,IMG_HEIGHT,3),dtype=np.uint8)
imgIdx = 2


pathToSave = r'/home/ngu0270181/unettopcam/result2'


maskFileList = os.listdir(TEST_PATH)
NofTest = len(maskFileList)

for imgIdx in range(NofTest):

    for i in range(IMG_HEIGHT):
        for j in range(IMG_WIDTH):
            imr[i,j] = int(preds_test_t[imgIdx,i,j]*255)
        #print(preds_test[imgIdx,i,j])
            imTest[i,j,0] = X_test[imgIdx,i,j,0]
            imTest[i,j,1] = X_test[imgIdx,i,j,1]
            imTest[i,j,2] = X_test[imgIdx,i,j,2]




    resultFileName = maskFileList[imgIdx]+'.jpg'
    fullPathToSave = os.path.join(pathToSave,resultFileName) 
    print(fullPathToSave)


    widthOri = sizes_test[imgIdx][1]
    heightOri = sizes_test[imgIdx][0]

    imr2 = cv2.resize(imr,(widthOri, heightOri) , interpolation = cv2.INTER_AREA)

    cv2.imwrite(fullPathToSave,imr2)
    imgIdx = imgIdx + 1


	



print("unetTopCam.py has been run!")


#cv2.waitKey(0)
#cv2.destroyAllWindows()