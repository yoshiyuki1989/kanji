import numpy as np
import pickle
import random
import os
import json
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import scipy.misc
from scipy import ndimage
from keras import backend as K
from keras import initializers
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
#%matplotlib inline

def save_model(model, model_name = 'model', weight_name = 'epochs_'):
    """
    Argument: model
                        file_name
    Return      : None(Create (1)model file (2)weight file)
    """
    model.save_weights('{}.hdf5'.format(weight_name))
    model_json = model.to_json()
    with open('{}.json'.format(model_name), 'w') as f:
        json.dump(model_json, f)
    return None

def load_model(model_file, weight_file, loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy']):
    with open(model_file) as f:
        model_json = json.load(f)
    model = model_from_json(model_json)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.load_weights(weight_file)
    return model
"""    
jis = []
for i in range(ord('ア'), ord('ン')+1):
    jis.append(chr(i))
for i in list('ィゥェォガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポャュョッヮヰヱ'):
    jis.remove(i)
jis_key = jis"""

def analyze_katakana(model, file):
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)   
    if np.sum(th2 == 255) > np.sum(th2 ==0):  #background =>black, character => white
        th2 = cv2.bitwise_not(th2)
    #plt.imshow(th2, interpolation='nearest')    =>draw converted picture
    ary = np.array(th2)
    ary = ary.astype(np.float32) / 15
    im_test = np.zeros([nb_classes * 160, img_rows, img_cols], dtype=np.float32)
    im_test = scipy.misc.imresize(ary, (img_rows, img_cols), mode='F')
    im_test = im_test.reshape((1, 32, 32, 1))
    preds = model.predict(im_test)
    return jis_key[np.argmax(preds)]


#original_models
nb_classes = 3037
img_rows, img_cols = 64, 64
data_file = "./kanji.pickle" # データファイル
max_bytes = 2**31 - 1

## read
with open(data_file, 'rb') as f_in:
    input_size = os.path.getsize(data_file)
    if input_size > max_bytes:
        bytes_in = f_in.read(max_bytes)
        for _ in range(max_bytes+1, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    else:
        bytes_in = f_in.read()
    data = pickle.loads(bytes_in)

data_1 = random.sample(data, 200000)

x = [] #picture
y = [] #label
for i, d in enumerate(data_1):
    (num, img) = d
    img = img.astype('float').reshape( img_rows, img_cols, 1) / 255 # 画像データを正規化
    y.append(np_utils.to_categorical(num, nb_classes)) # ラベルデータをone-hotベクトルに変換
    x.append(img)
print("data downloaded!")


X = np.array(x)
Y = np.array(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.03, shuffle = True)

datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
datagen.fit(X_train)

input_shape = (img_rows, img_cols, 1)

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
                                                               
#model.summary()
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=16), samples_per_epoch=X_train.shape[0],
                    nb_epoch=1, validation_data=(X_test, Y_test))


preds = model.evaluate(x = X_test, y =Y_test)
### END CODE HERE ###
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


save_model(model, model_name = 'model', weight_name = 'epochs_')
