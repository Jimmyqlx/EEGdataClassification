from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from keras.layers import Input,Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
dir = 'D:\\PycharmProjects\\ces734_final_project\\data'
total_fold = ['1','2','3','4','5','6','7','8','9','10']
result=[[],[]]

def read_mat(fold):
    #Features=[]
    #Labels=[]
    #Features=np.array()
    features_mat = sio.loadmat('{}/Feature{}.mat'.format(dir,fold[0]))
    features = features_mat['Feature{}'.format(fold[0])]
    features=np.transpose(features)
    labels_mat = sio.loadmat('{}/Y{}.mat'.format(dir,fold[0]))
    labels=labels_mat['Y{}'.format(fold[0])]
    labels=labels[0]
    #Labels.append(labels)
    for i in range(1,len(fold)):
        f_mat = sio.loadmat('{}/Feature{}.mat'.format(dir,fold[i]))
        f = f_mat['Feature{}'.format(fold[i])]
        f=np.transpose(f)
        features = np.concatenate((features,f))
        #Features.append(f)
        l_mat = sio.loadmat('{}/Y{}.mat'.format(dir,fold[i]))
        l = l_mat['Y{}'.format(fold[i])]
        l=l[0]
        labels = np.concatenate([labels,l])
        #Labels.append(labels)
    #Features = np.array(Features)
    #Labels = np.array(Labels)
    return features,labels
data_x, data_y  = read_mat(total_fold)
for j in range(0, 4):
    data_x_temp = data_x.copy()
    test_x = data_x_temp[(j) * math.floor(data_x.shape[0] / 4):(j + 1) * math.floor(data_x.shape[0] / 4)]
    train_x_1 = data_x_temp[0:j * math.floor(data_x.shape[0] / 4)]
    train_x_2 = data_x_temp[(j + 1) * math.floor(data_x.shape[0] / 4) + 1:]

    train_x = np.concatenate((train_x_1, train_x_2))
    data_y_temp = data_y.copy()

    test_y = data_y_temp[j * math.floor(data_y.shape[0] / 4):(j + 1) * math.floor(data_y.shape[0] / 4)]
    train_y_1 = data_y_temp[0:j * math.floor(data_y.shape[0] / 4)]
    train_y_2 = data_y_temp[(j + 1) * math.floor(data_y.shape[0] / 4) + 1:]
    train_y = np.concatenate((train_y_1, train_y_2))

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    train_y = np.reshape(train_y, (train_x.shape[0], 1))
    test_y = np.reshape(test_y, (test_x.shape[0], 1))

    model = Sequential()
    model.add(LSTM(75, input_shape=(320, 1)))
    # model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_x, train_y, epochs=8, validation_data=(test_x, test_y),verbose=1)
    print('Test set {}'.format(j))
    print('Train Accuracy:{}'.format(history.history['acc'][7]))
    print('Test Accuracy:{}'.format(history.history['val_acc'][7]))
    result[0].append(history.history['acc'][7])
    result[1].append(history.history['val_acc'][7])
print('All Train Accuracy:{}'.format(result[0]))
print('All Test Accuracy:{}'.format(result[1]))
avtrain=sum(result[0])/len(result[0])
avtest = sum(result[1]) / len(result[1])
print('Average Train Accuracy:{}'.format(avtrain))
print('Average Test Accuracy:{}'.format(avtest))