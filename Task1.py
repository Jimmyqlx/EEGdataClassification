from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from keras.layers import Input,Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
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
for i in range(0,len(total_fold)):
    total_temp=total_fold.copy()
    #print(i)
    #print(len(total_temp))
    del total_temp[i]
    #print(len(total_fold))
    #print(len(total_temp))
    train_fold=total_temp
    print(train_fold)
    test_fold=[]
    test_fold.append(total_fold[i])
    print(test_fold)
    train_x, train_y  = read_mat(train_fold)
    test_x, test_y = read_mat(test_fold)
    train_x = np.reshape(train_x, (train_x.shape[0],train_x.shape[1],1))
    test_x = np.reshape(test_x, (test_x.shape[0],test_x.shape[1],1))
    #train_y = np.reshape(train_y, (train_x.shape[0], 1))
    #test_y = np.reshape(test_y, (test_x.shape[0], 1))
    #train_y=np.reshape(train_y, (train_y.shape[0],1))
    #test_y=np.reshape(test_y, (test_y.shape[0],1))
    #print(train_x.shape)
    model = Sequential()
    model.add(LSTM(75, input_shape=(320,1)))
    #model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_x, train_y,epochs=1,validation_data=(test_x,test_y),verbose=0)
    result[0].append(history.history['acc'][0])
    result[1].append(history.history['val_acc'][0])
    print('Leave Subject{} Out'.format(i))
    print('Train Accuracy:{}'.format(history.history['acc'][0]))
    print('Test Accuracy:{}'.format(history.history['val_acc'][0]))
print('Train Accuracy:{}'.format(result[0]))
print('Test Accuracy:{}'.format(result[1]))
avtrain = sum(result[0]) / len(result[0])
avtest = sum(result[1]) / len(result[1])
print('Average Train Accuracy:{}'.format(avtrain))
print('Average Test Accuracy:{}'.format(avtest))
print(result)