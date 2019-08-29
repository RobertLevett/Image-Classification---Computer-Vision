import pickle

from keras import Sequential, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Flatten
import scipy.io
import numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder,StandardScaler


def build_model():
    model = Sequential()
    model.add(Dense(50, input_dim=6300, activation='relu', name="Input_Dense_1"))
    # model.add(Dense(50, activation='sigmoid', name="Dense_3"))
    model.add(Dense(15, name="Output"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model




if __name__ == '__main__':
    # mat = scipy.io.loadmat('/gpfs/home/zhv14ybu/computervision/cw2/own_model/data/spatial_200_gray_5_3_2_3.mat')
    mat = scipy.io.loadmat('/Users/jonathanwindle/Documents/ThirdYear/ComputerVision/cw2/deepLearning/spatial_300_gray_5_3_1_3.mat')
    train_feats = mat['train_features']
    mat = scipy.io.loadmat('/Users/jonathanwindle/Documents/ThirdYear/ComputerVision/cw2/deepLearning/spatial_300_gray_5_3_1_3_test.mat')
    # mat = scipy.io.loadmat('/gpfs/home/zhv14ybu/computervision/cw2/own_model/data/spatial_200_gray_5_3_2_3_test.mat')
    test_feats = mat['test_features']

    # define example
    # data = scipy.io.loadmat('/gpfs/home/zhv14ybu/computervision/cw2/own_model/data/values.mat')
    data = scipy.io.loadmat('/Users/jonathanwindle/Documents/ThirdYear/ComputerVision/cw2/code/values.mat')
    values = array(data['train_labels'], object)

    l = list()

    for v in values:
        l.append(str(v[0][0]))

    # print(l)

    # encoded = to_categorical(array(l))
    #
    # print(encoded)

    values = array(l)

    # print(values.shape)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded_train = label_encoder.fit_transform(values)
    # print(integer_encoded_train)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded_train = integer_encoded_train.reshape(len(integer_encoded_train), 1)
    onehot_encoded_train = onehot_encoder.fit_transform(integer_encoded_train)
    # print(onehot_encoded_train)


    values = array(data['test_labels'], object)

    l = list()

    for v in values:
        l.append(str(v[0][0]))

    # print(l)

    values = array(l)

    integer_encoded_test = label_encoder.transform(values)

    integer_encoded_test = integer_encoded_test.reshape(len(integer_encoded_test), 1)
    onehot_encoded_test = onehot_encoder.transform(integer_encoded_test)


    # print(train_feats.shape)

    scaler = StandardScaler()
    train_feats = scaler.fit_transform(train_feats)

    test_feats = scaler.transform(test_feats)

    model = build_model()
    checkpoint = ModelCheckpoint('./weights/model-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}-val_acc{val_acc:.5f}.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(patience=100)

    hist = model.fit(train_feats,onehot_encoded_train, epochs=100, batch_size=1, shuffle=True, validation_data=(test_feats, onehot_encoded_test), callbacks=[checkpoint, early_stop])


    with open('history.pkl', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)