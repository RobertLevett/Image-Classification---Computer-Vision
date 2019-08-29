import pickle

from keras import Model, Sequential
from keras.backend.tensorflow_backend import clear_session
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Flatten, Dense, Dropout, np
from keras.optimizers import Adagrad, Adam
from keras_preprocessing.image import ImageDataGenerator


def build_model(base, ignore_layers):
    # base = VGG16(include_top=False, input_shape = (224,224, 3))
    print(base.summary())
    step = base.output
    print(base.output.shape)
    flat = Flatten()(step)
    d1 = Dense(50,  activation='relu')(flat)
    # drop = Dropout(0.2)(d1)
    d2 = Dense(50,  activation='relu')(d1)

    out = Dense(15, activation='sigmoid')(d2)

    model = Model(inputs=base.input, outputs=out)

    for layer in model.layers[:ignore_layers]:
        layer.trainable = False

    for i,layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001),
                  metrics=['accuracy'])

    print(model.summary())

    return model



def train_model(model, epochs, train_dir, validation_dir, weight_path, hist_path):
    # filepath = './weights/transfer_32/model-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}-val_acc{val_acc:.5f}.h5'
    filepath = weight_path
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(patience=100)

    batch_size = 8

    train_datagen = ImageDataGenerator(
        preprocessing_function= preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    # train_datagen.fit(train_x, augment=False, rounds=1, seed=None)

    train_generator = train_datagen.flow_from_directory(
        directory = train_dir,
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=0,
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    # val_datagen.fit(val_x)

    val_generator = val_datagen.flow_from_directory(
        directory=validation_dir,
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=0,
    )

    # val_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(
    #     (1, 1, 3))
    # val_datagen.std = np.array([70.1154000000000, 69.4928684229438, 72.4153833334214], dtype=np.float32).reshape(
    #     (1, 1, 3))
    #
    # train_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(
    #     (1, 1, 3))
    # train_datagen.std = np.array([70.1154000000000, 69.4928684229438, 72.4153833334214], dtype=np.float32).reshape(
    #     (1, 1, 3))

    hist = model.fit_generator(train_generator, validation_data=val_generator, steps_per_epoch=train_generator.n / batch_size, \
                               validation_steps=val_generator.n / batch_size, epochs=epochs, callbacks=[checkpoint, early_stop])

    with open(hist_path, 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)


if __name__ == "__main__":
    base = VGG16(include_top=False, input_shape=(224, 224, 3))

    # newMod = Sequential()
    #
    # for layer in base.layers[:-1]:  # this is where I changed your code
    #     newMod.add(layer)

    print(base.summary())
    weight_path = './weights/model-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}-val_acc{val_acc:.5f}.h5'
    hist_path = './weights/trainHist'
    # train_data_path = '/gpfs/home/zhv14ybu/datasets/cv/train/'
    train_data_path = '/gpfs/home/zhv14ybu/datasets/cv/train/'
    test_data_path = '/gpfs/home/zhv14ybu/datasets/cv/test/'
    # test_data_path = '/gpfs/home/zhv14ybu/datasets/cv/test/'
    model = build_model(base, 19)
    train_model(model, 200, train_data_path, test_data_path, weight_path, hist_path)



