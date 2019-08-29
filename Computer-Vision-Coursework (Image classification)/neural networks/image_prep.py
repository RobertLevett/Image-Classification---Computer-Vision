from os import listdir
from random import shuffle
from shutil import copyfile

from pickle import dump
# from CaptionGeneration.COCO.pycocotools.coco import COCO
from keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.engine.saving import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import Model

# def train_test_split(dataDir, classes):

#     annFile = '{}/annotations/instances_{}.json'.format(dataDir, "train2014")
#     coco = COCO(annFile)
#     train = open("trainImages.txt", 'w')
#     train_list = list()

#     for c in classes:
#         catIds = coco.getCatIds(catNms=[c])
#         imgIds = coco.getImgIds(catIds=catIds)
#         for im in coco.loadImgs(imgIds):
#             train_list.append(im['file_name'])

#     annFile = '{}/annotations/instances_{}.json'.format(dataDir, "val2014")
#     coco = COCO(annFile)
#     test = open("testImages.txt", 'w')
#     test_list = list()
#     for c in classes:
#         catIds = coco.getCatIds(catNms=[c])
#         imgIds = coco.getImgIds(catIds=catIds)
#         for im in coco.loadImgs(imgIds):
#             test_list.append(im['file_name'])

#     shuffle(train_list)
#     shuffle(test_list)
#     for i in train_list:
#         train.write(i)
#         train.writelines("\n")
#     for i in test_list:
#         test.write(i)
#         test.writelines("\n")



def extract_features(direc, features, model):
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    for name in listdir(direc):
        filename = direc + '/' + name
        img = load_img(filename, target_size=(224, 224))
        img = img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        feature = model.predict(img, verbose=0)
        img_id = name.split('.')[0]
        features[img_id] = feature
        print('>%s' % name)
        print(feature.shape)
    return features



# def copy_files(dataDir,dataType, classes):
#     annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
#     coco = COCO(annFile)
#
#
#     features = dict()
#     for c in classes:
#         catIds = coco.getCatIds(catNms=[c])
#         imgIds = coco.getImgIds(catIds=catIds)
#         for im in coco.loadImgs(imgIds):
#             src = dataDir + "/" + dataType + "/" + im['file_name']
#             dst = "/Users/jonathanwindle/Documents/ThirdYear/Project/datasets/COCO/subset/" + dataType + "/" + im['file_name']
#             copyfile(src, dst)


# model = load_model('/gpfs/home/zhv14ybu/VGG-VOC_transfer/weights/transfer_128/model-ep008-loss0.36974-val_loss0.38923-val_acc0.85408.h5')
# model = load_model('/Users/jonathanwindle/Documents/ThirdYear/Project/repo/ThirdYearProject/FeatureLearning/VGG/VGG-Transfer/weights/transfer/model-ep008-loss0.36974-val_loss0.38923-val_acc0.85408.h5')
# model = load_model('/Users/jonathanwindle/Documents/ThirdYear/Project/repo/ThirdYearProject/FeatureLearning/VOC_Experiments/randomgridsearch/best_weights.271-val_loss 0.52637-acc0.83195.h5')
model = load_model('/home/ml/Documents/tensorTest/weights/transfer_Xcep/model-ep162-loss2.82299-val_loss2.40586-val_acc0.85074.h5')
print(model.summary())
# print(model.optimizer.lr)
# model = VGG16()
# dataDir='/gpfs/home/zhv14ybu/datasets/COCO'
# dataDir='/Users/jonathanwindle/Documents/ThirdYear/Project/datasets/COCO'
dataDir='/home/ml/Documents/datasets/COCO'

# train_test_split(dataDir, ['person', 'dog', 'boat'])
dataType='val2014'
features = dict()
features = extract_features(dataDir + "/" + dataType, features, model)
features = extract_features(dataDir + "/train2014" , features, model)
# print(features)
# # print(features)
dump(features, open('features_COCO_Xcep.pkl', 'wb'))