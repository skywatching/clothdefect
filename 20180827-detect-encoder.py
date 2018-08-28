# -*- coding: utf-8 -*-

import numpy as np
from keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
import keras.preprocessing.image
import keras.backend as K
#from keras.applications.Detect import ResNet50, preprocess_input
#from keras.applications.xception import Xception, preprocess_input
#from keras.applications.vgg19 import VGG19, preprocess_input
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

import os
import sys
import glob
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import keras.backend.tensorflow_backend as KTF
from keras import optimizers 
import datetime
import re
import math
import pandas as pd
import json
from pixel_shuffler import PixelShuffler

Defect_class = [
    'defect_1',
    'defect_2',
    'defect_3',
    'defect_4',
    'defect_5',
    'defect_6',
    'defect_7',
    'defect_8',
    'defect_9',
    'defect_10',
    'norm',
]
IMAGE_SHAPE=(640,640,3)
ENCODER_DIM = 3200

def preprocess_input(x):
    x = x/255.0
    return x

import imgaug as ia
from imgaug import augmenters as iaa
import logging
import skimage.io
import skimage.color

class ImgGenerator(object):
    def __init__(self, samples, class_mapping, batch_size, augment=True, shuffle=True):
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.samples = samples
        self.class_mapping = class_mapping
        self.classcnt = len(self.class_mapping.keys())
        print("Class count:{}".format(self.classcnt))
        if self.augment:
            self.augment_prepare()

    def augment_prepare(self):
        ia.seed(20180827)
        self.ia_seq = iaa.Sequential(
            [
                iaa.Affine(rotate=(-10, 10), shear=(-5, 5)),
            ],
            random_order=True,
        )
        """
        self.ia_seq = iaa.Sequential(
            [
                iaa.SomeOf([0, None],
                    [
                        #iaa.GaussianBlur(sigma=(0, 3.0)),
                        iaa.ContrastNormalization((0.75, 1.5)),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                        iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    ]
                ),
                iaa.Affine(rotate=(-5, 5), shear=(-5, 5)),
            ],
            random_order=True
        )
        """
    def augment_image(self, batch_images):
        seq_det = self.ia_seq.to_deterministic()
        images_aug = seq_det.augment_images(batch_images)
        return images_aug

    def len(self):
        return len(self.samples)

    def load_image(self, image_id):
        filepath = self.samples[image_id]
        classid = self.class_mapping[filepath.split('/')[-2]]
        img = skimage.io.imread(filepath)
        if img.ndim != 3:
            img = skimage.color.gray2rgb(img)
        img = keras.preprocessing.image.img_to_array(img)
        #print("Imgpath:{}\n label:{}".format(filepath, classid))
        return img, classid

    def flow(self):
        b = 0
        image_index = -1
        image_ids = np.arange(len(self.samples))
        block_ids = np.arange(12)
        bi = -1
        while True:
            try:
                bi = (bi + 1) % len(block_ids) 
                if bi == 0:
                    image_index = (image_index + 1) % len(image_ids)
                    if self.shuffle and image_index == 0:
                        np.random.shuffle(image_ids)

                image_id = image_ids[image_index]
                img, classid = self.load_image(image_id)

                if b == 0:
                    #batch_images = np.zeros((self.batch_size,) + img.shape, dtype=np.float32)
                    batch_images = np.zeros((self.batch_size,) + (640,640,3), dtype=np.float32)
                    batch_labels = np.zeros((self.batch_size, self.classcnt), dtype=np.int32)
                
                h,w = bi//4, bi%4
                batch_images[b] = img[h*640:(h+1)*640, w*640:(w+1)*640, :]
                batch_labels[b, classid] = 1

                b += 1
                if b >= self.batch_size:
                    if self.augment:
                        batch_images = self.augment_image(batch_images)
                    batch_images = preprocess_input(batch_images)
                    inputs = batch_images
                    outputs = batch_images   #batch_labels
                    yield inputs, outputs
                    b = 0
            except:
                logging.exception("Error for image {}".format(image_index))
                raise

def conv(filters):
    def block(x):
        x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        return x
    return block

def upscale(filters):
    def block(x):
        x = Conv2D(filters*4, kernel_size=3, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block
def Encoder():
    input_ = Input(shape=IMAGE_SHAPE)
    x = input_

    x = conv(128)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv(256)(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv(512)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv(1024)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(ENCODER_DIM)(x)
    #x = Dense(4*4*256)(x)
    x = Reshape((10,10,32))(x)
    x = upscale(128)(x)
    return Model(input_, x)

def Decoder():
    input_ = Input(shape=(20,20,128))
    x = input_
    x = upscale(256)(x)
    x = upscale(128)(x)
    x = upscale(128)(x)
    x = upscale(64)(x)
    x = upscale(64)(x)
    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
    return Model(input_, x)

def Classfy(classcnt):
    input_ = Input(shape=(20, 20, 128*12))
    x = input_
    x = Conv2D(1024, (3, 3), strides=(2, 2), name='gw_conv01')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(512, (3, 3), strides=(2, 2), name='gw_conv02')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(256, (3, 3), strides=(2, 2), name='gw_conv03')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=3)(x)
    x = Flatten()(x)
 
    x = Dense(classcnt, activation='softmax', name='predict')(x)
    model = Model(inputs=input_, outputs=x)
    return model

encoder = Encoder()
decoder = Decoder()
classfy = Classfy(11)
print(encoder.summary())
print(decoder.summary())

def detect_model(classcnt):

    x = Input(shape=IMAGE_SHAPE)

    autoencoder = Model(x, decoder(encoder(x)))

    optimizer = Adam( lr=5e-5, beta_1=0.5, beta_2=0.999 )
    autoencoder.compile( optimizer=optimizer, loss='mean_absolute_error')
    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return autoencoder


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def FScore2(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)


class WeightSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        dt ='{:%Y%m%dT%H%M}'.format(datetime.datetime.now())
        encoder.save_weights("./logs/encoder_{}.h5".format(dt))
        decoder.save_weights("./logs/decoder_{}.h5".format(dt))
                                    
class DetectTrainer():
    def __init__(self, work_dir, classes=11, step=4000):
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.steps_per_epoch = step
        self.work_dir = work_dir
        self.classes = classes

    def samples_preprocess(self, batch_size=4):
        self.batch_size = batch_size

        data = json.load(open('samples.json'))
        train_files = data['train']
        valid_files = data['valid']

        class_mapping = {defect:classid for classid, defect in enumerate(Defect_class)}
        print("Class mapping: {}".format(class_mapping))
        self.train_generator = ImgGenerator(train_files, class_mapping, self.batch_size)
        self.valid_generator = ImgGenerator(valid_files, class_mapping, self.batch_size, augment=False, shuffle=False)

    def set_log_dir(self, model_path=None):
        self.epoch = 0
        now = datetime.datetime.now()
        if model_path:
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/\w+(\d{4})-\w+\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))

        self.log_dir = os.path.join(self.work_dir, "{}{:%Y%m%dT%H%M}".format(self.name.lower(), now))
        self.weightfile = os.path.join(self.log_dir, "{}_*epoch*.h5".format(self.name.lower()))
        self.weightfile = self.weightfile.replace("*epoch*", "{epoch:04d}-{val_acc:.2f}")

    def build_model(self):
        self.model = detect_model(self.classes)
        #opt = optimizers.SGD(lr=0.1, momentum=0.9)
        #self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', precision, FScore2])
        #self.model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy', precision, FScore2])

        optimizer = Adam( lr=5e-5, beta_1=0.5, beta_2=0.999 )
        self.model.compile( optimizer=optimizer, loss='mean_absolute_error')

        print(self.model.summary())

    def train_model(self, epoch=20000):
        modelfile = None
        modelfiles = glob.glob(os.path.join(self.work_dir, "*/{}_*.h5".format(self.name.lower())))
        modelfiles = sorted(modelfiles)
        if len(modelfiles) > 0:
            modelfile = modelfiles[-1]
            print("Load Weight: {}".format(modelfile))
            self.model.load_weights(modelfile, by_name=True)
        self.set_log_dir(modelfile)

        callbacks = [
            TensorBoard(log_dir=self.log_dir, write_images=False),
            #ModelCheckpoint(self.weightfile, monitor='acc', verbose=1, save_weights_only=True, save_best_only=False),
            #EarlyStopping(monitor='acc', patience=50000, verbose=1)
            WeightSaver(),
        ]

        imgaug_train = self.train_generator.flow()
        imgaug_valid = self.valid_generator.flow()

        history_tl = self.model.fit_generator(
                generator = imgaug_train,
                initial_epoch = self.epoch,
                epochs = epoch,
                steps_per_epoch = self.steps_per_epoch,
                #validation_data = imgaug_valid,
                #validation_steps = self.valid_generator.len()//self.batch_size,
                validation_data = None,
                verbose = 1,
                callbacks = callbacks)
        self.epoch = max(self.epoch, epoch)
        print("Training finish!")

class DetectTester():
    def __init__(self, weight_dir, input_sample, classes=11):
        self.weight = weight_dir
        self.input = input_sample
        self.classes = classes

    def build_and_load_model(self):
        self.model = detect_model(self.classes)
        self.model.load_weights(self.weight)
    def load_image(self, filename):
        img = keras.preprocessing.image.load_img(filename)
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def handle_test(self, batch_size):
        imgfiles = glob.glob(os.path.join(self.input,"*/*.jpg"))
        predicts = np.zeros((len(imgfiles), self.classes), dtype=np.float32)
        for k,imgpath in enumerate(imgfiles):
            img = self.load_image(imgpath)
            preds = self.model.predict(img)
            predicts[k] = preds[0]

        predicts = np.clip(predicts, 1e-6, 1-1e-6)
        predicts = predicts.reshape(-1, 1)

        picfiles = [fn.split('/')[-1]+'|'+defect for fn in imgfiles for defect in Defect_class]
        picfiles = np.array(picfiles)
        picfiles = picfiles.reshape(-1,1)

        pddata = np.concatenate((picfiles, predicts), axis=1)
        df = pd.DataFrame(pddata, columns=['filename|defect', 'probability'])
        df['probability'] = df['probability'].astype('float32')

        dtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = 'submit-'+ dtime + '.csv'
        df.to_csv(fname, index=False, float_format='%.9f', encoding='utf-8')


import argparse
def args_generator():
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__))

    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="train detect model")
    test_parser = subparsers.add_parser("test", help="check tests images")

    train_parser.add_argument('--work-dir', help='model weights dir', default = './logs_vgg', type=str, metavar='')
    train_parser.add_argument('--gpus', help='wether or which gpus to use', default = '5', type=str, metavar='')
    train_parser.add_argument('--batch-size', help='batch size', default = 12, type=int, metavar='')
    train_parser.add_argument('--epoch', help='epoch', default = 5000, type=int, metavar='')
    train_parser.add_argument('--step', help='step', default = 1000, type=int, metavar='')
    train_parser.add_argument('--classes', help='number of different kind', default = 11, type=int, metavar='')

    test_parser.add_argument('weight', help='Path of inception reset50  model dir', type=str)
    test_parser.add_argument('--input', help='samples dir', default='./data/test', type=str, metavar='')
    test_parser.add_argument('--classes', help='number of different kind', default = 11, type=int, metavar='')
    test_parser.add_argument('--gpus', help='wether or which gpus to use', default = '5', type=str, metavar='')
    test_parser.add_argument('--batch-size', help='batch size', default = 1, type=int, metavar='')
    return parser.parse_args()

if __name__ == '__main__':
    args = args_generator()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.command == "train":
        detectTrainer = DetectTrainer(args.work_dir, args.classes, args.step)
        detectTrainer.samples_preprocess(args.batch_size)
        detectTrainer.build_model()
        detectTrainer.train_model(args.epoch)
    elif args.command == "test":
        detectTester = DetectTester(args.weight, args.input, args.classes, args.step)
        detectTester.build_and_load_model()
        detectTester.handle_test(args.batch_size)
