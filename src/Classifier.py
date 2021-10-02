from keras import Model
from Utils import *
from keras.layers import *
from keras.utils import np_utils
import math

class ClassifierHAR3D(object):

    def __init__(self):

        self.width = None
        self.height = None
        self.channels = None
        self.time = None
        self.batch_size_train = None
        self.batch_size_validation = None
        self.batch_size_test = None
        self.labels = None
        self.pathTrain = None
        self.pathTest = None
        self.pathValidation = None
        self.ftr = None
        self.ftv = None
        self.fts = None
        self.ftr_labels = None
        self.ftv_labels = None
        self.fts_labels = None
        self.predictions = None

    def generatorTrain3D(self):

        while True:

            for count in range(int(len(self.ftr) / self.batch_size_train)):

                batch_start = self.batch_size_train * count
                batch_stop = self.batch_size_train + (self.batch_size_train * count)

                lx1 = list()
                ly = list()

                for i in range(batch_start, batch_stop):

                    if self.ftr[i] != '.ipynb_checkpoints':

                        ly.append(self.ftr_labels[i])

                        optical_flow = extract_videos3D_optical_flow(self.pathTrain + self.ftr[i], self.height,
                                                                     self.width)

                        if len(optical_flow) < self.time:
                            while len(optical_flow) < self.time:
                                optical_flow.append(optical_flow[-1])
                        else:
                            optical_flow = optical_flow[0:self.time]

                        lx1.append(optical_flow)

                x1 = np.array(lx1)
                x1 = x1.astype('float32')
                x1 /= 255
                x1 = x1.reshape(x1.shape[0], self.time, self.height, self.width, self.channels)

                y = np.array(ly)
                y = np_utils.to_categorical(y, len(self.labels))

                yield x1, y

    def generatorTest3D(self):

        while True:

            for count in range(int(len(self.fts) / self.batch_size_test)):

                batch_start = self.batch_size_test * count
                batch_stop = self.batch_size_test + (self.batch_size_test * count)

                lx1 = list()

                for i in range(batch_start, batch_stop):

                    if self.fts[i] != '.ipynb_checkpoints':

                        optical_flow = extract_videos3D_optical_flow(self.pathTest + self.fts[i], self.height,
                                                                     self.width)

                        if len(optical_flow) < self.time:
                            while len(optical_flow) < self.time:
                                optical_flow.append(optical_flow[-1])
                        else:
                            optical_flow = optical_flow[0:self.time]

                        lx1.append(optical_flow)

                x1 = np.array(lx1)
                x1 = x1.astype('float32')
                x1 /= 255
                x1 = x1.reshape(x1.shape[0], self.time, self.height, self.width, self.channels)

                yield x1

    # Models

    def model_dense(self, input_shape):
        input1 = tf.keras.layers.Input(shape=input_shape)

        net = tf.keras.layers.Conv3D(filters=32, kernel_size=(7, 7, 7), strides=(1, 2, 2), activation='relu',
                                     padding='same')(input1)
        net = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(net)

        net1 = tf.keras.layers.BatchNormalization()(net)
        net1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net1)
        net1 = tf.keras.layers.BatchNormalization()(net1)
        net1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net1)

        net2 = tf.keras.layers.BatchNormalization()(net1)
        net2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net2)
        net2 = tf.keras.layers.BatchNormalization()(net2)
        net2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net2)

        c1 = tf.keras.layers.Concatenate()([net1, net2])

        net3 = tf.keras.layers.BatchNormalization()(c1)
        net3 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net3)
        net3 = tf.keras.layers.BatchNormalization()(net3)
        net3 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net3)

        c2 = tf.keras.layers.Concatenate()([net1, net2, net3])

        net4 = tf.keras.layers.BatchNormalization()(c2)
        net4 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net4)
        net4 = tf.keras.layers.BatchNormalization()(net4)
        net4 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net4)

        c3 = tf.keras.layers.Concatenate()([net1, net2, net3, net4])

        net5 = tf.keras.layers.BatchNormalization()(c3)
        net5 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net5)
        net5 = tf.keras.layers.BatchNormalization()(net5)
        net5 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net5)

        c4 = tf.keras.layers.Concatenate()([net1, net2, net3, net4, net5])

        net6 = tf.keras.layers.BatchNormalization()(c4)
        net6 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net6)
        net6 = tf.keras.layers.BatchNormalization()(net6)
        net6 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net6)

        t1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net6)
        t1 = tf.keras.layers.AveragePooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(t1)

        net7 = tf.keras.layers.BatchNormalization()(t1)
        net7 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net7)
        net7 = tf.keras.layers.BatchNormalization()(net7)
        net7 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net7)

        net8 = tf.keras.layers.BatchNormalization()(net7)
        net8 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net8)
        net8 = tf.keras.layers.BatchNormalization()(net8)
        net8 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net8)

        c5 = tf.keras.layers.Concatenate()([net7, net8])

        net9 = tf.keras.layers.BatchNormalization()(c5)
        net9 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net9)
        net9 = tf.keras.layers.BatchNormalization()(net9)
        net9 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net9)

        c6 = tf.keras.layers.Concatenate()([net7, net8, net9])

        net10 = tf.keras.layers.BatchNormalization()(c6)
        net10 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net10)
        net10 = tf.keras.layers.BatchNormalization()(net10)
        net10 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net10)

        c7 = tf.keras.layers.Concatenate()([net7, net8, net9, net10])

        net11 = tf.keras.layers.BatchNormalization()(c7)
        net11 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net11)
        net11 = tf.keras.layers.BatchNormalization()(net11)
        net11 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net11)

        c8 = tf.keras.layers.Concatenate()([net7, net8, net9, net10, net11])

        net12 = tf.keras.layers.BatchNormalization()(c8)
        net12 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net12)
        net12 = tf.keras.layers.BatchNormalization()(net12)
        net12 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net12)

        c9 = tf.keras.layers.Concatenate()([net7, net8, net9, net10, net11, net12])

        net13 = tf.keras.layers.BatchNormalization()(c9)
        net13 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net13)
        net13 = tf.keras.layers.BatchNormalization()(net13)
        net13 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net13)

        c10 = tf.keras.layers.Concatenate()([net7, net8, net9, net10, net11, net12, net13])

        net14 = tf.keras.layers.BatchNormalization()(c10)
        net14 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net14)
        net14 = tf.keras.layers.BatchNormalization()(net14)
        net14 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net14)

        c11 = tf.keras.layers.Concatenate()([net7, net8, net9, net10, net11, net12, net13, net14])

        net15 = tf.keras.layers.BatchNormalization()(c11)
        net15 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net15)
        net15 = tf.keras.layers.BatchNormalization()(net15)
        net15 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net15)

        c12 = tf.keras.layers.Concatenate()([net7, net8, net9, net10, net11, net12, net13, net14, net15])

        net16 = tf.keras.layers.BatchNormalization()(c12)
        net16 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net16)
        net16 = tf.keras.layers.BatchNormalization()(net16)
        net16 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net16)

        c13 = tf.keras.layers.Concatenate()([net7, net8, net9, net10, net11, net12, net13, net14, net15, net16])

        net17 = tf.keras.layers.BatchNormalization()(c13)
        net17 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net17)
        net17 = tf.keras.layers.BatchNormalization()(net17)
        net17 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net17)

        c14 = tf.keras.layers.Concatenate()([net7, net8, net9, net10, net11, net12, net13, net14, net15, net16, net17])

        net18 = tf.keras.layers.BatchNormalization()(c14)
        net18 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net18)
        net18 = tf.keras.layers.BatchNormalization()(net18)
        net18 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net18)

        t2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net18)
        t2 = tf.keras.layers.AveragePooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(t2)

        net19 = tf.keras.layers.BatchNormalization()(t2)
        net19 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net19)
        net19 = tf.keras.layers.BatchNormalization()(net19)
        net19 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net19)

        net20 = tf.keras.layers.BatchNormalization()(net19)
        net20 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net20)
        net20 = tf.keras.layers.BatchNormalization()(net20)
        net20 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net20)

        c15 = tf.keras.layers.Concatenate()([net19, net20])

        net21 = tf.keras.layers.BatchNormalization()(c15)
        net21 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net21)
        net21 = tf.keras.layers.BatchNormalization()(net21)
        net21 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net21)

        c16 = tf.keras.layers.Concatenate()([net19, net20, net21])

        net22 = tf.keras.layers.BatchNormalization()(c16)
        net22 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net22)
        net22 = tf.keras.layers.BatchNormalization()(net22)
        net22 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net22)

        c17 = tf.keras.layers.Concatenate()([net19, net20, net21, net22])

        net23 = tf.keras.layers.BatchNormalization()(c17)
        net23 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net23)
        net23 = tf.keras.layers.BatchNormalization()(net23)
        net23 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net23)

        c18 = tf.keras.layers.Concatenate()([net19, net20, net21, net22, net23])

        net24 = tf.keras.layers.BatchNormalization()(c18)
        net24 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net24)
        net24 = tf.keras.layers.BatchNormalization()(net24)
        net24 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net24)

        c19 = tf.keras.layers.Concatenate()([net19, net20, net21, net22, net23, net24])

        net25 = tf.keras.layers.BatchNormalization()(c19)
        net25 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net25)
        net25 = tf.keras.layers.BatchNormalization()(net25)
        net25 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net25)

        c20 = tf.keras.layers.Concatenate()([net19, net20, net21, net22, net23, net24, net25])

        net26 = tf.keras.layers.BatchNormalization()(c20)
        net26 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net26)
        net26 = tf.keras.layers.BatchNormalization()(net26)
        net26 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net26)

        c21 = tf.keras.layers.Concatenate()([net19, net20, net21, net22, net23, net24, net25, net26])

        net27 = tf.keras.layers.BatchNormalization()(c21)
        net27 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net27)
        net27 = tf.keras.layers.BatchNormalization()(net27)
        net27 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net27)

        c22 = tf.keras.layers.Concatenate()([net19, net20, net21, net22, net23, net24, net25, net26, net27])

        net28 = tf.keras.layers.BatchNormalization()(c22)
        net28 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net28)
        net28 = tf.keras.layers.BatchNormalization()(net28)
        net28 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net28)

        c23 = tf.keras.layers.Concatenate()([net19, net20, net21, net22, net23, net24, net25, net26, net27, net28])

        net29 = tf.keras.layers.BatchNormalization()(c23)
        net29 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net29)
        net29 = tf.keras.layers.BatchNormalization()(net29)
        net29 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net29)

        c24 = tf.keras.layers.Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29])

        net30 = tf.keras.layers.BatchNormalization()(c24)
        net30 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net30)
        net30 = tf.keras.layers.BatchNormalization()(net30)
        net30 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net30)

        c25 = tf.keras.layers.Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30])

        net31 = tf.keras.layers.BatchNormalization()(c25)
        net31 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net31)
        net31 = tf.keras.layers.BatchNormalization()(net31)
        net31 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net31)

        c26 = tf.keras.layers.Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31])

        net32 = tf.keras.layers.BatchNormalization()(c26)
        net32 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net32)
        net32 = tf.keras.layers.BatchNormalization()(net32)
        net32 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net32)

        c27 = tf.keras.layers.Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32])

        net33 = tf.keras.layers.BatchNormalization()(c27)
        net33 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net33)
        net33 = tf.keras.layers.BatchNormalization()(net33)
        net33 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net33)

        c28 = tf.keras.layers.Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33])

        net34 = tf.keras.layers.BatchNormalization()(c28)
        net34 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net34)
        net34 = tf.keras.layers.BatchNormalization()(net34)
        net34 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net34)

        c29 = tf.keras.layers.Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34])

        net35 = tf.keras.layers.BatchNormalization()(c29)
        net35 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net35)
        net35 = tf.keras.layers.BatchNormalization()(net35)
        net35 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net35)

        c30 = tf.keras.layers.Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34, net35])

        net36 = tf.keras.layers.BatchNormalization()(c30)
        net36 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net36)
        net36 = tf.keras.layers.BatchNormalization()(net36)
        net36 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net36)

        c31 = tf.keras.layers.Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34, net35, net36])

        net37 = tf.keras.layers.BatchNormalization()(c31)
        net37 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net37)
        net37 = tf.keras.layers.BatchNormalization()(net37)
        net37 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net37)

        c32 = tf.keras.layers.Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34, net35, net36, net37])

        net38 = tf.keras.layers.BatchNormalization()(c32)
        net38 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net38)
        net38 = tf.keras.layers.BatchNormalization()(net38)
        net38 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net38)

        c33 = tf.keras.layers.Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34, net35, net36, net37, net38])

        net39 = tf.keras.layers.BatchNormalization()(c33)
        net39 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net39)
        net39 = tf.keras.layers.BatchNormalization()(net39)
        net39 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net39)

        c34 = tf.keras.layers.Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34, net35, net36, net37, net38, net39])

        net40 = tf.keras.layers.BatchNormalization()(c34)
        net40 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net40)
        net40 = tf.keras.layers.BatchNormalization()(net40)
        net40 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net40)

        c35 = tf.keras.layers.Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34, net35, net36, net37, net38, net39, net40])

        net41 = tf.keras.layers.BatchNormalization()(c35)
        net41 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net41)
        net41 = tf.keras.layers.BatchNormalization()(net41)
        net41 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net41)

        c36 = tf.keras.layers.Concatenate()(
            [net19, net20, net21, net22, net23, net24, net25, net26, net27, net28, net29, net30, net31, net32, net33,
             net34, net35, net36, net37, net38, net39, net40, net41])

        net42 = tf.keras.layers.BatchNormalization()(c36)
        net42 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net42)
        net42 = tf.keras.layers.BatchNormalization()(net42)
        net42 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net42)

        t3 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net42)
        t3 = tf.keras.layers.AveragePooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(t3)

        net43 = tf.keras.layers.BatchNormalization()(t3)
        net43 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net43)
        net43 = tf.keras.layers.BatchNormalization()(net43)
        net43 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net43)

        net44 = tf.keras.layers.BatchNormalization()(net43)
        net44 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net44)
        net44 = tf.keras.layers.BatchNormalization()(net44)
        net44 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net44)

        c37 = tf.keras.layers.Concatenate()([net43, net44])

        net45 = tf.keras.layers.BatchNormalization()(c37)
        net45 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net45)
        net45 = tf.keras.layers.BatchNormalization()(net45)
        net45 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net45)

        c38 = tf.keras.layers.Concatenate()([net43, net44, net45])

        net46 = tf.keras.layers.BatchNormalization()(c38)
        net46 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net46)
        net46 = tf.keras.layers.BatchNormalization()(net46)
        net46 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net46)

        c39 = tf.keras.layers.Concatenate()([net43, net44, net45, net46])

        net47 = tf.keras.layers.BatchNormalization()(c39)
        net47 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net47)
        net47 = tf.keras.layers.BatchNormalization()(net47)
        net47 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net47)

        c40 = tf.keras.layers.Concatenate()([net43, net44, net45, net46, net47])

        net48 = tf.keras.layers.BatchNormalization()(c40)
        net48 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net48)
        net48 = tf.keras.layers.BatchNormalization()(net48)
        net48 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net48)

        c41 = tf.keras.layers.Concatenate()([net43, net44, net45, net46, net47, net48])

        net49 = tf.keras.layers.BatchNormalization()(c41)
        net49 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net49)
        net49 = tf.keras.layers.BatchNormalization()(net49)
        net49 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net49)

        c42 = tf.keras.layers.Concatenate()([net43, net44, net45, net46, net47, net48, net49])

        net50 = tf.keras.layers.BatchNormalization()(c42)
        net50 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net50)
        net50 = tf.keras.layers.BatchNormalization()(net50)
        net50 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net50)

        c43 = tf.keras.layers.Concatenate()([net43, net44, net45, net46, net47, net48, net49, net50])

        net51 = tf.keras.layers.BatchNormalization()(c43)
        net51 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net51)
        net51 = tf.keras.layers.BatchNormalization()(net51)
        net51 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net51)

        c44 = tf.keras.layers.Concatenate()([net43, net44, net45, net46, net47, net48, net49, net50, net51])

        net52 = tf.keras.layers.BatchNormalization()(c44)
        net52 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net52)
        net52 = tf.keras.layers.BatchNormalization()(net52)
        net52 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net52)

        c45 = tf.keras.layers.Concatenate()([net43, net44, net45, net46, net47, net48, net49, net50, net51, net52])

        net53 = tf.keras.layers.BatchNormalization()(c45)
        net53 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net53)
        net53 = tf.keras.layers.BatchNormalization()(net53)
        net53 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net53)

        c46 = tf.keras.layers.Concatenate()(
            [net43, net44, net45, net46, net47, net48, net49, net50, net51, net52, net53])

        net54 = tf.keras.layers.BatchNormalization()(c46)
        net54 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net54)
        net54 = tf.keras.layers.BatchNormalization()(net54)
        net54 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net54)

        c47 = tf.keras.layers.Concatenate()(
            [net43, net44, net45, net46, net47, net48, net49, net50, net51, net52, net53, net54])

        net55 = tf.keras.layers.BatchNormalization()(c47)
        net55 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net55)
        net55 = tf.keras.layers.BatchNormalization()(net55)
        net55 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net55)

        c48 = tf.keras.layers.Concatenate()(
            [net43, net44, net45, net46, net47, net48, net49, net50, net51, net52, net53, net54, net55])

        net56 = tf.keras.layers.BatchNormalization()(c48)
        net56 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net56)
        net56 = tf.keras.layers.BatchNormalization()(net56)
        net56 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net56)

        c49 = tf.keras.layers.Concatenate()(
            [net43, net44, net45, net46, net47, net48, net49, net50, net51, net52, net53, net54, net55, net56])

        net57 = tf.keras.layers.BatchNormalization()(c49)
        net57 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net57)
        net57 = tf.keras.layers.BatchNormalization()(net57)
        net57 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net57)

        c50 = tf.keras.layers.Concatenate()(
            [net43, net44, net45, net46, net47, net48, net49, net50, net51, net52, net53, net54, net55, net56, net57])

        net58 = tf.keras.layers.BatchNormalization()(c50)
        net58 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net58)
        net58 = tf.keras.layers.BatchNormalization()(net58)
        net58 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net58)

        c51 = tf.keras.layers.Concatenate()(
            [net43, net44, net45, net46, net47, net48, net49, net50, net51, net52, net53, net54, net55, net56, net57,
             net58])

        net59 = tf.keras.layers.BatchNormalization()(c51)
        net59 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(net59)
        net59 = tf.keras.layers.BatchNormalization()(net59)
        net59 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(net59)

        net59 = tf.keras.layers.AveragePooling3D(pool_size=(1, 3, 3))(net59)

        net59 = tf.keras.layers.MultiHeadAttention(num_heads=6, key_dim=int(math.trunc(32/6)), attention_axes=(1, 2, 3))(net59, net59, net59, None)

        net59 = tf.keras.layers.Bidirectional(
            tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(1, 1), activation='relu', return_sequences=True))(
            net59)

        net59 = tf.keras.layers.Flatten()(net59)
        
        net59 = tf.keras.layers.Dense(1024)(net59)
        net59 = tf.keras.layers.Dropout(0.25)(net59)
        net59 = tf.keras.layers.Dense(128)(net59)
        net59 = tf.keras.layers.Dropout(0.25)(net59)
        net59 = tf.keras.layers.Dense(16)(net59)
        net59 = tf.keras.layers.Dropout(0.25)(net59)
        output = tf.keras.layers.Dense(len(self.labels), activation='sigmoid')(net59)

        model = Model(inputs=input1, outputs=output)

        return model
