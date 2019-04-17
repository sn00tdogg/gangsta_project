import numpy as np

from keras import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPooling2D, add, ReLU, BatchNormalization, \
    concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from skimage import feature
from sklearn.model_selection import train_test_split



class CNN:
    def __init__(self):
        self.input_shape = (20, 20, 1)
        self.model = self.network()

    def network(self):
        input_img = Input(shape=self.input_shape)
        input_edges = Input(shape=self.input_shape)

        def residual_layers(input_layer):
            y = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
            y = BatchNormalization()(y)
            y = ReLU()(y)
            y = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
            y = BatchNormalization()(y)
            y = ReLU()(y)
            y = add([y, input_layer])
            y = ReLU()(y)
            return y

        def branches(input_layer):
            y = Conv2D(64, kernel_size=(3, 3), strides=(1, 1))(input_layer)
            y = ReLU()(y)
            # y = Conv2D(64, kernel_size=(3, 3), strides=(2, 2))(y)
            # y = ReLU()(y)
            y = MaxPooling2D(pool_size=(2, 2))(y)
            y = residual_layers(y)
            # y = residual_layers(y)
            # y = residual_layers(y)
            y = MaxPooling2D(pool_size=(2, 2))(y)
            y = BatchNormalization()(y)
            y = ReLU()(y)
            return Flatten()(y)

        img = branches(input_img)
        edg = branches(input_edges)

        x = concatenate([img, edg])
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(26, activation='softmax')(x)

        model = Model(inputs=[input_img, input_edges], outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, img, edges, y):
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        mcp_save = ModelCheckpoint('model_weights.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        callbacks = [early_stopping, mcp_save]
        self.model.fit(x=[img, edges], y=y, epochs=50, batch_size=64, validation_split=0.1, verbose=2,
                       callbacks=callbacks)

        model = load_model('model_weights.hdf5')
        history = model.evaluate(x=[img, edges], y=y)
        print("Train loss: ", history[0], ", train accuracy: ", history[1])

    def train_generator(self, img, edges, y):
        data_generator = ImageDataGenerator(featurewise_center=True,
                                            featurewise_std_normalization=True,
                                            rotation_range=90,
                                            width_shift_range=5,
                                            height_shift_range=5,
                                            horizontal_flip=True,
                                            zca_whitening=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        mcp_save = ModelCheckpoint('model_smooth_weights.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        callbacks = [early_stopping, mcp_save]
        img_train, img_val, y_train, y_val = train_test_split(img, y, test_size=0.1, random_state=42)
        edges_train, edges_val = train_test_split(edges, test_size=0.1, random_state=42)
        data_generator.fit(img_train)
        self.model.fit_generator(data_generator.flow([img_train, edges_train], y_train, batch_size=32),
                                 steps_per_epoch=len(img) / 32, epochs=100, callbacks=callbacks,
                                 validation_data=([img_val, edges_val], y_val), verbose=2)
        model = load_model('model_smooth_weights.hdf5')
        history = model.evaluate(x=[img, edges], y=y)
        print("Train loss: ", history[0], ", train accuracy: ", history[1])

    def test(self, img, edges, y):
        model = load_model('model_smooth_weights.hdf5')
        history = model.evaluate(x=[img, edges], y=y)
        print("Test loss: ", history[0], ", test accuracy: ", history[1])
        return history[1]


def run(img_train, img_test, edges_train, edges_test, y_train, y_test):
    network = CNN()
    network.train(img_train, edges_train, y_train)
    test_accuracy = network.test(img_test, edges_test, y_test)
    return test_accuracy
