import numpy as np
import time
import matplotlib.pyplot as plt

from keras import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPooling2D, add, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

from feature_extraction.data_processing import scale_input, add_dimension, grey_scale, \
    invert_colors, add_pictures_without_chars
from load_data import load_data_chars
import os

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CNN:
    def __init__(self, num_classes, sample, network_type='residual', model_weights=None):
        self.num_classes = num_classes
        self.input_shape = sample.shape
        self.model_weights = model_weights
        if model_weights:
            print('Loading model from ', model_weights, '...')
            self.model = load_model(model_weights)
        else:
            print('Initializing new ', network_type, ' model...')
            if network_type == 'residual':
                self.model = self.residual_network()
            elif network_type == 'simple':
                self.model = self.simple_network()
            else:
                NameError(network_type, ' is not a model.')

    def simple_network(self):
        input_img = Input(shape=self.input_shape)

        #####################
        # Convolution layer #
        #####################

        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input_img)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        ##########################
        # Fully connected layers #
        ##########################

        x = Dense(512, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def residual_network(self):
        input_img = Input(shape=self.input_shape)

        ######################
        # Convolution layers #
        ######################

        def residual_layers(input_layer):
            y = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                       activation='relu')(input_layer)
            y = BatchNormalization()(y)
            y = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                       activation='relu')(y)
            y = add([y, input_layer])
            y = BatchNormalization()(y)
            return y

        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input_img)
        x = BatchNormalization()(x)
        x = residual_layers(x)
        x = residual_layers(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = residual_layers(x)
        x = residual_layers(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)

        ##########################
        # Fully connected layers #
        ##########################

        x = Dense(256, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, x, y, model_weights):
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        mcp_save = ModelCheckpoint(model_weights, save_best_only=True, monitor='val_loss', mode='min')
        callbacks = [early_stopping, mcp_save]
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)
        self.model.fit(x=x_train, y=y_train, epochs=50, batch_size=64, validation_data=[x_val, y_val], verbose=2,
                       callbacks=callbacks)

        model = load_model(model_weights)
        history = model.evaluate(x=x, y=y)
        print("Train loss: ", history[0], ", train accuracy: ", history[1])

    def train_generator(self, x, y, model_weights):
        data_generator = ImageDataGenerator(rotation_range=10,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            width_shift_range=2,
                                            height_shift_range=2,
                                            vertical_flip=True,
                                            preprocessing_function=invert_colors)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
        mcp_save = ModelCheckpoint(model_weights, save_best_only=True, monitor='val_loss', mode='min')
        callbacks = [early_stopping, mcp_save]
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)
        data_generator.fit(x_train)
        self.model.fit_generator(data_generator.flow(x_train, y_train, batch_size=64),
                                 steps_per_epoch=len(x_train) / 64, epochs=200, callbacks=callbacks,
                                 validation_data=(x_val, y_val), verbose=2)

        model = load_model(model_weights)
        history = model.evaluate(x=x, y=y)
        print("Train loss: ", history[0], ", train accuracy: ", history[1])

    def test(self, x, y, model_weights):
        model = load_model(model_weights)
        history = model.evaluate(x=x, y=y)
        print("Test loss: ", history[0], ", test accuracy: ", history[1])
        return history[1]

    def predict_character(self, x):
        prediction = self.model.predict(x=x)
        return prediction

    def plot_training_history(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()


def fit_cnn(x, y, model_weights='model_weights.hdf5', network_type='simple', trials=1):
    print('=== Convolution Neural Network ===')
    test_accuracy = np.zeros(trials)
    running_time = np.zeros(trials)
    x = scale_input(x)
    x = grey_scale(x)
    x = add_dimension(x)
    x, y = add_pictures_without_chars(x, y)
    y = to_categorical(y, int(np.max(y)+1))
    for i in range(trials):
        print('Training network ', i + 1)
        start = time.time()
        random_state = 100 + i
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                            random_state=random_state,
                                                            stratify=y)
        network = CNN(num_classes=len(y[0]), sample=x_train[0], network_type=network_type)
        network.train_generator(x_train, y_train, model_weights)
        test_accuracy[i] = network.test(x_test, y_test, model_weights)
        running_time[i] = time.time() - start
        print('Running time: ', running_time[i])
    print('Average test accuracy over ', trials, ' trials: ', np.mean(test_accuracy))
    print('Average running time over ', trials, ' trials: ', np.mean(running_time))


if __name__ == "__main__":
    img, target = load_data_chars()
    fit_cnn(img, target, trials=1)
