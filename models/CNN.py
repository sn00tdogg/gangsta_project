import numpy as np
from keras import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPooling2D, add, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from feature_extraction import data_processing
import load_data


class CNN:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.input_shape = (20, 20, 1)
        self.model = self.network()

    def network(self):
        input_img = Input(shape=self.input_shape)

        def residual_layers(input_layer):
            y = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                       activation='relu')(input_layer)
            y = BatchNormalization()(y)
            y = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                       activation='relu')(y)
            y = BatchNormalization()(y)
            y = add([y, input_layer])
            return y

        ######################
        # Convolution layers #
        ######################
        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input_img)
        x = BatchNormalization()(x)
        x = residual_layers(x)
        x = residual_layers(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = residual_layers(x)
        x = residual_layers(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        ##########################
        # Fully connected layers #
        ##########################
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=x)
        # model = multi_gpu_model(model, gpus=8)
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
        data_generator = ImageDataGenerator(rotation_range=20,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            width_shift_range=2,
                                            height_shift_range=2,
                                            preprocessing_function=data_processing.invert_colors)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
        mcp_save = ModelCheckpoint(model_weights, save_best_only=True, monitor='val_loss', mode='min')
        callbacks = [early_stopping, mcp_save]
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)
        data_generator.fit(x_train)
        self.model.fit_generator(data_generator.flow(x_train, y_train, batch_size=64),
                                 steps_per_epoch=len(x_train) / 64, epochs=500, callbacks=callbacks,
                                 validation_data=(x_val, y_val), verbose=2)

        model = load_model(model_weights)
        history = model.evaluate(x=x, y=y)
        print("Train loss: ", history[0], ", train accuracy: ", history[1])

    def test(self, x, y, model_weights):
        model = load_model(model_weights)
        history = model.evaluate(x=x, y=y)
        print("Test loss: ", history[0], ", test accuracy: ", history[1])
        return history[1]

    def predict_character(self, x, model_weights):
        model = load_model(model_weights)
        prediction = model.predict(x=x)
        return prediction


def fit_cnn(x, y, model_weights='model_weights.hdf5', num_classes=26, trials=1):
    print('=== Convolution Neural Network ===')
    test_accuracy = np.zeros(trials)
    y = to_categorical(y, num_classes)
    x = data_processing.scale_input(x)
    x = data_processing.grey_scale(x)
    x = data_processing.add_dimension(x)
    for i in range(trials):
        print('Training network ', i + 1)
        random_state = 100 + i
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                            random_state=random_state,
                                                            stratify=y)
        network = CNN(num_classes)
        network.train_generator(x_train, y_train, model_weights)
        test_accuracy[i] = network.test(x_test, y_test, model_weights)
    print('Average test accuracy over ', trials, ' trials: ', np.mean(test_accuracy))


if __name__ == "__main__":
    img, target = load_data.load_dataset()
    fit_cnn(img, target)
