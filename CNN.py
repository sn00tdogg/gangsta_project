import numpy as np

from keras import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPooling2D, add, ReLU
from keras.callbacks import EarlyStopping


class CNN:
    def __init__(self):
        self.input_shape = (20, 20, 3)
        self.model = self.network()

    def network(self):
        input_img = Input(shape=self.input_shape)

        def residual_learning(input_layer):
            y = Conv2D(64, kernel_size=(1, 1), strides=(1, 1))(input_layer)
            y = ReLU()(y)
            y = Conv2D(64, kernel_size=(1, 1), strides=(1, 1))(y)
            y = ReLU()(y)
            y = add([y, input_layer])
            y = ReLU()(y)
            return y

        x = Conv2D(64, kernel_size=(3, 3))(input_img)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = residual_learning(x)
        x = residual_learning(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(26, activation='softmax')(x)

        model = Model(inputs=input_img, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, x, y):
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        callbacks = [early_stopping]
        self.model.fit(x=x, y=y, epochs=50, batch_size=64, validation_split=0.1, verbose=2, callbacks=callbacks)
        history = self.model.evaluate(x=x, y=y)
        print("Train loss: ", history[0], ", train accuracy: ", history[1])

    def test(self, x, y):
        history = self.model.evaluate(x=x, y=y)
        print("Test loss: ", history[0], ", test accuracy: ", history[1])


def run(x_train, x_test, y_train, y_test):
    network = CNN()
    network.train(x_train, y_train)
    network.test(x_test, y_test)
